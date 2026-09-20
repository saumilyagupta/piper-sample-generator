#!/usr/bin/env python3
"""Score models against a capture known to hold only the wake word and silence.

Every other measurement here has used a proxy. A recording whose contents are
known gives the two numbers that actually matter: how many spoken instances
are detected, and how often the detector fires when nobody is speaking. The
second is the false-accept rate, which no synthetic negative can stand in for.

Speech is located by energy rather than by the model, so the ground truth does
not depend on the thing being measured.
"""
import argparse
import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from piper_sample_generator.features import SAMPLE_RATE  # noqa: E402
from piper_sample_generator.train import load_model, score_live_window  # noqa: E402

WINDOW = int(1.5 * SAMPLE_RATE)
FRAME_SECONDS = 0.1
HOP = int(FRAME_SECONDS * SAMPLE_RATE)


def speech_regions(audio, frame_ms=25, margin_db=12.0, min_ms=200, join_ms=300):
    """Start and end times of speech, from energy against the noise floor.

    The floor is the 20th percentile of frame energy, which a recording that
    is mostly silence supplies reliably. Regions closer together than join_ms
    are merged, so a pause inside one phrase does not split it in two.
    """
    frame = int(frame_ms / 1000 * SAMPLE_RATE)
    frames = [audio[i:i + frame] for i in range(0, len(audio) - frame, frame)]
    energy = np.array([20 * np.log10(np.sqrt(np.mean(f ** 2)) + 1e-12) for f in frames])

    floor = np.percentile(energy, 20)
    loud = energy > floor + margin_db

    regions, start = [], None
    for i, is_loud in enumerate(loud):
        if is_loud and start is None:
            start = i
        elif not is_loud and start is not None:
            regions.append((start * frame_ms / 1000, i * frame_ms / 1000))
            start = None
    if start is not None:
        regions.append((start * frame_ms / 1000, len(loud) * frame_ms / 1000))

    merged = []
    for begin, end in regions:
        if merged and begin - merged[-1][1] < join_ms / 1000:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((begin, end))

    return [(b, e) for b, e in merged if e - b >= min_ms / 1000], floor


def detections(scores, threshold, refractory_frames=10):
    events, blocked = [], -1
    for i, score in enumerate(scores):
        if score >= threshold and i > blocked:
            events.append(i * FRAME_SECONDS + 1.5)
            blocked = i + refractory_frames
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture")
    parser.add_argument("models", nargs="+")
    args = parser.parse_args()

    audio = librosa.load(args.capture, sr=SAMPLE_RATE)[0]
    regions, floor = speech_regions(audio)
    duration = len(audio) / SAMPLE_RATE
    speech_seconds = sum(e - b for b, e in regions)

    print(f"{Path(args.capture).name}: {duration:.1f}s, noise floor {floor:.1f} dB")
    print(f"{len(regions)} spoken instances, {speech_seconds:.1f}s of speech, "
          f"{duration - speech_seconds:.1f}s of silence")
    print("  at " + ", ".join(f"{b:.1f}-{e:.1f}s" for b, e in regions))

    # A detection is credited to an utterance if it lands inside it or within
    # the window that still contains it, since the phrase stays in the buffer
    # for as long as the window is wide.
    def attribute(times):
        hit = set()
        spurious = 0
        for t in times:
            matched = None
            for i, (begin, end) in enumerate(regions):
                if begin <= t <= end + 1.4:
                    matched = i
                    break
            if matched is None:
                spurious += 1
            else:
                hit.add(matched)
        return len(hit), spurious

    for path in args.models:
        model, data = load_model(path)
        norm = data["norm"]
        scores = np.array([
            score_live_window(audio[end - WINDOW:end].copy(), model, norm)
            for end in range(WINDOW, len(audio), HOP)
        ])

        print(f"\n{path}")
        print(f"{'thresh':>7} {'found':>7} {'recall':>7} {'false':>6} "
              f"{'FA/hour':>8}  {'peak on silence':>15}")

        silent_frames = [
            i for i in range(len(scores))
            if not any(b <= i * FRAME_SECONDS + 1.5 <= e + 1.4 for b, e in regions)
        ]
        silence_peak = max((scores[i] for i in silent_frames), default=0.0)
        silence_seconds = len(silent_frames) * FRAME_SECONDS

        for threshold in (0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80):
            found, spurious = attribute(detections(scores, threshold))
            rate = spurious / silence_seconds * 3600 if silence_seconds else 0.0
            print(f"{threshold:7.2f} {found:3d}/{len(regions):<3} "
                  f"{found / max(1, len(regions)):7.2f} {spurious:6d} {rate:8.0f}"
                  f"  {silence_peak:15.3f}")


if __name__ == "__main__":
    main()
