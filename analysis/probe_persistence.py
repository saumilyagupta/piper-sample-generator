#!/usr/bin/env python3
"""Measure how long a detection stays above threshold.

A spoken phrase sits inside the rolling window for several consecutive frames,
so a real detection should hold. A window that happens to resemble the wake
word need not. If the two differ in run length, requiring a run is a stronger
filter than raising the threshold, because it discards spurious frames without
making genuine speech harder to reach.

This also reports what each requirement costs on real captures, so the setting
is chosen against measurement rather than picked.
"""
import argparse
import glob
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


def frame_scores(audio, model, norm):
    return np.array([
        score_live_window(audio[end - WINDOW:end].copy(), model, norm)
        for end in range(WINDOW, len(audio), HOP)
    ])


def runs_above(scores, threshold):
    """Lengths of each maximal run of consecutive frames above threshold."""
    lengths, current = [], 0
    for score in scores:
        if score >= threshold:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def detections(scores, threshold, min_run, refractory_frames):
    """Count detections requiring `min_run` consecutive frames above threshold."""
    count, blocked_until, current = 0, -1, 0
    for i, score in enumerate(scores):
        current = current + 1 if score >= threshold else 0
        if current >= min_run and i > blocked_until:
            count += 1
            blocked_until = i + refractory_frames
            current = 0
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--refractory", type=float, default=1.0)
    args = parser.parse_args()

    model, data = load_model(args.model)
    norm = data["norm"]
    refractory_frames = round(args.refractory / FRAME_SECONDS)

    captures = {Path(p).name: librosa.load(p, sr=SAMPLE_RATE)[0]
                for p in sorted(glob.glob("mic_debug/*.wav"))}

    print(f"{args.model}, threshold {args.threshold}, "
          f"{FRAME_SECONDS * 1000:.0f} ms frames\n")

    all_runs = []
    print(f"{'capture':>26} {'secs':>6} {'runs above threshold (frames)':>32}")
    for name, audio in captures.items():
        scores = frame_scores(audio, model, norm)
        lengths = runs_above(scores, args.threshold)
        all_runs += lengths
        shown = ", ".join(str(n) for n in lengths[:12]) or "none"
        print(f"{name:>26} {len(audio) / SAMPLE_RATE:6.1f} {shown:>32}")

    if all_runs:
        runs = np.array(all_runs)
        print(f"\n{len(runs)} runs: median {np.median(runs):.0f} frames "
              f"({np.median(runs) * FRAME_SECONDS:.1f}s), "
              f"{int((runs == 1).sum())} are a single frame")

    print(f"\ndetections per capture as the run requirement rises")
    header = "  ".join(f"{f'>={n}':>4}" for n in (1, 2, 3, 4, 5, 6))
    print(f"{'capture':>26}  {header}")
    for name, audio in captures.items():
        scores = frame_scores(audio, model, norm)
        row = "  ".join(
            f"{detections(scores, args.threshold, n, refractory_frames):4d}"
            for n in (1, 2, 3, 4, 5, 6)
        )
        print(f"{name:>26}  {row}")


if __name__ == "__main__":
    main()
