#!/usr/bin/env python3
"""Count how much of a recording a model classifies as the wake word.

A high peak confidence only says the model fired somewhere. A detector that
fires on a third of all windows would also do that, while being useless. This
reports the fraction of windows above threshold, which is the quantity that
turns into false accepts per hour once real background audio exists.
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
HOP = int(0.1 * SAMPLE_RATE)


def window_scores(audio, model, norm):
    """Confidence for each window, or empty when the clip is shorter than one."""
    if len(audio) < WINDOW:
        return np.zeros(0)

    return np.array([
        score_live_window(audio[end - WINDOW:end].copy(), model, norm)
        for end in range(WINDOW, len(audio), HOP)
    ])


def detection_events(scores, threshold, refractory_windows):
    """Collapse runs of above-threshold windows into one detection each.

    A single spoken phrase stays inside the window for several hops, so raw
    window counts overstate detections by that factor. Firmware has the same
    problem and solves it the same way: after firing, ignore further windows
    for a short refractory period.
    """
    events, blocked_until = [], -1
    for i, score in enumerate(scores):
        if score >= threshold and i > blocked_until:
            events.append(i)
            blocked_until = i + refractory_windows
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    captures = {Path(p).name: librosa.load(p, sr=SAMPLE_RATE)[0]
                for p in sorted(glob.glob("mic_debug/*.wav"))}

    for path in args.models:
        model, data = load_model(path)
        norm = data["norm"]
        print(f"\n{path}")
        for name, audio in captures.items():
            scores = window_scores(audio, model, norm)
            duration = len(audio) / SAMPLE_RATE
            if not len(scores):
                print(f"  {name}  {duration:5.1f}s  shorter than the window, skipped")
                continue
            # 1.0 s refractory: longer than the phrase, shorter than a
            # plausible gap between two deliberate wake word utterances.
            events = detection_events(scores, args.threshold, refractory_windows=10)
            times = ", ".join(f"{i * HOP / SAMPLE_RATE + 1.5:.1f}s" for i in events)
            print(f"  {name}  {duration:5.1f}s  {len(scores):4d} windows  "
                  f"max {scores.max():.3f}  "
                  f"{len(events)} detections at [{times}]")


if __name__ == "__main__":
    main()
