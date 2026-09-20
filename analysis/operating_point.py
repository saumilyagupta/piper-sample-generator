#!/usr/bin/env python3
"""Pick a threshold against both things it trades off, on the same model.

Raising the threshold rejects more confusables and misses more real speech, so
neither number means anything alone. This scores the synthesized rhyme class
and the real microphone captures once each, then reports both across
thresholds, which is the only way the choice can be made rather than guessed.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_confusables import PHRASES  # noqa: E402

WINDOW = int(1.5 * SAMPLE_RATE)
HOP = int(0.1 * SAMPLE_RATE)
REFRACTORY_FRAMES = 10


def detections(scores, threshold, min_run):
    count, blocked, run = 0, -1, 0
    for i, score in enumerate(scores):
        run = run + 1 if score >= threshold else 0
        if run >= min_run and i > blocked:
            count += 1
            blocked = i + REFRACTORY_FRAMES
            run = 0
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--confusable-dir", default="confusables")
    parser.add_argument("--min-run", type=int, default=1)
    args = parser.parse_args()

    model, data = load_model(args.model)
    norm = data["norm"]

    by_phrase = {phrase: [] for phrase in PHRASES}
    for path in sorted(Path(args.confusable_dir).glob("*.wav"),
                       key=lambda p: int(p.stem)):
        by_phrase[PHRASES[int(path.stem) % len(PHRASES)]].append(path)

    phrase_scores = {}
    for phrase, paths in by_phrase.items():
        if paths:
            phrase_scores[phrase] = np.array([
                score_live_window(librosa.load(p, sr=SAMPLE_RATE)[0], model, norm)
                for p in paths
            ])

    capture_scores = {}
    for path in sorted(glob.glob("mic_debug/*.wav")):
        audio = librosa.load(path, sr=SAMPLE_RATE)[0]
        if len(audio) < WINDOW:
            continue
        capture_scores[Path(path).name] = np.array([
            score_live_window(audio[end - WINDOW:end].copy(), model, norm)
            for end in range(WINDOW, len(audio), HOP)
        ])

    rhyme = {p: s for p, s in phrase_scores.items()
             if p not in ("hey limbo", "alexa", "computer")}

    print(f"{args.model}, min run {args.min_run} frame(s), "
          f"{len(rhyme)} confusables, {len(capture_scores)} captures\n")
    print(f"{'thresh':>7} {'wake TTS':>9} {'confusables':>12} {'worst':>7} "
          f"{'captures hit':>13} {'total det':>10}")

    for threshold in (0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
        wake = float((phrase_scores["hey limbo"] >= threshold).mean())
        firing = sum(1 for s in rhyme.values()
                     if float((s >= threshold).mean()) >= 0.5)
        worst = max(float((s >= threshold).mean()) for s in rhyme.values())

        per_capture = [detections(s, threshold, args.min_run)
                       for s in capture_scores.values()]
        hit = sum(1 for n in per_capture if n > 0)

        print(f"{threshold:7.2f} {wake:9.2f} {firing:5d}/{len(rhyme):<6} "
              f"{worst:7.2f} {hit:8d}/{len(per_capture):<4} {sum(per_capture):10d}")


if __name__ == "__main__":
    main()
