#!/usr/bin/env python3
"""Score models against real microphone captures and synthetic held-out clips.

The captures are the only real-voice audio in this project, so they are the
only evidence that separates a model that works from one that merely scores
well on the synthetic set it was trained on. Live windows are already longer
than the feature window, so padding never applies to them and the comparison
across models is unaffected by the padding change.
"""
import argparse
import glob
import random
import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from piper_sample_generator.features import HOP_LENGTH, N_FRAMES, SAMPLE_RATE  # noqa: E402
from piper_sample_generator.train import load_model, score_live_window  # noqa: E402

WINDOW = int(1.5 * SAMPLE_RATE)
SWEEP_HOP = int(0.02 * SAMPLE_RATE)


def peak_over_recording(audio, model, norm):
    """Highest confidence any window of the recording reaches."""
    best = 0.0
    for end in range(WINDOW, len(audio), SWEEP_HOP):
        best = max(best, score_live_window(audio[end - WINDOW:end].copy(), model, norm))
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+")
    parser.add_argument("--dataset", default="hey_limbo_v4")
    parser.add_argument("--clips", type=int, default=80)
    args = parser.parse_args()

    captures = {Path(p).name: librosa.load(p, sr=SAMPLE_RATE)[0]
                for p in sorted(glob.glob("mic_debug/*.wav"))}

    random.seed(5)
    synthetic = {}
    for kind in ("positive_dg", "negative_dg"):
        files = sorted(glob.glob(f"{args.dataset}/{kind}/*.wav"))
        synthetic[kind] = [librosa.load(f, sr=SAMPLE_RATE)[0]
                           for f in random.sample(files, min(args.clips, len(files)))]

    for path in args.models:
        model, data = load_model(path)
        norm = data["norm"]
        print(f"\n{path}  arch={data.get('architecture')} "
              f"test_acc={data.get('test_acc', 0):.3f}")

        print("  real microphone captures (peak confidence over a 20 ms sweep):")
        for name, audio in captures.items():
            print(f"    {name}  {peak_over_recording(audio, model, norm):.3f}")

        for kind, clips in synthetic.items():
            scores = np.array([score_live_window(c.copy(), model, norm) for c in clips])
            fired = float((scores >= 0.5).mean())
            print(f"  synthetic {kind:12s} n={len(scores)} "
                  f"mean {scores.mean():.3f}  >=0.5 {fired:.3f}")


if __name__ == "__main__":
    main()
