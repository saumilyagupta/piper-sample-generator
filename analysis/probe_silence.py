#!/usr/bin/env python3
"""Score models on audio containing no speech at all.

Reported live: high confidence in a silent room. Silence is the one input
whose correct answer is not in doubt, so it is the sharpest test available.

There is a specific reason to suspect it. Positives are now padded to a full
window with Gaussian noise, and a clip averaging 0.84s inside a 1.28s window
means roughly a third of every positive window is that noise. If the model
took the noise texture as evidence for the wake word rather than as filler,
an empty room would score high -- which is what was reported.
"""
import argparse
import glob
import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from piper_sample_generator.features import SAMPLE_RATE, WINDOW_SAMPLES  # noqa: E402
from piper_sample_generator.train import load_model, score_live_window  # noqa: E402

WINDOW = int(1.5 * SAMPLE_RATE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+")
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()

    rng = np.random.default_rng(0)

    cases = {"digital silence": [np.zeros(WINDOW, dtype=np.float32)]}
    for floor_db in (-80, -65, -55, -45, -35, -25):
        amp = 10 ** (floor_db / 20)
        cases[f"white noise {floor_db} dB"] = [
            rng.normal(0, amp, WINDOW).astype(np.float32) for _ in range(args.trials)
        ]

    # The quietest windows of the real captures: whatever the room actually
    # sounds like with nobody speaking, which white noise only approximates.
    quiet = []
    for path in sorted(glob.glob("mic_debug/*.wav")):
        audio = librosa.load(path, sr=SAMPLE_RATE)[0]
        if len(audio) < WINDOW:
            continue
        windows = [audio[e - WINDOW:e] for e in range(WINDOW, len(audio), WINDOW // 2)]
        windows.sort(key=lambda w: float(np.sqrt(np.mean(w ** 2))))
        quiet += windows[:3]
    if quiet:
        cases["quietest real windows"] = quiet[: args.trials]

    for path in args.models:
        model, data = load_model(path)
        norm = data["norm"]
        print(f"\n{path}")
        print(f"{'case':>24} {'n':>4} {'mean':>6} {'max':>6} {'>=0.7':>7}")
        for name, clips in cases.items():
            scores = np.array([score_live_window(c.copy(), model, norm) for c in clips])
            print(f"{name:>24} {len(scores):4d} {scores.mean():6.3f} "
                  f"{scores.max():6.3f} {float((scores >= 0.7).mean()):7.2f}")


if __name__ == "__main__":
    main()
