#!/usr/bin/env python3
"""Score a captured microphone session offline at a chosen cadence.

The server scores at whatever rate the browser sends, so a low live confidence
can mean either the model never fires on this voice or that the aligned window
was never sampled. Re-scoring the same recording at a fine cadence separates
the two: if a dense sweep finds high-confidence windows the live session
missed, the shortfall was sampling, not the model.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from piper_sample_generator.train import load_model, score_live_window  # noqa: E402

SAMPLE_RATE = 16000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("capture")
    parser.add_argument("--window", type=float, default=1.5)
    parser.add_argument("--hop-ms", type=float, default=20.0)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    model, data = load_model(args.model)
    norm = data["norm"]

    audio, rate = sf.read(args.capture, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(1)
    if rate != SAMPLE_RATE:
        raise SystemExit(f"{args.capture} is {rate} Hz, expected {SAMPLE_RATE}")

    window = int(args.window * SAMPLE_RATE)
    hop = max(1, int(args.hop_ms / 1000 * SAMPLE_RATE))

    rms = float(np.sqrt(np.mean(audio ** 2)))
    print(f"{args.capture}")
    print(f"  {len(audio) / SAMPLE_RATE:.1f}s  peak {np.abs(audio).max():.4f}  "
          f"rms {rms:.5f} ({20 * np.log10(rms + 1e-12):.1f} dBFS)")

    scores, times = [], []
    for end in range(window, len(audio) + hop, hop):
        chunk = audio[max(0, end - window):end]
        if len(chunk) < window:
            chunk = np.pad(chunk, (window - len(chunk), 0))
        scores.append(score_live_window(chunk.copy(), model, norm))
        times.append(end / SAMPLE_RATE)

    scores = np.array(scores)
    print(f"  {len(scores)} windows at {args.hop_ms:.0f} ms hop: "
          f"max {scores.max():.3f}  mean {scores.mean():.3f}")
    for cut in (0.3, 0.5, 0.7, 0.9):
        print(f"    windows >= {cut}: {int((scores >= cut).sum())}")

    order = np.argsort(scores)[::-1][: args.top]
    print("  highest-scoring windows (window end, seconds):")
    for i in sorted(order, key=lambda j: times[j]):
        print(f"    t={times[i]:6.2f}s  conf {scores[i]:.3f}")


if __name__ == "__main__":
    main()
