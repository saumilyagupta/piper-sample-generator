#!/usr/bin/env python3
"""Test a causal gain control as the fix for level sensitivity.

Real captures score 0.78-0.89 when attenuated 18 dB and 0.40-0.43 at their own
level, so the model is reading absolute loudness rather than only phonetics.
Per-clip peak normalization used to hide that, but it needs the whole clip and
cannot run on a stream.

A running root-mean-square estimate is causal: it depends only on samples
already seen, holds one state variable, and is what a microcontroller frontend
can afford. This measures whether normalizing each window to the training set's
own level restores the score.
"""
import glob
import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from piper_sample_generator.features import HOP_LENGTH, N_FRAMES, SAMPLE_RATE  # noqa: E402
from piper_sample_generator.train import load_model, score_live_window  # noqa: E402

WINDOW = N_FRAMES * HOP_LENGTH
HOP = int(0.02 * SAMPLE_RATE)

model, data = load_model("wakeword_tiny.pkl")
norm = data["norm"]


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)) + 1e-12)


train_rms = []
for path in sorted(glob.glob("hey_limbo_v4/positive_aug/*.wav"))[:400]:
    train_rms.append(rms(librosa.load(path, sr=SAMPLE_RATE)[0]))
train_rms = np.array(train_rms)
target = float(np.median(train_rms))
print(f"training positives: median rms {target:.5f} "
      f"({20 * np.log10(target):.1f} dBFS), "
      f"10th-90th pct {20 * np.log10(np.percentile(train_rms, 10)):.1f} to "
      f"{20 * np.log10(np.percentile(train_rms, 90)):.1f} dBFS")

captures = {Path(p).name: librosa.load(p, sr=SAMPLE_RATE)[0]
            for p in sorted(glob.glob("mic_debug/*.wav"))}
for name, audio in captures.items():
    print(f"{name}: rms {20 * np.log10(rms(audio)):.1f} dBFS")


def best(audio, normalize, max_gain=20.0):
    """Peak confidence over a dense sweep, optionally normalizing each window."""
    top = 0.0
    for end in range(WINDOW, len(audio), HOP):
        chunk = audio[end - WINDOW:end].copy()
        if normalize:
            gain = min(target / rms(chunk), 10 ** (max_gain / 20))
            chunk = np.clip(chunk * gain, -1.0, 1.0)
        top = max(top, score_live_window(chunk, model, norm))
    return top


print(f"\n{'capture':>28} {'as-is':>7} {'rms-normalized':>15}")
for name, audio in captures.items():
    print(f"{name:>28} {best(audio, False):7.3f} {best(audio, True):15.3f}")
