#!/usr/bin/env python3
"""Test whether the model depends on the digital silence used to pad windows.

Short clips are padded to a full window in the spectrogram domain with a fixed
-80 dB floor. No microphone produces -80 dB, so if the model leans on that
silence as evidence for the wake word it will score high on padded training
clips and low on the same phrase surrounded by a real room floor. Padding the
same clips in the time domain with low-level noise instead isolates the effect.
"""
import glob
import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from piper_sample_generator.features import (  # noqa: E402
    HOP_LENGTH, N_FRAMES, SAMPLE_RATE, apply_norm, log_mel_spectrogram,
)
from piper_sample_generator.train import load_model  # noqa: E402

import torch  # noqa: E402

WINDOW_SAMPLES = N_FRAMES * HOP_LENGTH

model, data = load_model("wakeword_tiny.pkl")
norm = data["norm"]


def score(spec):
    x = torch.tensor(apply_norm(spec, norm), dtype=torch.float32).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        return float(torch.softmax(model(x), dim=1)[0, 1])


def pad_with_noise(clip, floor_db, rng):
    """Centre the clip in a full window of low-level noise, in the time domain."""
    amp = 10 ** (floor_db / 20)
    buffer = rng.normal(0, amp, WINDOW_SAMPLES).astype(np.float32)
    if len(clip) >= WINDOW_SAMPLES:
        return clip[-WINDOW_SAMPLES:]
    start = (WINDOW_SAMPLES - len(clip)) // 2
    buffer[start:start + len(clip)] += clip
    return buffer


clips, durations = [], []
for path in sorted(glob.glob("hey_limbo_v4/positive_dg/*.wav"))[:60]:
    audio = librosa.load(path, sr=SAMPLE_RATE)[0]
    clips.append(audio)
    durations.append(len(audio) / SAMPLE_RATE)

print(f"{len(clips)} clips, duration mean {np.mean(durations):.2f}s "
      f"(window is {WINDOW_SAMPLES / SAMPLE_RATE:.2f}s)")
short = sum(1 for d in durations if d < WINDOW_SAMPLES / SAMPLE_RATE)
print(f"{short}/{len(clips)} are shorter than the window and get padded\n")

rng = np.random.default_rng(0)
padded = np.array([score(log_mel_spectrogram(c, pad_mode="center")) for c in clips])
print(f"spectrogram padding at -80 dB (as trained): mean {padded.mean():.3f} "
      f"recall@0.5 {float((padded >= 0.5).mean()):.3f}")

print(f"\n{'floor dB':>9} {'mean conf':>10} {'recall@0.5':>11}")
for floor_db in (-80, -65, -55, -50, -45, -40, -35):
    scores = np.array([
        score(log_mel_spectrogram(pad_with_noise(c, floor_db, rng), pad_mode="center"))
        for c in clips
    ])
    print(f"{floor_db:9d} {scores.mean():10.3f} {float((scores >= 0.5).mean()):11.3f}")
