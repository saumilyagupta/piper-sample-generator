#!/usr/bin/env python3
"""Compare the log-mel statistics of real microphone audio against training audio.

The model scores 0.85+ on synthesized clips and 0.4 on a real voice saying the
same words, so the two differ in a way the features expose. Feature extraction
now uses an absolute dB reference and a frozen per-bin normalization, which
means any systematic offset or tilt between the two domains lands directly on
the model's input and does not wash out. This prints both distributions per
mel band so the shape of the mismatch is visible rather than inferred.
"""
import glob
import random
import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from piper_sample_generator.features import N_MELS, log_mel_spectrogram  # noqa: E402
from piper_sample_generator.train import load_model  # noqa: E402

SAMPLE_RATE = 16000
WINDOW = int(1.5 * SAMPLE_RATE)


def load(path):
    """Read at the feature sample rate, resampling as extract_features does.

    The Piper clips are 22050 Hz on disk, so reading them at their native rate
    and treating the result as 16 kHz would shift every formant upward and
    make the comparison measure a resampling bug instead of the domain gap.
    """
    audio, _ = librosa.load(str(path), sr=SAMPLE_RATE)
    return audio


def voiced_windows(audio, count=40):
    """Windows centred on the loudest parts, i.e. where speech actually is."""
    hop = int(0.1 * SAMPLE_RATE)
    ends = list(range(WINDOW, len(audio), hop))
    energies = [float(np.sqrt(np.mean(audio[e - WINDOW:e] ** 2))) for e in ends]
    order = np.argsort(energies)[::-1][:count]
    return [audio[ends[i] - WINDOW:ends[i]] for i in order]


specs = {}

training = []
random.seed(11)
for kind in ("positive_dg", "positive_piper"):
    files = sorted(glob.glob(f"hey_limbo_v4/{kind}/*.wav"))
    for f in random.sample(files, min(60, len(files))):
        training.append(log_mel_spectrogram(load(f), pad_mode="end"))
specs["training positives"] = np.stack(training)

live = []
for path in sorted(glob.glob("mic_debug/*.wav")):
    for chunk in voiced_windows(load(path), count=25):
        live.append(log_mel_spectrogram(chunk, pad_mode="end"))
specs["live microphone"] = np.stack(live)

_, data = load_model("wakeword_tiny.pkl")
norm_mean = np.array(data["norm"]["mean"])
norm_std = np.array(data["norm"]["std"])

print(f"{'bin':>4} {'Hz':>6} {'train dB':>9} {'live dB':>8} {'delta':>7} {'norm sigma':>11}")
for b in range(N_MELS):
    train_db = specs["training positives"][:, b, :].mean()
    live_db = specs["live microphone"][:, b, :].mean()
    delta = live_db - train_db
    print(f"{b:4d} {'':>6} {train_db:9.1f} {live_db:8.1f} {delta:7.1f} "
          f"{delta / norm_std[b]:11.2f}")

for name, block in specs.items():
    flat = block.reshape(len(block), -1)
    print(f"\n{name}: n={len(block)} mean {flat.mean():.1f} dB  "
          f"std {flat.std():.1f}  min {flat.min():.1f}  max {flat.max():.1f}")

train_all = specs["training positives"].reshape(len(specs["training positives"]), -1)
live_all = specs["live microphone"].reshape(len(specs["live microphone"]), -1)
shift = (live_all.mean() - train_all.mean())
print(f"\noverall live-minus-training offset: {shift:.1f} dB")
print(f"mean per-bin sigma of that offset: "
      f"{np.mean([(specs['live microphone'][:, b, :].mean() - specs['training positives'][:, b, :].mean()) / norm_std[b] for b in range(N_MELS)]):.2f}")
