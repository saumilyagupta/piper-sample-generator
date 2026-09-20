#!/usr/bin/env python3
"""Re-score real microphone captures across input gains and noise floors.

The live mel spectrum sits about 12 dB above the training spectrum with half
its spread, which points at two candidate causes: the recording is simply at a
different level, or the training clips contain artificial digital silence that
no room produces. Scaling the capture tests the first. Adding a matching noise
floor to clean training clips tests the second. Whichever moves the score is
the one worth fixing in the data.
"""
import glob
import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from piper_sample_generator.train import load_model, score_live_window  # noqa: E402

SAMPLE_RATE = 16000
WINDOW = int(1.5 * SAMPLE_RATE)
HOP = int(0.02 * SAMPLE_RATE)

model, data = load_model("wakeword_tiny.pkl")
norm = data["norm"]


def best_score(audio):
    """Highest confidence over a dense sweep of the recording."""
    best = 0.0
    for end in range(WINDOW, len(audio), HOP):
        best = max(best, score_live_window(audio[end - WINDOW:end].copy(), model, norm))
    return best


captures = [librosa.load(p, sr=SAMPLE_RATE)[0] for p in sorted(glob.glob("mic_debug/*.wav"))]

print("real captures, rescored at different input gains")
print(f"{'gain dB':>8} " + " ".join(f"{f'cap{i+1}':>7}" for i in range(len(captures))))
for gain_db in (-18, -12, -6, 0, 6, 12, 18):
    scale = 10 ** (gain_db / 20)
    row = f"{gain_db:8d} "
    for audio in captures:
        row += f"{best_score(np.clip(audio * scale, -1, 1)):7.3f} "
    print(row)

clips = []
for path in sorted(glob.glob("hey_limbo_v4/positive_dg/*.wav"))[:40]:
    clips.append(librosa.load(path, sr=SAMPLE_RATE)[0])

rng = np.random.default_rng(0)
print("\nsynthetic positives with a room-like noise floor added")
print(f"{'floor dB':>9} {'mean conf':>10} {'recall@0.5':>11}")
for floor_db in (-90, -60, -50, -45, -40, -35, -30):
    amp = 10 ** (floor_db / 20)
    scores = []
    for clip in clips:
        noisy = clip + rng.normal(0, amp, len(clip)).astype(np.float32)
        scores.append(score_live_window(noisy, model, norm))
    scores = np.array(scores)
    print(f"{floor_db:9d} {scores.mean():10.3f} {float((scores >= 0.5).mean()):11.3f}")
