"""Measure how often the detector must re-score to catch the wake word.

The model only scores high when the phrase ends near the right edge of the
window, so how often the buffer is re-scored decides whether that moment is
ever sampled. Simulates a continuous stream with a real room noise floor.
"""
import glob, random
import numpy as np, soundfile as sf
from piper_sample_generator.train import load_model, score_live_window

SAMPLE_RATE = 16000
WINDOW = int(1.5 * SAMPLE_RATE)
FLOOR_DB = -45

model, data = load_model("wakeword_tiny.pkl")
norm = data["norm"]
rng = np.random.default_rng(0)


def stream_peak(clip, chunk):
    """Peak confidence over a stream of clip + 1.5 s of room noise."""
    amp = 10 ** (FLOOR_DB / 20)
    noise = rng.normal(0, amp, WINDOW).astype(np.float32)
    signal = np.concatenate([noise[: WINDOW // 2], clip + rng.normal(0, amp, len(clip)), noise])
    buffer = np.zeros(WINDOW, dtype=np.float32)
    best = 0.0
    for start in range(0, len(signal), chunk):
        piece = signal[start:start + chunk]
        buffer = np.roll(buffer, -len(piece))
        buffer[-len(piece):] = piece
        best = max(best, score_live_window(buffer.copy(), model, norm))
    return best


random.seed(3)
sets = {}
for kind in ("positive_dg", "negative_dg"):
    files = random.sample(sorted(glob.glob(f"hey_limbo_v4/{kind}/*.wav")), 60)
    clips = []
    for f in files:
        a, _ = sf.read(f, dtype="float32")
        clips.append(a.mean(1) if a.ndim > 1 else a)
    sets[kind] = clips

print(f"{'cadence':>9} {'recall@0.5':>11} {'FA@0.5':>8} {'recall@0.9':>11} {'FA@0.9':>8}")
for chunk, label in ((8000, "500 ms"), (4000, "250 ms"), (1600, "100 ms"), (800, "50 ms")):
    pos = np.array([stream_peak(c, chunk) for c in sets["positive_dg"]])
    neg = np.array([stream_peak(c, chunk) for c in sets["negative_dg"]])
    print(f"{label:>9} {float((pos>=0.5).mean()):11.3f} {float((neg>=0.5).mean()):8.3f}"
          f" {float((pos>=0.9).mean()):11.3f} {float((neg>=0.9).mean()):8.3f}")
