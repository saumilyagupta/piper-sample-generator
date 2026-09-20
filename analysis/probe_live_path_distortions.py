"""Reproduce, offline, each way the browser path can differ from training audio.

Live confidence never exceeding 0.4 while synthetic clips reach 0.95 recall
means the audio reaching the model is not the audio it was trained on. Each
case below applies one suspected distortion to clips the model already scores
correctly, so whichever one collapses the score is the one to fix.
"""
import glob, random
import numpy as np, soundfile as sf
from scipy.signal import resample_poly
from piper_sample_generator.train import load_model, score_live_window

model, data = load_model("wakeword_tiny.pkl")
norm = data["norm"]


def js_resample(x, in_rate, out_rate):
    """The client's resampler: linear interpolation, no anti-alias filter."""
    ratio = in_rate / out_rate
    out_len = int(len(x) / ratio)
    pos = np.arange(out_len) * ratio
    idx = np.floor(pos).astype(int)
    frac = pos - idx
    nxt = np.minimum(idx + 1, len(x) - 1)
    return (x[idx] + (x[nxt] - x[idx]) * frac).astype(np.float32)


def roundtrip_48k(x):
    """16k -> 48k (as a real mic would deliver) -> 16k via the client's resampler."""
    up = resample_poly(x, 48000, 16000).astype(np.float32)
    return js_resample(up, 48000, 16000)


def quantize_window(x, step=8000):
    """Only score on 0.5 s boundaries, as samplesPerSend=8000 forces."""
    return x[: (len(x) // step) * step] if len(x) >= step else x


CASES = {
    "baseline": lambda x: x,
    "48k roundtrip (client resampler)": roundtrip_48k,
    "gain -20 dB": lambda x: x * 0.1,
    "gain -34 dB": lambda x: x * 0.02,
    "gain +12 dB clipped": lambda x: np.clip(x * 4, -1, 1),
    "white noise at -30 dB added": lambda x: x + np.random.default_rng(0).normal(0, 0.03, len(x)).astype(np.float32),
    "0.5 s send quantization": quantize_window,
}

random.seed(3)
files = random.sample(sorted(glob.glob("hey_limbo_v4/positive_dg/*.wav")), 60)
clips = []
for f in files:
    a, _ = sf.read(f, dtype="float32")
    clips.append(a.mean(1) if a.ndim > 1 else a)

print(f"{'mean':>6} {'recall':>7}  case")
for name, fn in CASES.items():
    scores = np.array([score_live_window(fn(c.copy()), model, norm) for c in clips])
    print(f"{scores.mean():6.3f} {float((scores>=0.5).mean()):7.3f}  {name}")
