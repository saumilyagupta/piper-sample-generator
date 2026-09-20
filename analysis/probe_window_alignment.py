"""Score the wake word as it arrives live: embedded in a continuous stream.

Training clips are tight -- the phrase fills the file. A live window is 1.28 s
of a rolling buffer, so the phrase sits at some offset surrounded by whatever
the room is doing, and the buffer is re-scored on a fixed cadence rather than
when the phrase happens to end. This measures both effects: where in the
window the phrase lands, and what fills the rest of it.
"""
import glob, random
import numpy as np, soundfile as sf
from piper_sample_generator.train import load_model, score_live_window

SAMPLE_RATE = 16000
WINDOW = int(1.5 * SAMPLE_RATE)
model, data = load_model("wakeword_tiny.pkl")
norm = data["norm"]

random.seed(3)
clips = []
for f in random.sample(sorted(glob.glob("hey_limbo_v4/positive_dg/*.wav")), 60):
    a, _ = sf.read(f, dtype="float32")
    clips.append(a.mean(1) if a.ndim > 1 else a)

rng = np.random.default_rng(0)


def embed(clip, tail_seconds, floor_db):
    """Phrase followed by tail_seconds of room noise, inside a 1.5 s buffer."""
    amp = 10 ** (floor_db / 20)
    buffer = rng.normal(0, amp, WINDOW).astype(np.float32)
    tail = int(tail_seconds * SAMPLE_RATE)
    end = WINDOW - tail
    start = max(0, end - len(clip))
    buffer[start:end] += clip[-(end - start):]
    return buffer


print("phrase followed by N seconds of room noise, then scored")
print(f"{'tail_s':>7}" + "".join(f"{f'{db}dB':>9}" for db in (-80, -60, -45, -35)))
for tail in (0.0, 0.1, 0.25, 0.5, 0.75):
    row = f"{tail:7.2f}"
    for floor_db in (-80, -60, -45, -35):
        scores = [score_live_window(embed(c, tail, floor_db), model, norm) for c in clips]
        row += f"{np.mean(scores):9.3f}"
    print(row)
