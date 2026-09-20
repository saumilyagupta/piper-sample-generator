#!/usr/bin/env python3
"""Score a model on rhyming confusables it was never trained to reject.

Reported live: "bimbo", "simbo" and "timbo" fire the detector. The negative
set explains it -- the substitution map offered l -> r, n, d only, so the whole
"hey _imbo" rhyme class was absent from training. This synthesizes the class
and measures how far from rejection the model actually is, which decides
whether adding the phrases is enough or whether they are inseparable in
synthesized audio and need real speech.

Also reports how each phrase's audio differs from the wake word's, so a phrase
that scores badly can be told apart from one the synthesizer renders
identically -- a label the audio does not support cannot be trained on.
"""
import argparse
import shutil
import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from piper_sample_generator.features import SAMPLE_RATE, log_mel_spectrogram  # noqa: E402
from piper_sample_generator.train import load_model, score_live_window  # noqa: E402

PHRASES = [
    "hey limbo",
    "hey bimbo", "hey simbo", "hey timbo", "hey kimbo", "hey jimbo",
    "hey nimbo", "hey rimbo", "hey dimbo", "hey gimbo", "hey pimbo",
    "hey mimbo", "hey wimbo", "hey fimbo", "hey vimbo", "hey zimbo",
    "hey himbo", "hey shimbo", "hey thimbo",
    "hey lambo", "hey lumbo", "hey lembo",
    "alexa", "computer",
]


def temporal_signature(spec: np.ndarray, frames: int = 64) -> np.ndarray:
    """Flatten a spectrogram to a fixed length, keeping phoneme order.

    Pooling the time axis away would rate "limbo hey" as identical to
    "hey limbo", so the axis is resampled rather than collapsed.
    """
    source = np.linspace(0.0, 1.0, spec.shape[1])
    target = np.linspace(0.0, 1.0, frames)
    return np.stack([np.interp(target, source, row) for row in spec]).ravel()


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity on mean-centred vectors, cancelling the dB offset."""
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def synthesize(phrases, out_dir: Path, model_path: str, per_phrase: int) -> None:
    from piper_sample_generator.__main__ import generate_samples

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    generate_samples(
        text=phrases,
        output_dir=str(out_dir),
        model=model_path,
        max_samples=per_phrase * len(phrases),
        batch_size=16,
        slerp_weights=(0.5,),
        length_scales=(0.9, 1.0, 1.15),
        noise_scales=(0.667, 0.85),
        noise_scale_ws=(0.8,),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--voice", default="models/en-us-libritts-high.pt")
    parser.add_argument("--out-dir", default="confusables")
    parser.add_argument("--per-phrase", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--skip-synth", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not args.skip_synth:
        synthesize(PHRASES, out_dir, args.voice, args.per_phrase)

    # The generator enumerates jobs phrase-major within each voice, so clip N
    # holds phrase N % len(PHRASES).
    files = sorted(out_dir.glob("*.wav"), key=lambda p: int(p.stem))
    by_phrase = {phrase: [] for phrase in PHRASES}
    for path in files:
        by_phrase[PHRASES[int(path.stem) % len(PHRASES)]].append(path)

    model, data = load_model(args.model)
    norm = data["norm"]

    audio = {
        phrase: [librosa.load(p, sr=SAMPLE_RATE)[0] for p in paths]
        for phrase, paths in by_phrase.items()
    }

    reference = np.mean(
        [temporal_signature(log_mel_spectrogram(a, pad_mode="end"))
         for a in audio["hey limbo"]],
        axis=0,
    )

    print(f"{args.model}, {args.per_phrase} clips per phrase, "
          f"threshold {args.threshold}\n")
    print(f"{'phrase':>12} {'n':>3} {'mean':>6} {'fires':>6} {'audio sim':>10}")

    rows = []
    for phrase, clips in audio.items():
        if not clips:
            continue
        scores = np.array([score_live_window(c.copy(), model, norm) for c in clips])
        sims = [
            correlation(temporal_signature(log_mel_spectrogram(c, pad_mode="end")),
                        reference)
            for c in clips
        ]
        rows.append((phrase, len(clips), scores.mean(),
                     float((scores >= args.threshold).mean()), float(np.mean(sims))))

    target = next(r for r in rows if r[0] == "hey limbo")
    for phrase, n, mean, fires, sim in sorted(rows, key=lambda r: -r[3]):
        flag = ""
        if phrase != "hey limbo":
            if sim >= target[4]:
                flag = "  <-- audio matches the wake word"
            elif fires >= 0.5:
                flag = "  <-- fires, but audio differs: trainable"
        print(f"{phrase:>12} {n:3d} {mean:6.3f} {fires:6.2f} {sim:10.3f}{flag}")

    print(f"\nwake word self-similarity: {target[4]:.3f}")
    others = [r for r in rows if r[0] != "hey limbo"]
    print(f"confusables firing at >= {args.threshold}: "
          f"{sum(1 for r in others if r[3] >= 0.5)}/{len(others)}")


if __name__ == "__main__":
    main()
