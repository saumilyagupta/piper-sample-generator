#!/usr/bin/env python3
"""Measure how separable each generated negative phrase is from the wake word.

Background: an earlier measurement (neg_sims.json) concluded that 76 of 87
generated negatives were "indistinguishable" from "hey limbo". That conclusion
rested on a metric that does not survive scrutiny. It scored `what time is it`
and `alexa` as being as close to the wake word as `hey linbo` is, which cannot
be true of any metric that responds to what was actually said.

Two faults, both reproduced and fixed here:

  1. Raw cosine similarity on log-mel dB vectors. Those values are all negative
     and share a large common offset, so every pair scores above 0.99 and
     nothing is resolved. Fix: mean-center before the cosine, i.e. correlation.

  2. Pooling over the time axis, collapsing 40xT to 40 dims. That discards
     phoneme order, leaving only the spectral envelope -- which is near
     identical across clips from the same TTS engines regardless of content.
     Fix: resample the time axis to a fixed length and keep it.

Run `--probe` to see both metrics side by side on phrases whose true
relationship to the wake word is not in doubt. Run `--all` to rescore every
negative phrase with the corrected metric and write the result to JSON.
"""

import argparse
import json
import re
from pathlib import Path

import librosa
import numpy as np

SAMPLE_RATE = 16000
N_MELS = 40
N_FFT = 400
HOP_LENGTH = 160
FIXED_FRAMES = 64

# Phrases whose true relationship to the wake word is unambiguous, used to
# check that a metric responds to content at all.
PROBES = [
    ("hey linbo", "near-homophone"),
    ("hey lilbo", "near-homophone"),
    ("hay limbo", "near-homophone"),
    ("limbo", "partial"),
    ("hey", "partial"),
    ("alexa", "unrelated"),
    ("jarvis", "unrelated"),
    ("computer", "unrelated"),
    ("cortana", "unrelated"),
    ("thank you", "unrelated"),
    ("what time is it", "unrelated"),
]


def load_logmel(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE)
    if audio.size < 512:
        return None
    trimmed, _ = librosa.effects.trim(audio, top_db=15.0)
    if trimmed.size >= 512:
        audio = trimmed
    peak = float(np.abs(audio).max())
    if peak > 1e-6:
        audio = audio / peak
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=20, fmax=8000,
    )
    return librosa.power_to_db(mel, ref=np.max, top_db=80.0)


def time_pooled(spec):
    """The metric under suspicion: collapses the time axis entirely."""
    return spec.mean(axis=1)


def temporal(spec):
    """Keeps phoneme order by resampling the time axis to a fixed length."""
    src = np.linspace(0.0, 1.0, spec.shape[1])
    dst = np.linspace(0.0, 1.0, FIXED_FRAMES)
    return np.stack([np.interp(dst, src, row) for row in spec]).flatten()


def correlation(a, b):
    """Cosine on mean-centered vectors, so the shared dB offset cancels."""
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def collect(files, embed):
    vecs = []
    for path in files:
        spec = load_logmel(path)
        if spec is not None:
            vecs.append(embed(spec))
    return vecs


def phrase_files(directory, prefix, phrase_index, n_phrases, limit):
    """Deepgram jobs were enumerated as [(phrase, voice) for voice for phrase],
    so file index N holds phrase N % n_phrases."""
    out = []
    for path in sorted(directory.glob(f"{prefix}_*.wav")):
        index = int(re.search(rf"{prefix}_(\d+)", path.name).group(1))
        if index % n_phrases == phrase_index:
            out.append(path)
            if len(out) >= limit:
                break
    return out


def load_phrases(path):
    return [p.strip() for p in Path(path).read_text().splitlines() if p.strip()]


def positive_reference(root, embed, per_phrase):
    """Return (reference vector, baseline), where baseline is the wake word's
    similarity to itself measured across two disjoint halves of its voices."""
    files = sorted((root / "positive").glob("dg_*.wav"))[: per_phrase * 2]
    vecs = collect(files, embed)
    half = len(vecs) // 2
    baseline = correlation(np.mean(vecs[:half], axis=0), np.mean(vecs[half:], axis=0))
    return np.mean(vecs, axis=0), baseline


def run_probe(args):
    root = Path(args.dataset)
    phrases = load_phrases(args.neg_phrases)
    n_phrases = len(phrases)
    print(f"{n_phrases} negative phrases, {args.per_phrase} clips each\n")

    for name, embed in (("time_pooled (suspect)", time_pooled),
                        ("temporal (corrected)", temporal)):
        reference, baseline = positive_reference(root, embed, args.per_phrase)
        print(f"--- {name} ---")
        print(f"positive self-similarity (baseline): {baseline:.4f}")

        rows = []
        for phrase, kind in PROBES:
            if phrase not in phrases:
                continue
            vecs = collect(
                phrase_files(root / "negative", "dg", phrases.index(phrase),
                             n_phrases, args.per_phrase),
                embed,
            )
            if vecs:
                rows.append((phrase, kind, correlation(reference, np.mean(vecs, axis=0))))

        for phrase, kind, score in sorted(rows, key=lambda r: -r[2]):
            flag = "ABOVE baseline" if score >= baseline else ""
            print(f"  {score:.4f}  {phrase:<18} [{kind}] {flag}")
        print()


def run_all(args):
    root = Path(args.dataset)
    phrases = load_phrases(args.neg_phrases)
    n_phrases = len(phrases)

    reference, baseline = positive_reference(root, temporal, args.per_phrase)
    print(f"temporal baseline (positive self-similarity): {baseline:.4f}")

    sims = {}
    for i, phrase in enumerate(phrases):
        vecs = collect(
            phrase_files(root / "negative", "dg", i, n_phrases, args.per_phrase),
            temporal,
        )
        if vecs:
            sims[phrase] = correlation(reference, np.mean(vecs, axis=0))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{n_phrases}")

    keep = sorted((k for k, v in sims.items() if v < baseline), key=lambda k: sims[k])
    drop = sorted((k for k, v in sims.items() if v >= baseline), key=lambda k: -sims[k])

    Path(args.out).write_text(json.dumps(
        {"metric": "temporal", "baseline": baseline, "sims": sims}, indent=2))

    print(f"\nKEEP (separable, below baseline): {len(keep)}")
    for k in keep:
        print(f"  {sims[k]:.4f}  {k}")
    print(f"\nDROP (at or above baseline, unlearnable): {len(drop)}")
    for k in drop:
        print(f"  {sims[k]:.4f}  {k}")
    print(f"\n{len(drop)}/{len(sims)} negatives unlearnable -> wrote {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="hey_limbo_v2")
    parser.add_argument("--neg-phrases", default="neg_phrases.txt")
    parser.add_argument("--per-phrase", type=int, default=16)
    parser.add_argument("--probe", action="store_true",
                        help="Compare both metrics on known-relationship phrases")
    parser.add_argument("--all", action="store_true",
                        help="Rescore every negative with the corrected metric")
    parser.add_argument("--out", default="neg_sims_temporal.json")
    args = parser.parse_args()

    if args.all:
        run_all(args)
    else:
        run_probe(args)


if __name__ == "__main__":
    main()
