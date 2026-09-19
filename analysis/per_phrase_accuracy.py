#!/usr/bin/env python3
"""Score a trained model phrase by phrase to find which negatives it cannot learn.

The similarity metric in probe_negative_separability.py is a proxy: it predicts
which negatives a model will struggle with. This measures the thing itself. It
runs the trained model over the Deepgram negatives, groups by the phrase that
was spoken, and reports how often each phrase is correctly rejected.

A phrase near 0% rejection is one the model classifies as the wake word every
time. If it is a near-homophone the synthesizer renders identically, that is
not a model failure -- it is a label the audio does not support, and it caps
training accuracy for every architecture. Those are the phrases to drop.

Clip N holds phrase N % len(all_phrases) against the original phrase list the
Deepgram audio was generated from, regardless of which phrases survive now.
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from piper_sample_generator.features import apply_norm, extract_features  # noqa: E402
from piper_sample_generator.train import load_model  # noqa: E402


def score_files(model, norm, files, batch=256):
    """Positive-class probability for each file."""
    specs, kept = [], []
    for path in files:
        spec = extract_features(path, pad_mode="start")
        if spec is not None:
            specs.append(spec)
            kept.append(path)

    if not specs:
        return [], np.zeros(0)

    X = apply_norm(np.stack(specs), norm)
    probs = []
    with torch.no_grad():
        for start in range(0, len(X), batch):
            chunk = torch.tensor(X[start:start + batch], dtype=torch.float32)
            probs.append(torch.softmax(model(chunk), dim=1)[:, 1].numpy())

    return kept, np.concatenate(probs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--dataset", default="hey_limbo_v3")
    parser.add_argument("--original-phrases", default="neg_phrases.txt")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit-per-phrase", type=int, default=53)
    args = parser.parse_args()

    model, data = load_model(args.model)
    norm = data["norm"]
    print(f"model {args.model}  arch={data.get('architecture')} "
          f"test_acc={data.get('test_acc'):.3f}\n")

    all_phrases = [
        p.strip() for p in Path(args.original_phrases).read_text().splitlines() if p.strip()
    ]
    dataset = Path(args.dataset)

    by_phrase = defaultdict(list)
    for path in sorted((dataset / "negative_dg").glob("dg_*.wav")):
        index = int(re.search(r"dg_(\d+)", path.name).group(1))
        phrase = all_phrases[index % len(all_phrases)]
        if len(by_phrase[phrase]) < args.limit_per_phrase:
            by_phrase[phrase].append(path)

    pos_files = sorted((dataset / "positive_dg").glob("dg_*.wav"))
    _, pos_probs = score_files(model, norm, pos_files)
    recall = float((pos_probs >= args.threshold).mean())
    print(f"positives: {len(pos_probs)} clips, recall {recall:.3f}, "
          f"mean confidence {pos_probs.mean():.3f}\n")

    rows = []
    for phrase, files in by_phrase.items():
        _, probs = score_files(model, norm, files)
        if len(probs):
            rows.append((phrase, float((probs < args.threshold).mean()), float(probs.mean())))

    rows.sort(key=lambda r: r[1])

    print(f"{'reject':>7} {'meanconf':>9}  phrase")
    for phrase, reject, mean_conf in rows:
        mark = "  <-- unlearnable" if reject < 0.25 else ""
        print(f"{reject:7.3f} {mean_conf:9.3f}  {phrase}{mark}")

    unlearnable = [r[0] for r in rows if r[1] < 0.25]
    overall = float(np.mean([r[1] for r in rows]))
    print(f"\nmean rejection across phrases: {overall:.3f}")
    print(f"phrases rejected under 25% of the time: {len(unlearnable)}/{len(rows)}")
    if unlearnable:
        print("  " + ", ".join(unlearnable))


if __name__ == "__main__":
    main()
