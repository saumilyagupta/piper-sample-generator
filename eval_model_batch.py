#!/usr/bin/env python3
"""Batch-evaluate wake word model on directory of positive/negative samples."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from piper_sample_generator.train import apply_norm, extract_features, load_model


def evaluate(model_file: str, positive_dir: str, negative_dir: str, limit: int = 30):
    model, data = load_model(model_file)
    norm = data["norm"]

    def predict_dir(directory, true_label):
        correct = 0
        total = 0
        for wav in sorted(Path(directory).glob("*.wav"))[:limit]:
            spec = extract_features(wav)
            if spec is None:
                continue
            x = torch.tensor(apply_norm(spec, norm), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                probs = F.softmax(model(x), dim=1)[0]
                pred = int(torch.argmax(probs).item())
            total += 1
            correct += int(pred == true_label)
        return correct, total

    pos_correct, pos_total = predict_dir(positive_dir, 1)
    neg_correct, neg_total = predict_dir(negative_dir, 0)

    print(f"Positive samples: {pos_correct}/{pos_total} correct ({100*pos_correct/max(1,pos_total):.1f}%)")
    print(f"Negative samples: {neg_correct}/{neg_total} correct ({100*neg_correct/max(1,neg_total):.1f}%)")
    print(f"Overall: {pos_correct+neg_correct}/{pos_total+neg_total} correct "
          f"({100*(pos_correct+neg_correct)/max(1,pos_total+neg_total):.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("positive_dir")
    parser.add_argument("negative_dir")
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    evaluate(args.model, args.positive_dir, args.negative_dir, args.limit)
