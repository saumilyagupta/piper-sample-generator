#!/usr/bin/env python3
"""CLI wrapper for piper_sample_generator.train.train_model."""

import argparse
import logging

from piper_sample_generator.train import train_model

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train wake word detector (70/15/15 split, best val-loss checkpoint)")
    parser.add_argument("positive_dir", help="Directory with positive samples")
    parser.add_argument("negative_dir", help="Directory with negative samples")
    parser.add_argument("--output", default="wakeword_model.pkl", help="Output model file")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (epochs)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    args = parser.parse_args()

    success = train_model(
        args.positive_dir,
        args.negative_dir,
        args.output,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        batch_size=args.batch_size,
    )
    exit(0 if success else 1)
