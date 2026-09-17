#!/usr/bin/env python3
"""Simple wake word detector model training."""

import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
import librosa
import pickle

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)


def extract_mfcc_features(wav_path: str, n_mfcc: int = 13, max_frames: int = 100) -> np.ndarray:
    """Extract MFCC features from audio file."""
    try:
        y, sr = librosa.load(wav_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        # Pad or truncate to consistent shape
        if mfcc.shape[1] < max_frames:
            mfcc = np.pad(mfcc, ((0, 0), (0, max_frames - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :max_frames]

        return mfcc.flatten()
    except Exception as e:
        _LOGGER.warning(f"Error processing {wav_path}: {e}")
        return None


def load_dataset(positive_dir: str, negative_dir: str) -> tuple:
    """Load and extract features from positive/negative samples."""
    X = []
    y = []

    # Load positive samples
    _LOGGER.info(f"Loading positive samples from {positive_dir}")
    positive_path = Path(positive_dir)
    for wav_file in sorted(positive_path.glob("*.wav")):
        features = extract_mfcc_features(str(wav_file))
        if features is not None:
            X.append(features)
            y.append(1)  # Positive = 1

    # Load negative samples
    _LOGGER.info(f"Loading negative samples from {negative_dir}")
    negative_path = Path(negative_dir)
    for wav_file in sorted(negative_path.glob("*.wav")):
        features = extract_mfcc_features(str(wav_file))
        if features is not None:
            X.append(features)
            y.append(0)  # Negative = 0

    _LOGGER.info(f"Loaded {len(X)} total samples")
    if len(X) == 0:
        raise ValueError("No samples loaded from dataset")

    return np.array(X), np.array(y)


def train_model(positive_dir: str, negative_dir: str, output_model: str = "wakeword_model.pkl"):
    """Train a simple classifier on wake word data."""
    _LOGGER.info("Loading dataset...")
    X, y = load_dataset(positive_dir, negative_dir)

    if len(X) == 0:
        _LOGGER.error("No samples loaded!")
        return False

    _LOGGER.info(f"Dataset: {len(X)} samples, {X[0].shape[0]} features")
    _LOGGER.info(f"Class distribution: {np.sum(y)} positive, {len(y) - np.sum(y)} negative")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    _LOGGER.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Train simple classifier
    _LOGGER.info("Training classifier...")
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        _LOGGER.error("scikit-learn required. Install: pip install scikit-learn")
        return False

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    # Evaluate
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)

    _LOGGER.info(f"Training accuracy: {train_score:.3f}")
    _LOGGER.info(f"Testing accuracy: {test_score:.3f}")

    # Save model
    _LOGGER.info(f"Saving model to {output_model}")
    with open(output_model, "wb") as f:
        pickle.dump({"model": clf, "n_mfcc": 13}, f)

    _LOGGER.info("✓ Model trained successfully")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train wake word detector")
    parser.add_argument("positive_dir", help="Directory with positive samples")
    parser.add_argument("negative_dir", help="Directory with negative samples")
    parser.add_argument("--output", default="wakeword_model.pkl", help="Output model file")
    args = parser.parse_args()

    success = train_model(args.positive_dir, args.negative_dir, args.output)
    exit(0 if success else 1)
