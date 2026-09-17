#!/usr/bin/env python3
"""Test trained wake word detector."""

import argparse
import logging
import pickle
from pathlib import Path

import librosa
import numpy as np

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)


def predict(model_file: str, audio_file: str) -> dict:
    """Predict if audio contains wake word."""
    # Load model
    with open(model_file, "rb") as f:
        data = pickle.load(f)
        clf = data["model"]
        n_mfcc = data["n_mfcc"]

    # Extract features
    try:
        y, sr = librosa.load(audio_file, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        features = mfcc.flatten()
    except Exception as e:
        _LOGGER.error(f"Error loading audio: {e}")
        return None

    # Predict
    pred = clf.predict([features])[0]
    prob = clf.predict_proba([features])[0]

    return {
        "prediction": "POSITIVE (Wake word detected)" if pred == 1 else "NEGATIVE (No wake word)",
        "confidence_negative": prob[0],
        "confidence_positive": prob[1],
        "label": int(pred),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test wake word detector")
    parser.add_argument("model", help="Trained model file (.pkl)")
    parser.add_argument("audio", help="Audio file to test (.wav)")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"ERROR: Model file not found: {args.model}")
        exit(1)

    if not Path(args.audio).exists():
        print(f"ERROR: Audio file not found: {args.audio}")
        exit(1)

    result = predict(args.model, args.audio)
    if result:
        print(f"\nPrediction: {result['prediction']}")
        print(f"Confidence (negative): {result['confidence_negative']:.3f}")
        print(f"Confidence (positive): {result['confidence_positive']:.3f}")
