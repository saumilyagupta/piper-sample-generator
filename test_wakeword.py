#!/usr/bin/env python3
"""Test trained wake word detector (MLP + best val-loss checkpoint)."""

import argparse
import logging
import pickle
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F

from piper_sample_generator.train import WakeWordMLP

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)


def extract_mfcc_features(wav_path: str, n_mfcc: int, max_frames: int) -> np.ndarray:
    """Extract fixed-length MFCC features matching training preprocessing."""
    y, sr = librosa.load(wav_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    if mfcc.shape[1] < max_frames:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_frames - mfcc.shape[1])), mode="constant")
    else:
        mfcc = mfcc[:, :max_frames]

    return mfcc.flatten()


def predict(model_file: str, audio_file: str) -> dict:
    """Predict if audio contains wake word."""
    with open(model_file, "rb") as f:
        data = pickle.load(f)

    model = WakeWordMLP(input_dim=data["input_dim"])
    model.load_state_dict(data["model_state_dict"])
    model.eval()

    try:
        features = extract_mfcc_features(audio_file, data["n_mfcc"], data["max_frames"])
    except Exception as e:
        _LOGGER.error(f"Error loading audio: {e}")
        return None

    features = data["scaler"].transform([features])
    x = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
        pred = int(torch.argmax(probs).item())

    return {
        "prediction": "POSITIVE (Wake word detected)" if pred == 1 else "NEGATIVE (No wake word)",
        "confidence_negative": probs[0].item(),
        "confidence_positive": probs[1].item(),
        "label": pred,
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
