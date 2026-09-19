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

from piper_sample_generator.train import WakeWordCNN, log_mel_spectrogram

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)


def predict(model_file: str, audio_file: str) -> dict:
    """Predict if audio contains wake word."""
    with open(model_file, "rb") as f:
        data = pickle.load(f)

    model = WakeWordCNN()
    model.load_state_dict(data["model_state_dict"])
    model.eval()

    try:
        audio, _sr = librosa.load(audio_file, sr=16000)
        spec = log_mel_spectrogram(audio)
    except Exception as e:
        _LOGGER.error(f"Error loading audio: {e}")
        return None

    norm = data["norm"]
    x = torch.tensor((spec - norm["mean"]) / norm["std"], dtype=torch.float32).unsqueeze(0)

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
