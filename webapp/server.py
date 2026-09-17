#!/usr/bin/env python3
"""FastAPI web app for live wake word detection from a browser-selected microphone."""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from piper_sample_generator.train import WakeWordMLP, mfcc_from_audio, trim_leading_silence

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)

SAMPLE_RATE = 16000
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Wake Word Detector")

# Populated at startup by main()
_MODEL = None
_MODEL_DATA = None
_WINDOW_SECONDS = 1.5
_THRESHOLD = 0.5


def load_model(model_file: str):
    with open(model_file, "rb") as f:
        data = pickle.load(f)

    model = WakeWordMLP(input_dim=data["input_dim"])
    model.load_state_dict(data["model_state_dict"])
    model.eval()
    return model, data


def predict(audio: np.ndarray) -> tuple:
    """Return (label, confidence_positive)."""
    audio = trim_leading_silence(audio)
    mfcc = mfcc_from_audio(audio, _MODEL_DATA["n_mfcc"], _MODEL_DATA["max_frames"])
    features = _MODEL_DATA["scaler"].transform([mfcc])
    x = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        probs = F.softmax(_MODEL(x), dim=1)[0]

    return int(torch.argmax(probs).item()), probs[1].item()


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/config")
async def config():
    return {
        "sample_rate": SAMPLE_RATE,
        "window_seconds": _WINDOW_SECONDS,
        "threshold": _THRESHOLD,
    }


@app.websocket("/ws")
async def websocket_audio(websocket: WebSocket):
    """Receive raw float32 PCM at 16kHz, run detection on a rolling window."""
    await websocket.accept()

    window_samples = int(_WINDOW_SECONDS * SAMPLE_RATE)
    audio_buffer = np.zeros(window_samples, dtype=np.float32)

    try:
        while True:
            raw = await websocket.receive_bytes()
            chunk = np.frombuffer(raw, dtype=np.float32)

            if len(chunk) == 0:
                continue

            if len(chunk) >= window_samples:
                audio_buffer = chunk[-window_samples:].copy()
            else:
                audio_buffer = np.roll(audio_buffer, -len(chunk))
                audio_buffer[-len(chunk):] = chunk

            peak = float(np.abs(audio_buffer).max())
            if peak < 1e-4:
                await websocket.send_json({"status": "silent", "confidence": 0.0, "peak": peak})
                continue

            label, confidence = predict(audio_buffer.copy())
            detected = label == 1 and confidence >= _THRESHOLD

            await websocket.send_json(
                {
                    "status": "detected" if detected else "rejected",
                    "confidence": confidence,
                    "peak": peak,
                }
            )
    except WebSocketDisconnect:
        _LOGGER.info("Client disconnected")


def main() -> None:
    global _MODEL, _MODEL_DATA, _WINDOW_SECONDS, _THRESHOLD

    parser = argparse.ArgumentParser(description="Wake word detection web app")
    parser.add_argument("model", help="Trained model file (.pkl)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--window", type=float, default=1.5, help="Audio window seconds")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection confidence threshold")
    args = parser.parse_args()

    _WINDOW_SECONDS = args.window
    _THRESHOLD = args.threshold

    _LOGGER.info(f"Loading model: {args.model}")
    _MODEL, _MODEL_DATA = load_model(args.model)
    _LOGGER.info(
        f"Model loaded (best epoch {_MODEL_DATA.get('best_epoch')}, "
        f"test acc {_MODEL_DATA.get('test_acc', 0):.3f})"
    )

    import uvicorn

    _LOGGER.info(f"Open http://{args.host}:{args.port} in your browser")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
