#!/usr/bin/env python3
"""FastAPI web app for live wake word detection from a browser-selected microphone."""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from piper_sample_generator.train import load_model, score_live_window

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
_DEBUG_DIR = None


def predict(audio: np.ndarray) -> float:
    """Return positive-class confidence for a live audio window."""
    return score_live_window(audio, _MODEL, _MODEL_DATA["norm"])


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

    # When capturing, every sample and score is retained so the audio the model
    # actually saw can be replayed offline. A live microphone is the one input
    # no synthetic dataset can stand in for, so when live confidence disagrees
    # with offline evaluation this recording is the only way to tell which of
    # the two is wrong.
    captured = [] if _DEBUG_DIR else None
    scores = [] if _DEBUG_DIR else None

    try:
        while True:
            raw = await websocket.receive_bytes()
            chunk = np.frombuffer(raw, dtype=np.float32)

            if len(chunk) == 0:
                continue

            if captured is not None:
                captured.append(chunk.copy())

            if len(chunk) >= window_samples:
                audio_buffer = chunk[-window_samples:].copy()
            else:
                audio_buffer = np.roll(audio_buffer, -len(chunk))
                audio_buffer[-len(chunk):] = chunk

            peak = float(np.abs(audio_buffer).max())
            if peak < 1e-4:
                if scores is not None:
                    scores.append((len(captured), 0.0, peak))
                await websocket.send_json({"status": "silent", "confidence": 0.0, "peak": peak})
                continue

            confidence = predict(audio_buffer.copy())
            detected = confidence >= _THRESHOLD

            if scores is not None:
                scores.append((len(captured), confidence, peak))

            await websocket.send_json(
                {
                    "status": "detected" if detected else "rejected",
                    "confidence": confidence,
                    "peak": peak,
                }
            )
    except WebSocketDisconnect:
        _LOGGER.info("Client disconnected")
    finally:
        if captured:
            _write_capture(captured, scores)


def _write_capture(captured, scores) -> None:
    """Persist a session's microphone audio alongside the scores it produced."""
    _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    audio = np.concatenate(captured)
    wav_path = _DEBUG_DIR / f"mic-{stamp}.wav"
    sf.write(wav_path, audio, SAMPLE_RATE)

    csv_path = _DEBUG_DIR / f"mic-{stamp}.csv"
    with csv_path.open("w") as handle:
        handle.write("chunk,confidence,peak\n")
        for index, confidence, peak in scores:
            handle.write(f"{index},{confidence:.4f},{peak:.5f}\n")

    best = max((s[1] for s in scores), default=0.0)
    _LOGGER.info(
        f"Captured {len(audio) / SAMPLE_RATE:.1f}s to {wav_path.name} "
        f"({len(scores)} scores, peak confidence {best:.3f})"
    )


def main() -> None:
    global _MODEL, _MODEL_DATA, _WINDOW_SECONDS, _THRESHOLD, _DEBUG_DIR

    parser = argparse.ArgumentParser(description="Wake word detection web app")
    parser.add_argument("model", help="Trained model file (.pkl)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--window", type=float, default=1.5, help="Audio window seconds")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--debug-dir", help="Record received microphone audio and scores here")
    args = parser.parse_args()

    _WINDOW_SECONDS = args.window
    _THRESHOLD = args.threshold
    _DEBUG_DIR = Path(args.debug_dir) if args.debug_dir else None

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
