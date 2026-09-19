#!/usr/bin/env python3
"""Live wake word detection from the system microphone."""

import argparse
import logging
import time

import numpy as np
import sounddevice as sd

from piper_sample_generator.train import load_model, score_live_window

logging.basicConfig(level=logging.INFO, format="%(message)s")
_LOGGER = logging.getLogger(__name__)

SAMPLE_RATE = 16000


def predict(model, data, audio: np.ndarray) -> float:
    """Return positive-class confidence for a live audio window."""
    return score_live_window(audio, model, data["norm"])


def listen(model_file: str, window_seconds: float, threshold: float, device: int = None):
    model, data = load_model(model_file)
    window_samples = int(window_seconds * SAMPLE_RATE)

    _LOGGER.info(f"Listening for wake word (window={window_seconds}s, threshold={threshold})...")
    _LOGGER.info("Press Ctrl+C to stop.\n")

    audio_buffer = np.zeros(window_samples, dtype=np.float32)

    def callback(indata, _frames, _time_info, status):
        nonlocal audio_buffer
        if status:
            _LOGGER.warning(str(status))

        chunk = indata[:, 0].astype(np.float32)
        audio_buffer = np.roll(audio_buffer, -len(chunk))
        audio_buffer[-len(chunk):] = chunk

    chunk_duration = 0.5  # seconds, how often we run inference
    blocksize = int(chunk_duration * SAMPLE_RATE)

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=blocksize,
        device=device,
        callback=callback,
    ):
        try:
            while True:
                time.sleep(chunk_duration)

                # Skip until buffer has real audio (avoid triggering on silence at startup)
                if np.abs(audio_buffer).max() < 1e-4:
                    _LOGGER.info("(listening, no audio detected yet...)")
                    continue

                confidence = predict(model, data, audio_buffer.copy())

                if confidence >= threshold:
                    _LOGGER.info(f"DETECTED  (confidence: {confidence:.3f})")
                else:
                    _LOGGER.info(f"rejected  (confidence: {confidence:.3f})")
        except KeyboardInterrupt:
            _LOGGER.info("\nStopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live wake word detection from microphone")
    parser.add_argument("model", nargs="?", help="Trained model file (.pkl)")
    parser.add_argument(
        "--window", type=float, default=1.5, help="Audio window length in seconds (default: 1.5)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Positive-class confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--device", type=int, default=None, help="Input device index (default: system default). Use --list-devices to see options."
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="List available audio input devices and exit"
    )
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        exit(0)

    if not args.model:
        parser.error("the following arguments are required: model")

    listen(args.model, args.window, args.threshold, args.device)
