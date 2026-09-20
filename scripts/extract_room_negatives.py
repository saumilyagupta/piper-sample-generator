#!/usr/bin/env python3
"""Slice the non-speech parts of real captures into negative training clips.

Every negative so far has been either synthesized speech or procedurally
generated noise. Neither is what a microphone in this room actually delivers,
and the gap shows: the best model peaks at 0.511 across 37 seconds of a silent
room, just under the threshold it is served at.

The captures already hold that audio. Given a recording whose speech regions
are known, the stretches between them are real room noise with the real
microphone's response, recorded through the real capture path -- the one class
of negative that cannot be synthesized.

Speech is located by energy rather than by the model, so what gets excluded
does not depend on what is being trained.
"""
import argparse
import sys
import wave
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from piper_sample_generator.features import SAMPLE_RATE, WINDOW_SAMPLES  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from score_labelled_capture import speech_regions  # noqa: E402

# Kept clear of every speech region by more than the feature window, so no
# clip can contain a trailing fragment of the wake word.
GUARD_SECONDS = 1.6


def non_speech_windows(audio, regions, hop_seconds=0.4):
    """Full-length windows that do not overlap speech or its guard band."""
    hop = int(hop_seconds * SAMPLE_RATE)
    windows = []
    for start in range(0, len(audio) - WINDOW_SAMPLES, hop):
        begin = start / SAMPLE_RATE
        end = (start + WINDOW_SAMPLES) / SAMPLE_RATE
        clear = all(
            end < region_begin - GUARD_SECONDS or begin > region_end + GUARD_SECONDS
            for region_begin, region_end in regions
        )
        if clear:
            windows.append(audio[start:start + WINDOW_SAMPLES])
    return windows


def write_wav(path: Path, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setframerate(SAMPLE_RATE)
        handle.setsampwidth(2)
        handle.setnchannels(1)
        handle.writeframes(pcm.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+")
    parser.add_argument("--output-dir", default="room_negatives")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    written = 0
    for capture in args.captures:
        audio = librosa.load(capture, sr=SAMPLE_RATE)[0]
        regions, floor = speech_regions(audio)
        windows = non_speech_windows(audio, regions)

        stem = Path(capture).stem
        for i, window in enumerate(windows):
            write_wav(out / f"room_{stem}_{i}.wav", window)
        written += len(windows)

        rms = 20 * np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-12)
        print(f"{Path(capture).name}: {len(audio) / SAMPLE_RATE:.1f}s, "
              f"floor {floor:.1f} dB, overall {rms:.1f} dBFS, "
              f"{len(regions)} speech regions -> {len(windows)} clips")

    print(f"\n{written} room-noise clips in {out}")


if __name__ == "__main__":
    main()
