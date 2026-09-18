#!/usr/bin/env python3
import argparse
import audioop
import wave
from pathlib import Path
from typing import Union

import numpy as np
from audiomentations import (
    AddGaussianSNR,
    ApplyImpulseResponse,
    Compose,
    Gain,
    PitchShift,
    TimeStretch,
)

_DIR = Path(__file__).parent


def augment_directory(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    sample_rate: int,
    copies: int = 1,
) -> None:
    """Augment every WAV in input_dir, writing `copies` variants of each.

    `copies` lets a small source set be expanded to match a larger one, which
    matters when each class must draw evenly from several synthesis engines:
    otherwise the model can separate the classes by vocoder fingerprint
    instead of by what was actually said.
    """
    impulses = list((_DIR / "impulses").glob("*.wav"))

    augment = Compose(
        transforms=[
            # Vary pitch and speaking rate. TTS voices each have one fixed
            # delivery, so without this the model only ever sees a handful of
            # prosodies and misses the same words said higher, lower, faster
            # or slower than the synthesiser happened to render them.
            PitchShift(min_semitones=-3.0, max_semitones=3.0, p=0.6),
            TimeStretch(min_rate=0.85, max_rate=1.2, leave_length_unchanged=False, p=0.6),
            Gain(min_gain_db=-12, max_gain_db=0),
            ApplyImpulseResponse(impulses),
            # Real microphone audio always carries a noise floor, while TTS
            # output is digitally silent between words. Without this, a model
            # trained here keys on that unnatural silence and rejects live
            # speech the moment any room noise is present.
            AddGaussianSNR(min_snr_db=8.0, max_snr_db=40.0, p=1.0),
        ]
    )

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_wav in sorted(input_dir.glob("*.wav")):
        with wave.open(str(input_wav), "rb") as input_wav_file:
            assert input_wav_file.getsampwidth() == 2
            assert input_wav_file.getnchannels() == 1

            input_rate = input_wav_file.getframerate()
            input_audio = (
                np.frombuffer(
                    input_wav_file.readframes(input_wav_file.getnframes()),
                    dtype=np.int16,
                ).astype(np.float32)
                / 32767.0
            )

        for copy_index in range(copies):
            stem = input_wav.stem if copies == 1 else f"{input_wav.stem}_c{copy_index}"
            output_wav = output_dir / f"{stem}.wav"

            output_audio = augment(input_audio, sample_rate=input_rate)
            output_audio_16 = audio_float_to_int16(output_audio)

            if sample_rate != input_rate:
                output_audio_16, _state = audioop.ratecv(
                    output_audio_16, 2, 1, input_rate, sample_rate, None
                )

            with wave.open(str(output_wav), "wb") as output_wav_file:
                output_wav_file.setframerate(sample_rate or input_rate)
                output_wav_file.setsampwidth(2)
                output_wav_file.setnchannels(1)
                output_wav_file.writeframes(output_audio_16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--sample-rate", type=int, required=True)
    parser.add_argument("--copies", type=int, default=1,
                        help="Augmented variants to write per input file")
    args = parser.parse_args()

    augment_directory(args.input_dir, args.output_dir, args.sample_rate, args.copies)


def audio_float_to_int16(
    audio: np.ndarray, max_wav_value: float = 32767.0
) -> np.ndarray:
    # Don't normalize
    audio_norm = audio * max_wav_value
    audio_norm = np.clip(audio_norm, -max_wav_value, max_wav_value)
    audio_norm = audio_norm.astype("int16")
    return audio_norm


if __name__ == "__main__":
    main()
