#!/usr/bin/env python3
import argparse
import audioop
import wave
from pathlib import Path
from typing import Union

import numpy as np
from audiomentations import AddGaussianSNR, ApplyImpulseResponse, Compose, Gain

_DIR = Path(__file__).parent


def augment_directory(
    input_dir: Union[str, Path], output_dir: Union[str, Path], sample_rate: int
) -> None:
    """Apply volume + impulse response augmentation to all WAVs in input_dir."""
    impulses = list((_DIR / "impulses").glob("*.wav"))

    augment = Compose(
        transforms=[
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

    for input_wav in input_dir.glob("*.wav"):
        output_wav = output_dir / (input_wav.relative_to(input_dir))
        output_wav.parent.mkdir(parents=True, exist_ok=True)

        with (
            wave.open(str(input_wav), "rb") as input_wav_file,
            wave.open(str(output_wav), "wb") as output_wav_file,
        ):
            assert input_wav_file.getsampwidth() == 2
            assert input_wav_file.getnchannels() == 1

            input_audio = (
                np.frombuffer(
                    input_wav_file.readframes(input_wav_file.getnframes()),
                    dtype=np.int16,
                ).astype(np.float32)
                / 32767.0
            )

            output_audio = augment(
                input_audio, sample_rate=input_wav_file.getframerate()
            )
            output_wav_file.setframerate(sample_rate or input_wav_file.getframerate())
            output_wav_file.setsampwidth(2)
            output_wav_file.setnchannels(1)

            output_audio_16 = audio_float_to_int16(output_audio)
            if sample_rate != input_wav_file.getframerate():
                output_audio_16, _state = audioop.ratecv(
                    output_audio_16,
                    2,
                    1,
                    input_wav_file.getframerate(),
                    sample_rate,
                    None,
                )

            output_wav_file.writeframes(output_audio_16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--sample-rate", type=int, required=True)
    args = parser.parse_args()

    augment_directory(args.input_dir, args.output_dir, args.sample_rate)


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
