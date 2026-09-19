#!/usr/bin/env python3
"""Augment synthesized clips so a detector trained on them survives real audio."""

import argparse
import audioop
import logging
import wave
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from audiomentations import (
    AddGaussianSNR,
    ApplyImpulseResponse,
    Compose,
    Gain,
    PitchShift,
    TimeStretch,
)

_LOGGER = logging.getLogger(__name__)

_DIR = Path(__file__).parent
BUNDLED_IMPULSES = _DIR / "impulses"
IMPULSE_CACHE = _DIR / "impulses_resampled"

# Feature extraction no longer peak-normalizes, so loudness reaches the model
# instead of being divided out. The model has to learn through it, which means
# training audio must span the range a microphone actually delivers. Measured
# on these clips, speech at -40 dB gain still lands ~10 dB above the feature
# floor, so that is the useful lower bound; below it the int16 output starts
# quantizing the signal away.
MIN_GAIN_DB = -40.0
MAX_GAIN_DB = 3.0

# Only a fraction of clips get a room response. The bundled impulses are music
# production effects -- "Blatty Plate", "Fat Bass", "Reverse Gate", "Symphonic"
# -- with 3-4 second tails, not the response of a small room to a voice a metre
# away. Applying one to every clip would train the model on reverb no device
# microphone will ever hear. Point --impulse-dir at a real RIR corpus (openSLR
# RIRS_NOISES small/medium room, or the MIT Acoustical Reverberation Scene) and
# raising this toward 1.0 becomes the right move.
IMPULSE_PROB = 0.5


def _resample_impulses(impulse_dir: Path, sample_rate: int) -> List[Path]:
    """Resample impulse responses to sample_rate once, caching the result.

    audiomentations resamples on every call otherwise, which is both slow and
    noisy: every augmented clip logged a resample warning.
    """
    import librosa
    import soundfile as sf

    sources = sorted(p for p in impulse_dir.glob("*.wav"))
    if not sources:
        return []

    cache = IMPULSE_CACHE / f"{impulse_dir.name}_{sample_rate}"
    cache.mkdir(parents=True, exist_ok=True)

    out = []
    for source in sources:
        target = cache / source.name
        if not target.exists() or target.stat().st_mtime < source.stat().st_mtime:
            audio, _ = librosa.load(source, sr=sample_rate, mono=True)
            peak = float(np.abs(audio).max())
            if peak > 1e-9:
                audio = audio / peak
            sf.write(target, audio, sample_rate)
        out.append(target)

    return out


def build_augmenter(
    sample_rate: int,
    impulse_dir: Optional[Union[str, Path]] = None,
    impulse_prob: float = IMPULSE_PROB,
) -> Compose:
    """Assemble the augmentation chain."""
    impulse_dir = Path(impulse_dir) if impulse_dir else BUNDLED_IMPULSES
    impulses = _resample_impulses(impulse_dir, sample_rate)

    transforms = [
        # Vary pitch and speaking rate. TTS voices each have one fixed
        # delivery, so without this the model only ever sees a handful of
        # prosodies and misses the same words said higher, lower, faster
        # or slower than the synthesiser happened to render them.
        PitchShift(min_semitones=-3.0, max_semitones=3.0, p=0.6),
        TimeStretch(min_rate=0.85, max_rate=1.2, leave_length_unchanged=False, p=0.6),
    ]

    # Ordering matters: ApplyImpulseResponse renormalizes its output to a fixed
    # -6 dBFS, so anything it touches comes out at exactly one level no matter
    # what it went in at. With Gain ahead of it the whole gain range collapses
    # to a single value. It also means the previous chain stamped -6 dBFS onto
    # the ~50% of clips that got a room response and left the rest alone,
    # handing the model a loud level cue that had nothing to do with the label.
    if impulses:
        transforms.append(ApplyImpulseResponse(impulses, p=impulse_prob))
    else:
        _LOGGER.warning(f"No impulse responses found in {impulse_dir}")

    transforms.extend([
        Gain(min_gain_db=MIN_GAIN_DB, max_gain_db=MAX_GAIN_DB, p=1.0),
        # Real microphone audio always carries a noise floor, while TTS
        # output is digitally silent between words. Without this, a model
        # trained here keys on that unnatural silence and rejects live
        # speech the moment any room noise is present.
        AddGaussianSNR(min_snr_db=5.0, max_snr_db=40.0, p=1.0),
    ])

    return Compose(transforms=transforms)


def augment_directory(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    sample_rate: int,
    copies: int = 1,
    impulse_dir: Optional[Union[str, Path]] = None,
    impulse_prob: float = IMPULSE_PROB,
) -> None:
    """Augment every WAV in input_dir, writing `copies` variants of each.

    `copies` lets a small source set be expanded to match a larger one, which
    matters when each class must draw evenly from several synthesis engines:
    otherwise the model can separate the classes by vocoder fingerprint
    instead of by what was actually said.
    """
    augment = build_augmenter(sample_rate, impulse_dir, impulse_prob)

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


def audio_float_to_int16(
    audio: np.ndarray, max_wav_value: float = 32767.0
) -> np.ndarray:
    # Don't normalize
    audio_norm = audio * max_wav_value
    audio_norm = np.clip(audio_norm, -max_wav_value, max_wav_value)
    audio_norm = audio_norm.astype("int16")
    return audio_norm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--sample-rate", type=int, required=True)
    parser.add_argument("--copies", type=int, default=1,
                        help="Augmented variants to write per input file")
    parser.add_argument("--impulse-dir", default=None,
                        help="Directory of impulse response WAVs. Defaults to "
                             "the bundled effects; point this at a real room "
                             "impulse response corpus when you have one.")
    parser.add_argument("--impulse-prob", type=float, default=IMPULSE_PROB,
                        help="Probability a clip gets a room response applied")
    args = parser.parse_args()

    augment_directory(args.input_dir, args.output_dir, args.sample_rate,
                      args.copies, args.impulse_dir, args.impulse_prob)


if __name__ == "__main__":
    main()
