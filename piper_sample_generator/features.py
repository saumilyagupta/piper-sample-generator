#!/usr/bin/env python3
"""Log-mel features for the wake word detector, computable on a microcontroller.

The original feature path could only ever run on a host. Three of its steps
need the whole clip in hand before they can produce a single output frame:

  trim_silence()        Finds the speech region by scanning the entire clip.
  peak normalization    Divides by the clip maximum, unknown until the end.
  power_to_db(ref=max)  Sets the dB reference from the clip maximum, likewise.

On a device the audio arrives as a stream, so none of these are available. They
also caused a subtler failure on the host: the live rolling-window path
recomputed all three per window, over a window whose contents differed from any
training clip, so the model was fed a different normalization at inference than
it saw during training. That is the train/serve skew behind live detection
scoring far lower than batch evaluation on the same phrase.

The replacements here are all causal:

  no trimming           The model is trained with the clip at a random offset
                        inside the window instead, which makes it robust to
                        where the phrase falls rather than trying to center it.
  fixed dB reference    An absolute reference, so a frame's value depends only
                        on that frame.
  frozen per-bin norm   Mean and standard deviation per mel bin, computed once
                        over the training set and stored in the model file.

Loudness invariance, previously supplied by peak normalization, now comes from
gain augmentation spanning microphone levels (see augment.py). The frontend no
longer hides level differences, so the model has to learn through them.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import librosa
import numpy as np

_LOGGER = logging.getLogger(__name__)

SAMPLE_RATE = 16000
N_MELS = 40
N_FRAMES = 128          # ~1.28s at a 10ms hop
N_FFT = 400             # 25ms window
HOP_LENGTH = 160        # 10ms hop
FMIN = 20
FMAX = 8000

# Absolute dB reference. Power of 1.0 maps to 0 dB, so a frame's value is a
# function of that frame alone.
DB_REF = 1.0
DB_FLOOR = -80.0
DB_CEIL = 20.0

PadMode = str  # "start" | "center" | "random" | "end"


def log_mel_spectrogram(
    audio: np.ndarray,
    pad_mode: PadMode = "start",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Fixed-size log-mel spectrogram for a 16 kHz waveform.

    Args:
        audio: Mono float waveform, nominally in [-1, 1].
        pad_mode: Where to place the clip within the N_FRAMES window when it is
            shorter. "random" during training, so the model learns to spot the
            phrase wherever it lands; "start" for reproducible evaluation;
            "end" matches a live rolling buffer, which fills from the right.
        rng: Random source for pad_mode="random".

    Returns:
        Array of shape (N_MELS, N_FRAMES), float32, in dB.
    """
    audio = np.asarray(audio, dtype=np.float32)

    if audio.size < N_FFT:
        audio = np.pad(audio, (0, N_FFT - audio.size))

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
    )
    mel_db = librosa.power_to_db(mel, ref=DB_REF, top_db=None)
    mel_db = np.clip(mel_db, DB_FLOOR, DB_CEIL)

    return _fit_to_window(mel_db, pad_mode, rng).astype(np.float32)


def _fit_to_window(
    mel_db: np.ndarray, pad_mode: PadMode, rng: Optional[np.random.Generator]
) -> np.ndarray:
    """Pad or crop the time axis to exactly N_FRAMES.

    Padding uses the dB floor rather than zeros: at an absolute reference, 0 dB
    is full-scale energy, so zero-padding would splice a burst of loud noise
    onto every short clip.
    """
    frames = mel_db.shape[1]

    if frames == N_FRAMES:
        return mel_db

    if frames > N_FRAMES:
        if pad_mode == "end":
            return mel_db[:, -N_FRAMES:]
        if pad_mode == "random":
            rng = rng or np.random.default_rng()
            start = int(rng.integers(0, frames - N_FRAMES + 1))
            return mel_db[:, start:start + N_FRAMES]
        if pad_mode == "center":
            start = (frames - N_FRAMES) // 2
            return mel_db[:, start:start + N_FRAMES]
        return mel_db[:, :N_FRAMES]

    room = N_FRAMES - frames
    if pad_mode == "random":
        rng = rng or np.random.default_rng()
        left = int(rng.integers(0, room + 1))
    elif pad_mode == "center":
        left = room // 2
    elif pad_mode == "end":
        left = room
    else:
        left = 0

    return np.pad(
        mel_db, ((0, 0), (left, room - left)), mode="constant", constant_values=DB_FLOOR
    )


def extract_features(
    wav_path: Union[str, Path],
    pad_mode: PadMode = "start",
    rng: Optional[np.random.Generator] = None,
) -> Optional[np.ndarray]:
    """Load a WAV and return its spectrogram, or None if unreadable."""
    try:
        audio, _sr = librosa.load(str(wav_path), sr=SAMPLE_RATE)
        return log_mel_spectrogram(audio, pad_mode=pad_mode, rng=rng)
    except Exception as e:
        _LOGGER.warning(f"Error processing {wav_path}: {e}")
        return None


def compute_norm(specs: np.ndarray) -> Dict[str, list]:
    """Per-mel-bin mean and standard deviation over a training set.

    Per bin rather than a single scalar: mel bins differ by tens of dB in both
    level and spread, and one scalar leaves the low bins dominating the input
    purely because they carry more energy.

    Args:
        specs: Array of shape (n_samples, N_MELS, N_FRAMES).

    Returns:
        Dict with "mean" and "std", each a list of N_MELS floats.
    """
    mean = specs.mean(axis=(0, 2))
    std = specs.std(axis=(0, 2))
    std = np.where(std < 1e-3, 1.0, std)
    return {"mean": mean.tolist(), "std": std.tolist()}


def apply_norm(specs: np.ndarray, norm: Dict) -> np.ndarray:
    """Normalize spectrograms with stored statistics.

    Accepts both the per-bin statistics written by compute_norm and the single
    scalar mean/std used by models trained before this change, so existing
    .pkl files keep loading.
    """
    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["std"], dtype=np.float32)

    if mean.ndim == 1:  # per-bin: broadcast down the frequency axis
        shape = (1, -1, 1) if specs.ndim == 3 else (-1, 1)
        mean = mean.reshape(shape)
        std = std.reshape(shape)

    return ((specs - mean) / std).astype(np.float32)
