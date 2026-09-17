#!/usr/bin/env python3
"""Wake word detector training with train/val/test split and best-val-loss checkpointing."""

import copy
import logging
import pickle
from pathlib import Path
from typing import Union

import librosa
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_LOGGER = logging.getLogger(__name__)

N_MFCC = 13
MAX_FRAMES = 100  # retained for models trained before bin-pooled features
N_BINS = 16


def speech_features(
    audio: np.ndarray, n_mfcc: int = N_MFCC, n_bins: int = N_BINS
) -> np.ndarray:
    """Duration-normalized, alignment-invariant MFCC features.

    Trims to the speech region, then splits it into a fixed number of equal
    time bins and takes mean+std of each MFCC coefficient per bin.

    This replaces flat truncate/pad-to-N-frames features, which encoded clip
    duration (training clips filled ~40 of 100 frames, the rest zero padding)
    rather than the phrase, and shifted every value when the phrase moved
    within the window. Binning makes the vector independent of duration,
    onset position and speaking rate, so a live capture and a training clip
    of the same phrase land in the same place.
    """
    audio = trim_silence(audio)

    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if peak > 1e-6:
        audio = audio / peak

    if audio.size < 512:
        return np.zeros(n_bins * n_mfcc * 2, dtype=np.float32)

    mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=n_mfcc)

    # Guarantee at least one frame per bin so no bin is empty.
    if mfcc.shape[1] < n_bins:
        mfcc = np.repeat(mfcc, int(np.ceil(n_bins / mfcc.shape[1])), axis=1)

    parts = np.array_split(mfcc, n_bins, axis=1)
    feats = [np.concatenate([p.mean(axis=1), p.std(axis=1)]) for p in parts]

    return np.concatenate(feats).astype(np.float32)


def mfcc_from_audio(
    audio: np.ndarray, n_mfcc: int = N_MFCC, max_frames: int = MAX_FRAMES
) -> np.ndarray:
    """Extract fixed-length MFCC features from a 16kHz float32 waveform.

    Peak-normalizes first so features describe spectral shape, not loudness.
    Without this the model keys on absolute level (training clips are loud,
    microphone input is quiet), and the same phrase scores 0.85 loud but 0.06
    quiet. Normalizing here keeps training and inference in lockstep.
    """
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if peak > 1e-6:
        audio = audio / peak

    mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=n_mfcc)

    if mfcc.shape[1] < max_frames:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_frames - mfcc.shape[1])), mode="constant")
    else:
        mfcc = mfcc[:, :max_frames]

    return mfcc.flatten()


def trim_leading_silence(audio: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """Drop leading near-silence so speech starts at sample 0.

    Training samples are zero-trimmed at generation time, so their MFCC frames
    begin at the onset of speech. A live rolling buffer instead right-aligns
    audio, which shifts every frame and makes the features unrecognizable to
    the model — this realigns it.

    `threshold` is a fraction of the clip's own peak, not an absolute level:
    an absolute cutoff picks the wrong onset on quiet input and misaligns
    every frame.
    """
    if audio.size == 0:
        return audio

    peak = float(np.abs(audio).max())
    if peak <= 1e-6:
        return audio

    loud = np.flatnonzero(np.abs(audio) >= threshold * peak)
    if loud.size == 0:
        return audio
    return audio[loud[0]:]


def trim_silence(audio: np.ndarray, top_db: float = 15.0) -> np.ndarray:
    """Trim to the speech region using short-time energy.

    Uses frame energy in dB relative to peak rather than a raw sample
    threshold: with a sample threshold, any noise floor above the cutoff
    defeats trimming entirely, so a live window keeps its full 1.5s of noise
    and the time bins get diluted instead of spanning just the phrase.
    """
    if audio.size < 512:
        return audio

    peak = float(np.abs(audio).max())
    if peak <= 1e-6:
        return audio

    try:
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    except Exception:
        return audio

    return trimmed if trimmed.size >= 512 else audio


def write_noise_negatives(output_dir, count: int = 300, seed: int = 0) -> int:
    """Write noise/silence WAVs so the model learns to reject non-speech.

    The generated dataset only contains spoken phrases, so a detector trained
    on it has never seen "no one is talking" and will happily classify room
    noise as a wake word.
    """
    import wave

    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(count):
        duration = rng.uniform(0.6, 1.5)
        n = int(duration * 16000)
        kind = i % 3

        if kind == 0:  # broadband noise
            audio = rng.normal(0, rng.uniform(0.02, 0.3), n)
        elif kind == 1:  # low-frequency rumble / hum
            t = np.arange(n) / 16000
            freq = rng.uniform(50, 260)
            audio = rng.uniform(0.05, 0.3) * np.sin(2 * np.pi * freq * t)
            audio += rng.normal(0, 0.02, n)
        else:  # near-silence with a faint noise floor
            audio = rng.normal(0, rng.uniform(0.0005, 0.01), n)

        audio = np.clip(audio, -1.0, 1.0)
        pcm = (audio * 32767.0).astype(np.int16)

        with wave.open(str(output_dir / f"noise_{i}.wav"), "wb") as wav_file:
            wav_file.setframerate(16000)
            wav_file.setsampwidth(2)
            wav_file.setnchannels(1)
            wav_file.writeframes(pcm.tobytes())

    return count


def score_live_window(
    audio: np.ndarray,
    model: "WakeWordMLP",
    scaler,
    n_mfcc: int = N_MFCC,
    n_bins: int = N_BINS,
) -> float:
    """Positive-class confidence for a live audio window."""
    features = scaler.transform([speech_features(audio, n_mfcc, n_bins)])

    with torch.no_grad():
        probs = torch.softmax(model(torch.tensor(features, dtype=torch.float32)), dim=1)

    return float(probs[0, 1].item())


def extract_mfcc_features(wav_path: str, n_mfcc: int = N_MFCC, n_bins: int = N_BINS) -> np.ndarray:
    """Extract features from an audio file, matching the live inference path."""
    try:
        y, _sr = librosa.load(wav_path, sr=16000)
        return speech_features(y, n_mfcc, n_bins)
    except Exception as e:
        _LOGGER.warning(f"Error processing {wav_path}: {e}")
        return None


def load_dataset(positive_dir: Union[str, Path], negative_dir: Union[str, Path]) -> tuple:
    """Load and extract features from positive/negative samples."""
    X, y = [], []

    _LOGGER.info(f"Loading positive samples from {positive_dir}")
    for wav_file in sorted(Path(positive_dir).glob("*.wav")):
        features = extract_mfcc_features(str(wav_file))
        if features is not None:
            X.append(features)
            y.append(1)

    _LOGGER.info(f"Loading negative samples from {negative_dir}")
    for wav_file in sorted(Path(negative_dir).glob("*.wav")):
        features = extract_mfcc_features(str(wav_file))
        if features is not None:
            X.append(features)
            y.append(0)

    if not X:
        raise ValueError("No samples loaded from dataset")

    _LOGGER.info(f"Loaded {len(X)} total samples")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


class WakeWordMLP(nn.Module):
    """Small feedforward classifier over MFCC features."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x):
        return self.net(x)


def split_train_val_test(X: np.ndarray, y: np.ndarray, seed: int = 42) -> tuple:
    """70/15/15 train/val/test split (stratified)."""
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.50, random_state=seed, stratify=y_rest
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_model(
    positive_dir: Union[str, Path],
    negative_dir: Union[str, Path],
    output_model: Union[str, Path] = "wakeword_model.pkl",
    epochs: int = 100,
    patience: int = 15,
    lr: float = 1e-3,
    batch_size: int = 32,
    noise_negatives: int = 300,
) -> bool:
    """Train MLP classifier with 70/15/15 split, checkpointing best val-loss model."""
    if noise_negatives:
        existing = len(list(Path(negative_dir).glob("noise_*.wav")))
        if existing < noise_negatives:
            _LOGGER.info(f"Adding {noise_negatives} noise/silence negatives")
            write_noise_negatives(negative_dir, noise_negatives)

    _LOGGER.info("Loading dataset...")
    X, y = load_dataset(positive_dir, negative_dir)

    _LOGGER.info(f"Dataset: {len(X)} samples, {X.shape[1]} features")
    _LOGGER.info(f"Class distribution: {int(np.sum(y))} positive, {len(y) - int(np.sum(y))} negative")

    X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(X, y)
    _LOGGER.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Normalize features (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _LOGGER.info(f"Training on: {device}")

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.long, device=device)

    model = WakeWordMLP(input_dim=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Negatives outnumber positives ~3:1 (each phrase yields more negative
    # variations than positive ones, plus noise samples). Unweighted, the
    # model stays biased toward "reject" and true positives score below 0.5
    # even when they separate cleanly from negatives.
    counts = np.bincount(y_train, minlength=2).astype(np.float64)
    weights = torch.tensor(
        (counts.sum() / (2.0 * np.maximum(counts, 1))), dtype=torch.float32, device=device
    )
    _LOGGER.info(f"Class weights: negative={weights[0]:.2f}, positive={weights[1]:.2f}")
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    epochs_without_improvement = 0

    n_train = len(X_train_t)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0

        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(idx)

        train_loss = epoch_loss / n_train

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
            val_acc = (val_logits.argmax(dim=1) == y_val_t).float().mean().item()

        _LOGGER.info(
            f"Epoch {epoch + 1}/{epochs} | train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                _LOGGER.info(f"Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
                break

    _LOGGER.info(f"Best val_loss={best_val_loss:.4f} at epoch {best_epoch}")

    # Restore best checkpoint before final evaluation
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        train_logits = model(X_train_t)
        train_acc = (train_logits.argmax(dim=1) == y_train_t).float().mean().item()

        val_logits = model(X_val_t)
        val_acc = (val_logits.argmax(dim=1) == y_val_t).float().mean().item()

        test_logits = model(X_test_t)
        test_loss = criterion(test_logits, y_test_t).item()
        test_acc = (test_logits.argmax(dim=1) == y_test_t).float().mean().item()

    _LOGGER.info("=" * 50)
    _LOGGER.info(f"Best model (epoch {best_epoch}):")
    _LOGGER.info(f"  Train accuracy: {train_acc:.3f}")
    _LOGGER.info(f"  Val accuracy:   {val_acc:.3f} (val_loss={best_val_loss:.4f})")
    _LOGGER.info(f"  Test accuracy:  {test_acc:.3f} (test_loss={test_loss:.4f})")
    _LOGGER.info("=" * 50)

    _LOGGER.info(f"Saving best model to {output_model}")
    with open(output_model, "wb") as f:
        pickle.dump(
            {
                "model_state_dict": best_state,
                "input_dim": X.shape[1],
                "n_mfcc": N_MFCC,
                "n_bins": N_BINS,
                "scaler": scaler,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "test_acc": test_acc,
            },
            f,
        )

    _LOGGER.info("Model trained successfully")
    return True
