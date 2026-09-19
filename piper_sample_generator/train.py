#!/usr/bin/env python3
"""Wake word detector: CNN over log-mel spectrograms.

Earlier versions pooled MFCC coefficients into time bins, which averaged away
the short consonant transitions that separate "limbo" from "nimbo". Measured on
real renders, near-miss negatives scored more similar to the wake word than two
renderings of the wake word scored to each other, so no classifier could
separate them. A convolutional model reads the time-frequency surface directly
and keeps that detail.
"""

import copy
import logging
import pickle
import wave
from pathlib import Path
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

_LOGGER = logging.getLogger(__name__)

SAMPLE_RATE = 16000
N_MELS = 40
N_FRAMES = 128          # ~1.28s at a 10ms hop
N_FFT = 400             # 25ms window
HOP_LENGTH = 160        # 10ms hop
DB_FLOOR = -80.0


def trim_silence(audio: np.ndarray, top_db: float = 15.0) -> np.ndarray:
    """Trim to the speech region using short-time energy.

    Uses frame energy in dB relative to peak rather than a raw sample
    threshold: with a sample threshold, any noise floor above the cutoff
    defeats trimming entirely, so a live window keeps its full buffer of noise.
    """
    if audio.size < 512:
        return audio

    if float(np.abs(audio).max()) <= 1e-6:
        return audio

    try:
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    except Exception:
        return audio

    return trimmed if trimmed.size >= 512 else audio


def log_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    """Fixed-size log-mel spectrogram for a 16kHz waveform.

    Peak-normalizes so the features describe spectral shape rather than
    loudness; training clips are loud while microphone input is quiet, and
    without this the same phrase scores very differently at the two levels.
    """
    audio = trim_silence(audio)

    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if peak > 1e-6:
        audio = audio / peak

    if audio.size < N_FFT:
        audio = np.pad(audio, (0, N_FFT - audio.size))

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=20,
        fmax=8000,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=-DB_FLOOR)

    # Pad with the dB floor rather than zeros, so padding reads as silence
    # instead of as full-scale energy.
    if mel_db.shape[1] < N_FRAMES:
        pad = N_FRAMES - mel_db.shape[1]
        mel_db = np.pad(
            mel_db, ((0, 0), (0, pad)), mode="constant", constant_values=DB_FLOOR
        )
    else:
        mel_db = mel_db[:, :N_FRAMES]

    return mel_db.astype(np.float32)


def extract_features(wav_path: Union[str, Path]) -> Optional[np.ndarray]:
    """Load a WAV and return its spectrogram, or None if unreadable."""
    try:
        audio, _sr = librosa.load(str(wav_path), sr=SAMPLE_RATE)
        return log_mel_spectrogram(audio)
    except Exception as e:
        _LOGGER.warning(f"Error processing {wav_path}: {e}")
        return None


def load_dataset(
    positive_dir: Union[str, Path], negative_dir: Union[str, Path]
) -> Tuple[np.ndarray, np.ndarray]:
    """Load spectrograms and labels from positive/negative directories."""
    X, y = [], []

    for directory, label in ((positive_dir, 1), (negative_dir, 0)):
        kind = "positive" if label else "negative"
        _LOGGER.info(f"Loading {kind} samples from {directory}")
        for wav_file in sorted(Path(directory).glob("*.wav")):
            features = extract_features(wav_file)
            if features is not None:
                X.append(features)
                y.append(label)

    if not X:
        raise ValueError("No samples loaded from dataset")

    _LOGGER.info(f"Loaded {len(X)} total samples")
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int64)


class WakeWordCNN(nn.Module):
    """Small convolutional classifier over log-mel spectrograms."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def split_train_val_test(X: np.ndarray, y: np.ndarray, seed: int = 42) -> tuple:
    """70/15/15 train/val/test split (stratified)."""
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.50, random_state=seed, stratify=y_rest
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def score_live_window(audio: np.ndarray, model: WakeWordCNN, norm: dict) -> float:
    """Positive-class confidence for a live audio window."""
    spec = log_mel_spectrogram(audio)
    spec = (spec - norm["mean"]) / norm["std"]
    x = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)

    return float(probs[0, 1].item())


def write_noise_negatives(output_dir, count: int = 300, seed: int = 0) -> int:
    """Write varied non-speech WAVs so the model learns to reject them.

    The generated dataset only contains spoken phrases, so a detector trained
    on it has never seen "no one is talking" and will happily classify room
    noise as a wake word. Covers the sound types a microphone actually picks
    up between utterances rather than white noise alone.
    """
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def broadband(n):
        return rng.normal(0, rng.uniform(0.02, 0.3), n)

    def hum(n):
        """Mains hum / fan / HVAC: a low tone plus harmonics."""
        t = np.arange(n) / SAMPLE_RATE
        base = rng.uniform(50, 120)
        sig = np.sin(2 * np.pi * base * t)
        for harmonic in (2, 3):
            sig += rng.uniform(0.2, 0.6) * np.sin(2 * np.pi * base * harmonic * t)
        return rng.uniform(0.05, 0.3) * sig + rng.normal(0, 0.01, n)

    def near_silence(n):
        return rng.normal(0, rng.uniform(0.0005, 0.01), n)

    def babble(n):
        """Speech-shaped noise: energy in the vocal band, no words."""
        spectrum = np.fft.rfft(rng.normal(0, 1, n))
        freqs = np.fft.rfftfreq(n, 1 / SAMPLE_RATE)
        shape = np.exp(-((freqs - 500) ** 2) / (2 * 700 ** 2)) + 0.3
        shaped = np.fft.irfft(spectrum * shape, n)
        peak = np.abs(shaped).max()
        return rng.uniform(0.05, 0.35) * shaped / (peak if peak > 1e-9 else 1.0)

    def transients(n):
        """Clicks, taps, keyboard: short bursts over a quiet floor."""
        sig = rng.normal(0, 0.005, n)
        for _ in range(rng.integers(2, 8)):
            pos = rng.integers(0, max(1, n - 400))
            length = rng.integers(40, 400)
            burst = rng.normal(0, rng.uniform(0.1, 0.5), length)
            burst *= np.exp(-np.linspace(0, 6, length))
            sig[pos:pos + length] += burst
        return sig

    def sweep(n):
        """Passing traffic / motor: a slowly gliding tone."""
        f0, f1 = rng.uniform(80, 200), rng.uniform(200, 600)
        freq = np.linspace(f0, f1, n)
        phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
        return rng.uniform(0.05, 0.25) * np.sin(phase) + rng.normal(0, 0.01, n)

    kinds = [broadband, hum, near_silence, babble, transients, sweep]

    for i in range(count):
        n = int(rng.uniform(0.6, 1.5) * SAMPLE_RATE)
        audio = np.clip(kinds[i % len(kinds)](n), -1.0, 1.0)
        pcm = (audio * 32767.0).astype(np.int16)

        with wave.open(str(output_dir / f"noise_{i}.wav"), "wb") as wav_file:
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.setsampwidth(2)
            wav_file.setnchannels(1)
            wav_file.writeframes(pcm.tobytes())

    return count


def train_model(
    positive_dir: Union[str, Path],
    negative_dir: Union[str, Path],
    output_model: Union[str, Path] = "wakeword_model.pkl",
    epochs: int = 60,
    patience: int = 10,
    lr: float = 1e-3,
    batch_size: int = 64,
    noise_negatives: int = 300,
) -> bool:
    """Train the CNN with a 70/15/15 split, keeping the best val-loss epoch."""
    if noise_negatives:
        existing = len(list(Path(negative_dir).glob("noise_*.wav")))
        if existing < noise_negatives:
            _LOGGER.info(f"Adding {noise_negatives} noise/silence negatives")
            write_noise_negatives(negative_dir, noise_negatives)

    _LOGGER.info("Loading dataset...")
    X, y = load_dataset(positive_dir, negative_dir)

    _LOGGER.info(f"Dataset: {len(X)} samples, spectrogram {X.shape[1]}x{X.shape[2]}")
    _LOGGER.info(
        f"Class distribution: {int(np.sum(y))} positive, {len(y) - int(np.sum(y))} negative"
    )

    X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(X, y)
    _LOGGER.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Normalize using training statistics only.
    mean = float(X_train.mean())
    std = float(X_train.std()) or 1.0
    norm = {"mean": mean, "std": std}
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _LOGGER.info(f"Training on: {device}")

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.long, device=device)

    model = WakeWordCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Negatives outnumber positives, which biases the model toward "reject"
    # and pushes true positives below 0.5 even when they separate cleanly.
    counts = np.bincount(y_train, minlength=2).astype(np.float64)
    weights = torch.tensor(
        counts.sum() / (2.0 * np.maximum(counts, 1)), dtype=torch.float32, device=device
    )
    _LOGGER.info(f"Class weights: negative={weights[0]:.2f}, positive={weights[1]:.2f}")
    criterion = nn.CrossEntropyLoss(weight=weights)

    def evaluate(X_t, y_t):
        model.eval()
        total_loss, correct = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(X_t), 256):
                xb, yb = X_t[start:start + 256], y_t[start:start + 256]
                logits = model(xb)
                total_loss += criterion(logits, yb).item() * len(xb)
                correct += (logits.argmax(dim=1) == yb).sum().item()
        return total_loss / len(X_t), correct / len(X_t)

    best_val_loss = float("inf")
    best_state, best_epoch = None, -1
    epochs_without_improvement = 0
    n_train = len(X_train_t)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0

        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(X_train_t[idx]), y_train_t[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)

        val_loss, val_acc = evaluate(X_val_t, y_val_t)
        _LOGGER.info(
            f"Epoch {epoch + 1}/{epochs} | train_loss={epoch_loss / n_train:.4f} "
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
                _LOGGER.info(f"Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(best_state)

    _, train_acc = evaluate(X_train_t, y_train_t)
    _, val_acc = evaluate(X_val_t, y_val_t)
    test_loss, test_acc = evaluate(X_test_t, y_test_t)

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
                "architecture": "cnn",
                "n_mels": N_MELS,
                "n_frames": N_FRAMES,
                "norm": norm,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "test_acc": test_acc,
            },
            f,
        )

    _LOGGER.info("Model trained successfully")
    return True
