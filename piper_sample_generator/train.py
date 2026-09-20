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
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from piper_sample_generator.features import (  # noqa: F401  (re-exported)
    DB_FLOOR,
    PAD_NOISE_MAX_DB,
    PAD_NOISE_MIN_DB,
    HOP_LENGTH,
    N_FFT,
    N_FRAMES,
    N_MELS,
    SAMPLE_RATE,
    apply_norm,
    compute_norm,
    extract_features,
    log_mel_spectrogram,
)

_LOGGER = logging.getLogger(__name__)


def load_dataset(
    positive_dir: Union[str, Path],
    negative_dir: Union[str, Path],
    pad_mode: str = "random",
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load spectrograms and labels from positive/negative directories.

    pad_mode defaults to "random", placing each clip at a random offset within
    the window. Nothing aligns the phrase at inference -- a live rolling buffer
    catches it wherever it falls -- so training every clip flush to the start
    would teach the model an alignment that never holds in practice. Each clip
    is read once, so a clip and a shifted copy of itself cannot straddle the
    train/test split.
    """
    rng = np.random.default_rng(seed)
    X, y = [], []

    for directory, label in ((positive_dir, 1), (negative_dir, 0)):
        kind = "positive" if label else "negative"
        _LOGGER.info(f"Loading {kind} samples from {directory}")
        for wav_file in sorted(Path(directory).glob("*.wav")):
            features = extract_features(wav_file, pad_mode=pad_mode, rng=rng)
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


def _separable_block(in_ch: int, out_ch: int, pool: bool = True) -> nn.Sequential:
    """Depthwise 3x3 followed by a pointwise 1x1, the MobileNet factorization.

    A dense 3x3 conv costs in_ch * out_ch * 9 weights; splitting it costs
    in_ch * 9 + in_ch * out_ch. At these widths that is roughly an eight-fold
    reduction for a small accuracy cost, which is the trade a microcontroller
    wants.
    """
    layers = [
        nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False),
        nn.BatchNorm2d(in_ch),
        nn.ReLU(),
        nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class WakeWordCNNTiny(nn.Module):
    """Wake word classifier sized for an ESP32-S3.

    Two constraints drove the shape, and neither is the parameter count.

    The first is peak activation. WakeWordCNN's opening conv produces
    32x40x128, which is 640 KB in float32 and does not fit in the SRAM budget
    however small the weights are. Striding the first conv by 2 in both axes
    cuts that to 32x20x64 before anything else runs.

    The second is that pooling the whole time axis into a single value, as
    WakeWordCNN does with AdaptiveAvgPool2d(1), discards the order of what was
    said. "limbo hey" and "hey limbo" contain the same sounds. The final pool
    here collapses frequency completely but keeps time in four coarse bins, so
    the ordering survives into the classifier. A fixed-kernel AvgPool2d is also
    what exports cleanly; adaptive pooling does not.
    """

    def __init__(self, n_mels: int = N_MELS, n_frames: int = N_FRAMES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            _separable_block(32, 32, pool=True),
            _separable_block(32, 64, pool=True),
            _separable_block(64, 64, pool=False),
        )

        # 40x128 -> stem 20x64 -> 10x32 -> 5x16, then pool frequency away and
        # keep four time bins.
        freq = n_mels // 8
        time = n_frames // 8
        self.time_bins = 4
        self.pool = nn.AvgPool2d(kernel_size=(freq, time // self.time_bins))

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64 * self.time_bins, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


class WakeWordCNNSmall(nn.Module):
    """A wider WakeWordCNNTiny, for when width rather than data is the limit.

    Same depth, same stride-2 stem, same four-time-bin pool: the only change
    is channel width, 32-64-96-128 against 32-32-64-64, plus a wider
    classifier. Depthwise-separable blocks make that cheap, since widening
    costs in_ch * out_ch in the pointwise convolution rather than nine times
    that.

    Peak activation is set by the stem and is therefore unchanged, which is
    the constraint that actually decides whether this runs on an ESP32-S3.
    The weights grow, and weights are the part that can live in flash.
    """

    def __init__(self, n_mels: int = N_MELS, n_frames: int = N_FRAMES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            _separable_block(32, 64, pool=True),
            _separable_block(64, 96, pool=True),
            _separable_block(96, 128, pool=False),
        )

        freq = n_mels // 8
        time = n_frames // 8
        self.time_bins = 4
        self.pool = nn.AvgPool2d(kernel_size=(freq, time // self.time_bins))

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * self.time_bins, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


ARCHITECTURES = {
    "cnn": WakeWordCNN,
    "tiny": WakeWordCNNTiny,
    "small": WakeWordCNNSmall,
}


def build_model(arch: str = "tiny") -> nn.Module:
    """Instantiate an architecture by name."""
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {arch!r}; expected one of "
                         f"{sorted(ARCHITECTURES)}")
    return ARCHITECTURES[arch]()


def load_model(model_file: Union[str, Path]) -> Tuple[nn.Module, dict]:
    """Load a trained model and its metadata from a .pkl file.

    Models saved before the tiny architecture existed have no "architecture"
    key and are always the original CNN.
    """
    with open(model_file, "rb") as f:
        data = pickle.load(f)

    model = build_model(data.get("architecture", "cnn"))
    model.load_state_dict(data["model_state_dict"])
    model.eval()
    return model, data


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def split_train_val_test(X: np.ndarray, y: np.ndarray, seed: int = 42) -> tuple:
    """70/15/15 train/val/test split (stratified)."""
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.50, random_state=seed, stratify=y_rest
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def score_live_window(audio: np.ndarray, model: nn.Module, norm: dict) -> float:
    """Positive-class confidence for a live audio window.

    Reads the most recent N_FRAMES of the buffer, since a rolling buffer fills
    from the right and the newest audio is what the caller is asking about.
    """
    spec = log_mel_spectrogram(audio, pad_mode="end")
    spec = apply_norm(spec, norm)
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

    def window_padding(n):
        """The exact noise short clips are padded with, as a negative.

        Without this the padding is only ever seen inside a positive window,
        and since positives are shorter than negatives they carry more of it,
        so its presence predicts the wake word. Drawing from the same range
        here puts the identical audio on the other side of the boundary.
        """
        floor_db = rng.uniform(PAD_NOISE_MIN_DB, PAD_NOISE_MAX_DB)
        return rng.normal(0, 10 ** (floor_db / 20), n)

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

    # window_padding appears twice: it is the one kind that has to outweigh
    # its own appearance inside every positive window, rather than merely be
    # represented among the sounds a microphone picks up.
    def digital_silence(n):
        """Exact zeros, which a muted or unplugged input delivers."""
        return np.zeros(n)

    kinds = [broadband, hum, near_silence, babble, transients, sweep,
             window_padding, window_padding, digital_silence]

    # Each kind is rescaled across a band spanning the padding levels.
    #
    # Measured, the kinds above land where their own amplitude constants put
    # them: broadband, hum, babble and sweep all sit between -37 and -10 dBFS,
    # entirely above the -65..-35 dB range short clips are padded with. So the
    # padding band was covered by white noise alone, and the model still had
    # room to read a quiet non-white room as speech. Rescaling keeps the
    # timbres and makes every one of them cover the band that matters.
    QUIET_DB, LOUD_DB = -72.0, -15.0

    for i in range(count):
        # Padding negatives fill the window outright, so that "the whole
        # window is this noise" is itself a labelled example.
        if kinds[i % len(kinds)] in (window_padding, digital_silence):
            n = int(rng.uniform(1.3, 1.6) * SAMPLE_RATE)
        else:
            n = int(rng.uniform(0.6, 1.5) * SAMPLE_RATE)
        kind = kinds[i % len(kinds)]
        audio = kind(n)

        if kind is not digital_silence:
            rms = float(np.sqrt(np.mean(audio ** 2)))
            if rms > 1e-9:
                target = 10 ** (rng.uniform(QUIET_DB, LOUD_DB) / 20)
                audio = audio * (target / rms)

        audio = np.clip(audio, -1.0, 1.0)
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
    noise_negatives: int = 1500,
    arch: str = "tiny",
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

    # Normalize using training statistics only, per mel bin.
    norm = compute_norm(X_train)
    X_train = apply_norm(X_train, norm)
    X_val = apply_norm(X_val, norm)
    X_test = apply_norm(X_test, norm)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _LOGGER.info(f"Training on: {device}")

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.long, device=device)

    model = build_model(arch).to(device)
    _LOGGER.info(f"Architecture: {arch} ({count_parameters(model):,} parameters)")
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
                "architecture": arch,
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
