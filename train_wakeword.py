#!/usr/bin/env python3
"""Wake word detector training with train/val/test split and best-val-loss checkpointing."""

import copy
import logging
import pickle
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)

N_MFCC = 13
MAX_FRAMES = 100


def extract_mfcc_features(wav_path: str, n_mfcc: int = N_MFCC, max_frames: int = MAX_FRAMES) -> np.ndarray:
    """Extract fixed-length MFCC features from an audio file."""
    try:
        y, sr = librosa.load(wav_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        if mfcc.shape[1] < max_frames:
            mfcc = np.pad(mfcc, ((0, 0), (0, max_frames - mfcc.shape[1])), mode="constant")
        else:
            mfcc = mfcc[:, :max_frames]

        return mfcc.flatten()
    except Exception as e:
        _LOGGER.warning(f"Error processing {wav_path}: {e}")
        return None


def load_dataset(positive_dir: str, negative_dir: str) -> tuple:
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
    positive_dir: str,
    negative_dir: str,
    output_model: str = "wakeword_model.pkl",
    epochs: int = 100,
    patience: int = 15,
    lr: float = 1e-3,
    batch_size: int = 32,
) -> bool:
    """Train MLP classifier with 70/15/15 split, checkpointing best val-loss model."""
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
    criterion = nn.CrossEntropyLoss()

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
                "max_frames": MAX_FRAMES,
                "scaler": scaler,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "test_acc": test_acc,
            },
            f,
        )

    _LOGGER.info("✓ Model trained successfully")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train wake word detector (70/15/15 split, best val-loss checkpoint)")
    parser.add_argument("positive_dir", help="Directory with positive samples")
    parser.add_argument("negative_dir", help="Directory with negative samples")
    parser.add_argument("--output", default="wakeword_model.pkl", help="Output model file")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (epochs)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    args = parser.parse_args()

    success = train_model(
        args.positive_dir,
        args.negative_dir,
        args.output,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        batch_size=args.batch_size,
    )
    exit(0 if success else 1)
