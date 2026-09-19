#!/usr/bin/env bash
# Set up microWakeWord (Track B) under WSL.
#
# microWakeWord is the framework behind ESPHome's on-device wake word support.
# It produces streaming int8 models for TensorFlow Lite Micro that run an
# inference in under 10 ms on an ESP32-S3, and it already consumes this repo:
# piper-sample-generator is its documented source of positive samples.
#
# Two things it has that our from-scratch model does not, and which are the
# reason to run it at all:
#
#   Streaming inference. It scores each new 20 ms slice against retained state
#   rather than recomputing a full window, so it is genuinely continuous rather
#   than a windowed classifier re-run on a timer.
#
#   Real ambient negatives. Pre-computed spectrogram features from AudioSet,
#   FMA, Common Voice and DiPCo -- music, household noise, television,
#   conversation. Our dataset is entirely synthetic speech plus procedurally
#   generated noise, and false accepts per hour cannot be measured, let alone
#   minimized, without real background audio.
#
# Its training loop optimizes for that directly: it drives
# ambient_false_positives_per_hour below a target first, and only then
# maximizes accuracy.
#
# Usage, from a WSL Ubuntu-24.04 shell:
#   cd /mnt/c/Users/saumi/Desktop/CODES/piper-sample-generator
#   bash scripts/setup_wsl_mww.sh

set -euo pipefail

ROOT="${ROOT:-$HOME/mww}"
VENV="$ROOT/.venv"

if ! command -v python3.11 >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
fi

# Deliberately outside the repo and outside the Windows .venv. That one is
# Python 3.12 Windows binaries on an NTFS mount; this needs Linux 3.11, and
# mixing a TensorFlow install into the PyTorch training environment invites
# version conflicts in both.
mkdir -p "$ROOT"
cd "$ROOT"

if [ ! -d micro-wake-word ]; then
    git clone https://github.com/OHF-Voice/micro-wake-word.git
fi

python3.11 -m venv "$VENV"
# shellcheck disable=SC1090
source "$VENV/bin/activate"
pip install --upgrade pip wheel

pip install -e ./micro-wake-word
pip install pymicro-features mmap-ninja datasets huggingface_hub

# Real ambient negative features. Large (tens of GB); pulled once and reused.
# These must match the microfrontend settings the positive features are
# generated with, or the two sets are not comparable:
#   sample_rate=16000 window_size=30 window_step=20 num_channels=40
#   upper_band_limit=7500 lower_band_limit=125 enable_pcan=True
#   min_signal_remaining=0.05 out_scale=1 out_type=uint16
echo
echo "Fetching ambient negative features from huggingface..."
python - <<'PY'
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="kahrendt/microwakeword",
    repo_type="dataset",
    allow_patterns=["*"],
)
print("ambient features at:", path)
PY

echo
echo "Done. Next:"
echo "  source $VENV/bin/activate"
echo "  # generate positives with this repo, then convert wav dirs to features"
echo "  # and train:  python micro-wake-word/microwakeword/model_train_eval.py mixednet \\"
echo "  #               --training_config mww/training_parameters.yaml \\"
echo "  #               --train 1 --test_tflite_streaming_quantized 1"
