#!/usr/bin/env bash
# Set up the PyTorch -> ONNX -> TFLite int8 export toolchain under WSL.
#
# This runs in WSL rather than native Windows because the TensorFlow and
# onnx2tf stack is unreliable there: recent tensorflow wheels for Windows are
# CPU-only and lag behind, and onnx2tf depends on tensorflow addons that are
# not built for Windows at all.
#
# Usage, from a WSL Ubuntu-24.04 shell:
#   cd /mnt/c/Users/saumi/Desktop/CODES/piper-sample-generator
#   bash scripts/setup_wsl_export.sh
#   source .venv-export/bin/activate
#   python export/to_tflite.py wakeword_tiny.pkl --dataset hey_limbo_v3
#   python export/verify_parity.py wakeword_tiny.pkl export/build/wakeword_int8.tflite

set -euo pipefail

VENV="${VENV:-.venv-export}"

if ! command -v python3.11 >/dev/null 2>&1; then
    echo "Installing Python 3.11 (onnx2tf and tensorflow do not both support 3.12 yet)"
    sudo apt-get update
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
fi

# Kept separate from the Windows .venv: that one lives on an NTFS mount with
# Windows binaries in it, which a Linux interpreter cannot use.
python3.11 -m venv "$VENV"
# shellcheck disable=SC1090
source "$VENV/bin/activate"

pip install --upgrade pip wheel

# torch CPU-only. The export path never trains, and the CUDA wheels are ~2.5 GB.
pip install --index-url https://download.pytorch.org/whl/cpu torch

pip install \
    "tensorflow==2.18.*" \
    onnx \
    onnxruntime \
    onnxsim \
    onnx2tf \
    "onnx-graphsurgeon" \
    "sng4onnx" \
    librosa \
    scikit-learn \
    numpy

echo
echo "Done. Activate with: source $VENV/bin/activate"
python -c "import tensorflow as tf, torch, onnx; \
print('tensorflow', tf.__version__); \
print('torch', torch.__version__); \
print('onnx', onnx.__version__)"
