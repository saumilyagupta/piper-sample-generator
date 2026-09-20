#!/usr/bin/env bash
# Set up the PyTorch -> ONNX -> TFLite int8 export toolchain under WSL.
#
# This runs in WSL rather than native Windows because the TensorFlow and
# onnx2tf stack is unreliable there: recent tensorflow wheels for Windows are
# CPU-only and lag behind, and onnx2tf depends on packages that are not built
# for Windows at all.
#
# No sudo required. Ubuntu 24.04 ships Python 3.12 without the python3-venv
# package, so `python3 -m venv` fails on the missing ensurepip, and PEP 668
# blocks `pip install --user` into the system interpreter. The way through is
# to create the environment with --without-pip, which needs only the stdlib
# venv module, and bootstrap pip inside it with get-pip.py. Nothing outside
# the environment is modified.
#
# The environment lives in $HOME rather than the repo: the repo sits on an
# NTFS mount under /mnt/c, where package installs are slow and symlinks in
# site-packages do not behave.
#
# Usage, from a WSL Ubuntu-24.04 shell:
#   cd /mnt/c/Users/saumi/Desktop/CODES/piper-sample-generator
#   bash scripts/setup_wsl_export.sh
#   ~/venv-export/bin/python export/to_tflite.py wakeword_tiny.pkl --dataset hey_limbo_v4
#   ~/venv-export/bin/python export/verify_parity.py wakeword_tiny.pkl \
#       export/build/wakeword_int8.tflite --dataset hey_limbo_v4

set -euo pipefail

VENV="${VENV:-$HOME/venv-export}"
PY="$VENV/bin/python"

if [ ! -x "$PY" ]; then
    python3 -m venv --without-pip "$VENV"
    curl -sS -o /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py
    "$PY" /tmp/get-pip.py -q
fi

"$PY" -m pip install -q --upgrade pip wheel setuptools

# CPU-only torch: the export path never trains, and the CUDA wheels are ~2.5 GB.
"$PY" -m pip install -q --index-url https://download.pytorch.org/whl/cpu torch

"$PY" -m pip install -q \
    "tensorflow==2.18.*" \
    onnx \
    onnxruntime \
    onnxsim \
    onnx2tf \
    onnx_graphsurgeon \
    sng4onnx \
    librosa \
    scikit-learn

echo
"$PY" -c 'import tensorflow as tf, torch, onnx
print("tensorflow", tf.__version__)
print("torch     ", torch.__version__)
print("onnx      ", onnx.__version__)'
echo
echo "Done. Interpreter: $PY"
