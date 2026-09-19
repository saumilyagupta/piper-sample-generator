#!/usr/bin/env python3
"""Export a trained wake word model to full-integer int8 TFLite for ESP32-S3.

Route: PyTorch -> ONNX -> (onnx2tf) -> TensorFlow SavedModel -> TFLite int8.

The conversion is split rather than handed to onnx2tf's own quantizer so the
quantization step stays under direct control: TFLite Micro needs int8 in and
int8 out with no float operations left anywhere, and a converter that silently
leaves a float input or a float softmax at the boundary produces a model the
interpreter will refuse to allocate.

Quantization needs a representative dataset -- real normalized spectrograms,
not random noise. Activation ranges are calibrated from whatever it is shown,
so noise yields ranges that do not match real audio and the quantized model
collapses to a constant output. That is the usual cause of an int8 model that
returns the same answer for every input.

This runs under WSL, not native Windows: the TensorFlow and onnx2tf toolchain
is unreliable on Windows. See scripts/setup_wsl_export.sh.
"""

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from piper_sample_generator.features import N_FRAMES, N_MELS, apply_norm  # noqa: E402
from piper_sample_generator.train import load_dataset, load_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_LOGGER = logging.getLogger(__name__)

REPRESENTATIVE_SAMPLES = 300


def build_representative_data(dataset_dir: Path, norm: dict, count: int) -> np.ndarray:
    """Normalized spectrograms spanning both classes, for range calibration."""
    X, y = load_dataset(dataset_dir / "positive_aug", dataset_dir / "negative_aug")
    X = apply_norm(X, norm)

    rng = np.random.default_rng(0)
    # Draw from both classes: calibrating on negatives alone would miss the
    # activation ranges a wake word actually produces.
    picks = []
    for label in (0, 1):
        idx = np.flatnonzero(y == label)
        picks.append(rng.choice(idx, size=min(count // 2, len(idx)), replace=False))
    picks = np.concatenate(picks)

    return X[picks].astype(np.float32)


def export_onnx(model, onnx_path: Path) -> None:
    import torch

    dummy = torch.zeros(1, 1, N_MELS, N_FRAMES, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["spectrogram"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,   # folds BatchNorm into the preceding conv
        dynamo=False,
    )
    _LOGGER.info(f"Wrote {onnx_path} ({onnx_path.stat().st_size / 1024:.1f} KB)")


def simplify_onnx(onnx_path: Path) -> None:
    try:
        import onnx
        from onnxsim import simplify
    except ImportError:
        _LOGGER.warning("onnxsim not installed, skipping simplification")
        return

    model = onnx.load(str(onnx_path))
    simplified, ok = simplify(model)
    if ok:
        onnx.save(simplified, str(onnx_path))
        _LOGGER.info("Simplified ONNX graph")
    else:
        _LOGGER.warning("ONNX simplification failed, continuing with the original")


def onnx_to_saved_model(onnx_path: Path, saved_model_dir: Path) -> None:
    if saved_model_dir.exists():
        shutil.rmtree(saved_model_dir)

    # onnx2tf is invoked as a subprocess: importing it pulls in a TensorFlow
    # session that conflicts with the converter set up later in this process.
    subprocess.run(
        [sys.executable, "-m", "onnx2tf", "-i", str(onnx_path),
         "-o", str(saved_model_dir), "-nuo", "--non_verbose"],
        check=True,
    )
    _LOGGER.info(f"Wrote SavedModel to {saved_model_dir}")


def saved_model_to_int8(
    saved_model_dir: Path, tflite_path: Path, representative: np.ndarray
) -> None:
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    def representative_dataset():
        for sample in representative:
            # onnx2tf emits NHWC, so the channel axis moves to the end.
            yield [sample.reshape(1, N_MELS, N_FRAMES, 1).astype(np.float32)]

    converter.representative_dataset = representative_dataset

    tflite_path.write_bytes(converter.convert())
    _LOGGER.info(f"Wrote {tflite_path} ({tflite_path.stat().st_size / 1024:.1f} KB)")


def report(tflite_path: Path) -> None:
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    arena = sum(
        int(np.prod(d["shape"])) * np.dtype(d["dtype"]).itemsize
        for d in interpreter.get_tensor_details()
        if d["shape"] is not None and len(d["shape"])
    )

    print()
    print(f"  model size       {tflite_path.stat().st_size / 1024:8.1f} KB")
    print(f"  input            {inp['dtype'].__name__} {tuple(inp['shape'])}")
    print(f"  output           {out['dtype'].__name__} {tuple(out['shape'])}")
    print(f"  tensor footprint {arena / 1024:8.1f} KB (upper bound on the arena)")
    float_ops = [
        d["name"] for d in interpreter.get_tensor_details()
        if d["dtype"] == np.float32
    ]
    print(f"  float32 tensors  {len(float_ops)}"
          f"{' <-- not fully quantized' if float_ops else ' (fully quantized)'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Trained model .pkl")
    parser.add_argument("--dataset", default="hey_limbo_v3",
                        help="Dataset directory for the representative set")
    parser.add_argument("--out-dir", default="export/build")
    parser.add_argument("--samples", type=int, default=REPRESENTATIVE_SAMPLES)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model, data = load_model(args.model)
    _LOGGER.info(f"Loaded {args.model} (arch={data.get('architecture')}, "
                 f"test_acc={data.get('test_acc')})")

    representative = build_representative_data(
        Path(args.dataset), data["norm"], args.samples
    )
    np.save(out / "representative.npy", representative)
    _LOGGER.info(f"Representative set: {representative.shape}")

    onnx_path = out / "wakeword.onnx"
    export_onnx(model, onnx_path)
    simplify_onnx(onnx_path)

    saved_model_dir = out / "saved_model"
    onnx_to_saved_model(onnx_path, saved_model_dir)

    tflite_path = out / "wakeword_int8.tflite"
    saved_model_to_int8(saved_model_dir, tflite_path, representative)
    report(tflite_path)


if __name__ == "__main__":
    main()
