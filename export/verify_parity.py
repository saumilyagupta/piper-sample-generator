#!/usr/bin/env python3
"""Check the exported int8 model still agrees with the PyTorch model it came from.

Quantization failures are quiet. The usual ones are an int8 model that returns
a constant for every input, because activation ranges were calibrated on data
unlike the real thing, and a model whose output is subtly shifted, because a
layer fused differently after conversion. Neither raises an error; both destroy
detection while the file still loads and runs.

So the exported model is compared against the original on identical inputs. The
gates are agreement on the predicted class for at least 99% of clips and a
maximum probability difference under 0.02. A constant-output model fails the
first immediately, since it cannot track the float model across both classes.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from piper_sample_generator.features import N_FRAMES, N_MELS, apply_norm  # noqa: E402
from piper_sample_generator.train import load_dataset, load_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_LOGGER = logging.getLogger(__name__)

MAX_PROB_DELTA = 0.02
MIN_AGREEMENT = 0.99


def torch_probabilities(model, X: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        return torch.softmax(logits, dim=1).numpy()


def tflite_probabilities(tflite_path: Path, X: np.ndarray) -> np.ndarray:
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    in_scale, in_zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]

    probs = []
    for sample in X:
        # onnx2tf emits NHWC.
        value = sample.reshape(1, N_MELS, N_FRAMES, 1).astype(np.float32)
        quantized = np.clip(
            np.round(value / in_scale + in_zero), -128, 127
        ).astype(inp["dtype"])

        interpreter.set_tensor(inp["index"], quantized)
        interpreter.invoke()

        logits = (
            interpreter.get_tensor(out["index"]).astype(np.float32) - out_zero
        ) * out_scale
        exp = np.exp(logits - logits.max())
        probs.append((exp / exp.sum()).ravel())

    return np.stack(probs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Trained model .pkl")
    parser.add_argument("tflite", help="Exported int8 .tflite")
    parser.add_argument("--dataset", default="hey_limbo_v3")
    parser.add_argument("--samples", type=int, default=200)
    args = parser.parse_args()

    model, data = load_model(args.model)

    dataset = Path(args.dataset)
    X, y = load_dataset(dataset / "positive_aug", dataset / "negative_aug")
    X = apply_norm(X, data["norm"])

    rng = np.random.default_rng(1)
    picks = []
    for label in (0, 1):
        idx = np.flatnonzero(y == label)
        picks.append(rng.choice(idx, size=min(args.samples // 2, len(idx)), replace=False))
    picks = np.concatenate(picks)
    X, y = X[picks], y[picks]

    reference = torch_probabilities(model, X)
    exported = tflite_probabilities(Path(args.tflite), X)

    delta = np.abs(reference - exported).max()
    agreement = float((reference.argmax(1) == exported.argmax(1)).mean())
    ref_acc = float((reference.argmax(1) == y).mean())
    exp_acc = float((exported.argmax(1) == y).mean())
    distinct = len(np.unique(exported.argmax(1)))

    print()
    print(f"  samples             {len(X)}")
    print(f"  max prob delta      {delta:.4f}  (gate < {MAX_PROB_DELTA})")
    print(f"  class agreement     {agreement:.4f}  (gate >= {MIN_AGREEMENT})")
    print(f"  pytorch accuracy    {ref_acc:.4f}")
    print(f"  tflite  accuracy    {exp_acc:.4f}")
    print(f"  distinct predictions {distinct}  (1 means a collapsed model)")

    failures = []
    if distinct < 2:
        failures.append("exported model predicts a single class for every input")
    if agreement < MIN_AGREEMENT:
        failures.append(f"class agreement {agreement:.4f} < {MIN_AGREEMENT}")
    if delta >= MAX_PROB_DELTA:
        failures.append(f"max probability delta {delta:.4f} >= {MAX_PROB_DELTA}")

    print()
    if failures:
        for failure in failures:
            print(f"  FAIL: {failure}")
        sys.exit(1)

    print("  PASS: exported model matches the PyTorch model")


if __name__ == "__main__":
    main()
