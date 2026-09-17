#!/usr/bin/env python3
"""Generate complete training dataset with positive and negative variations."""

import argparse
import logging
from pathlib import Path

from piper_sample_generator import generate_samples_onnx
from piper_sample_generator.variations import generate_wakeword_variations

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)


def main() -> int:
    """Generate training dataset with variations."""
    parser = argparse.ArgumentParser(
        description="Generate wake word training dataset with positive/negative variations"
    )
    parser.add_argument("phrase", help="Wake word phrase (e.g., 'hey limbo')")
    parser.add_argument(
        "--model",
        required=True,
        action="append",
        help="Path to Piper voice model (.onnx)",
    )
    parser.add_argument(
        "--samples-per-variation",
        type=int,
        default=10,
        help="Audio samples per phrase variation",
    )
    parser.add_argument(
        "--output-dir",
        default="training_dataset",
        help="Output directory for training data",
    )
    parser.add_argument(
        "--length-scales",
        nargs="+",
        type=float,
        default=[0.75, 1.0, 1.25],
        help="Speaking speed variations",
    )
    parser.add_argument(
        "--noise-scales",
        nargs="+",
        type=float,
        default=[0.667, 0.75],
        help="Audio variation levels",
    )
    args = parser.parse_args()

    phrase = args.phrase.strip()
    output_base = Path(args.output_dir)

    # Generate variations
    _LOGGER.info(f"Generating variations for: {phrase}")
    positive, negative = generate_wakeword_variations(phrase)

    _LOGGER.info(f"Positive variations: {len(positive)}")
    _LOGGER.info(f"Negative variations: {len(negative)}")

    # Create output directories
    positive_dir = output_base / "positive"
    negative_dir = output_base / "negative"
    positive_dir.mkdir(parents=True, exist_ok=True)
    negative_dir.mkdir(parents=True, exist_ok=True)

    # Generate positive samples
    _LOGGER.info(f"\nGenerating POSITIVE samples ({len(positive)} variations)...")
    total_positive = len(positive) * args.samples_per_variation
    try:
        generate_samples_onnx(
            positive,
            positive_dir,
            args.model,
            max_samples=total_positive,
            length_scales=tuple(args.length_scales),
            noise_scales=tuple(args.noise_scales),
        )
        _LOGGER.info(f"✓ Generated {total_positive} positive samples")
    except Exception as e:
        _LOGGER.error(f"Error generating positive samples: {e}")
        return 1

    # Generate negative samples
    _LOGGER.info(f"\nGenerating NEGATIVE samples ({len(negative)} variations)...")
    total_negative = len(negative) * args.samples_per_variation
    try:
        generate_samples_onnx(
            negative,
            negative_dir,
            args.model,
            max_samples=total_negative,
            length_scales=tuple(args.length_scales),
            noise_scales=tuple(args.noise_scales),
        )
        _LOGGER.info(f"✓ Generated {total_negative} negative samples")
    except Exception as e:
        _LOGGER.error(f"Error generating negative samples: {e}")
        return 1

    # Summary
    _LOGGER.info("\n" + "=" * 60)
    _LOGGER.info("TRAINING DATASET SUMMARY")
    _LOGGER.info("=" * 60)
    _LOGGER.info(f"Wake word: {phrase}")
    _LOGGER.info(f"Positive samples: {total_positive}")
    _LOGGER.info(f"  - Variations: {len(positive)}")
    _LOGGER.info(f"  - Samples/variation: {args.samples_per_variation}")
    _LOGGER.info(f"Negative samples: {total_negative}")
    _LOGGER.info(f"  - Variations: {len(negative)}")
    _LOGGER.info(f"  - Samples/variation: {args.samples_per_variation}")
    _LOGGER.info(f"Total samples: {total_positive + total_negative}")
    _LOGGER.info(f"Output directory: {output_base.absolute()}")
    _LOGGER.info("=" * 60)

    _LOGGER.info("\nNext: Run augmentation to add acoustic variation:")
    _LOGGER.info(
        f"python3 -m piper_sample_generator.augment "
        f"--sample-rate 16000 {positive_dir} {positive_dir}_augmented/"
    )
    _LOGGER.info(
        f"python3 -m piper_sample_generator.augment "
        f"--sample-rate 16000 {negative_dir} {negative_dir}_augmented/"
    )

    return 0


if __name__ == "__main__":
    exit(main())
