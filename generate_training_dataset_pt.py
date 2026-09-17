#!/usr/bin/env python3
"""Generate training dataset with positive/negative variations (PyTorch generator)."""

import argparse
import logging
from pathlib import Path

from piper_sample_generator.__main__ import generate_samples
from piper_sample_generator.variations import generate_wakeword_variations

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)


def main() -> int:
    """Generate training dataset with variations using PyTorch model."""
    parser = argparse.ArgumentParser(
        description="Generate wake word training dataset (PyTorch .pt model)"
    )
    parser.add_argument("phrase", help="Wake word phrase (e.g., 'hey limbo')")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to PyTorch generator model (.pt)",
    )
    parser.add_argument(
        "--samples-per-variation",
        type=int,
        default=5,
        help="Audio samples per phrase variation",
    )
    parser.add_argument(
        "--output-dir",
        default="training_dataset",
        help="Output directory for training data",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="GPU batch size",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=100,
        help="Max LibriTTS speakers to use",
    )
    parser.add_argument(
        "--length-scales",
        nargs="+",
        type=float,
        default=[0.75, 1.0, 1.25],
        help="Speaking speed variations",
    )
    parser.add_argument(
        "--slerp-weights",
        nargs="+",
        type=float,
        default=[0.3, 0.5, 0.7],
        help="Speaker blending weights",
    )
    parser.add_argument(
        "--noise-scales",
        nargs="+",
        type=float,
        default=[0.667],
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
        generate_samples(
            positive,
            positive_dir,
            args.model,
            max_samples=total_positive,
            batch_size=args.batch_size,
            length_scales=tuple(args.length_scales),
            slerp_weights=tuple(args.slerp_weights),
            noise_scales=tuple(args.noise_scales),
            max_speakers=args.max_speakers,
        )
        _LOGGER.info(f"✓ Generated {total_positive} positive samples")
    except Exception as e:
        _LOGGER.error(f"Error generating positive samples: {e}")
        return 1

    # Generate negative samples
    _LOGGER.info(f"\nGenerating NEGATIVE samples ({len(negative)} variations)...")
    total_negative = len(negative) * args.samples_per_variation
    try:
        generate_samples(
            negative,
            negative_dir,
            args.model,
            max_samples=total_negative,
            batch_size=args.batch_size,
            length_scales=tuple(args.length_scales),
            slerp_weights=tuple(args.slerp_weights),
            noise_scales=tuple(args.noise_scales),
            max_speakers=args.max_speakers,
        )
        _LOGGER.info(f"✓ Generated {total_negative} negative samples")
    except Exception as e:
        _LOGGER.error(f"Error generating negative samples: {e}")
        return 1

    # Summary
    _LOGGER.info("\n" + "=" * 70)
    _LOGGER.info("TRAINING DATASET GENERATED")
    _LOGGER.info("=" * 70)
    _LOGGER.info(f"Wake word phrase: {phrase}")
    _LOGGER.info(f"\nPOSITIVE SAMPLES: {total_positive}")
    _LOGGER.info(f"  Phonetic variations: {len(positive)}")
    _LOGGER.info(f"  Samples per variation: {args.samples_per_variation}")
    _LOGGER.info(f"  Directory: {positive_dir}")
    _LOGGER.info(f"\nNEGATIVE SAMPLES: {total_negative}")
    _LOGGER.info(f"  Phonetic variations: {len(negative)}")
    _LOGGER.info(f"  Samples per variation: {args.samples_per_variation}")
    _LOGGER.info(f"  Directory: {negative_dir}")
    _LOGGER.info(f"\nTOTAL: {total_positive + total_negative} samples")
    _LOGGER.info(f"Dataset ratio: {100*total_positive/(total_positive+total_negative):.0f}% positive")
    _LOGGER.info("=" * 70)

    _LOGGER.info("\n→ Next step: Augment with room acoustics & volume variation:")
    _LOGGER.info(
        f"python3 -m piper_sample_generator.augment "
        f"--sample-rate 16000 {positive_dir} {positive_dir}_aug/"
    )
    _LOGGER.info(
        f"python3 -m piper_sample_generator.augment "
        f"--sample-rate 16000 {negative_dir} {negative_dir}_aug/"
    )

    return 0


if __name__ == "__main__":
    exit(main())
