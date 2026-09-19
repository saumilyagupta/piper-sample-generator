#!/usr/bin/env python3
"""Assemble a wake word training set from both synthesis engines.

Two constraints shape this, and the naive approach violates both.

Engine balance. Positive and negative clips must be drawn from the same mix of
synthesis engines. If positives came from Piper and negatives from Deepgram,
the cleanest way to separate the classes would be vocoder fingerprint rather
than what was said, and the model would learn exactly that -- scoring well in
evaluation and failing completely on a real voice.

Sample budget. The package CLI splits --max-samples across phrases in
proportion to how many there are. With 3 positive phrases against 67 negative
ones, positives would receive 4% of the budget. This generates each class to an
explicit target instead.

Existing Deepgram clips are reused rather than resynthesized: the phrase list
shrank but the audio for the surviving phrases is unchanged, and the API key
used to produce it should be rotated rather than used again. Their filenames
carry no phrase, but the generator enumerated jobs as
[(phrase, voice) for voice in voices for phrase in phrases], so clip N holds
phrase N % len(phrases) against the original 87-phrase list.
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import List

# Running a file under scripts/ puts scripts/ on sys.path rather than the repo
# root, which hides the top-level piper_train package the generator imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from piper_sample_generator.augment import augment_directory
from piper_sample_generator.variations import generate_wakeword_variations

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOGGER = logging.getLogger(__name__)

SAMPLE_RATE = 16000


def copy_deepgram_clips(
    source_dir: Path, output_dir: Path, all_phrases: List[str], keep: List[str]
) -> int:
    """Copy previously synthesized Deepgram clips for the surviving phrases."""
    import re

    keep_indices = {all_phrases.index(p) for p in keep if p in all_phrases}
    if not keep_indices:
        _LOGGER.warning(f"No phrases from {output_dir.name} found in the original list")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in sorted(source_dir.glob("dg_*.wav")):
        index = int(re.search(r"dg_(\d+)", path.name).group(1))
        if index % len(all_phrases) in keep_indices:
            shutil.copy(path, output_dir / path.name)
            copied += 1

    _LOGGER.info(f"Copied {copied} Deepgram clips into {output_dir}")
    return copied


def generate_piper_clips(
    phrases: List[str], output_dir: Path, model: str, count: int
) -> None:
    """Synthesize phrases with the local Piper LibriTTS voice."""
    from piper_sample_generator.__main__ import generate_samples

    output_dir.mkdir(parents=True, exist_ok=True)
    _LOGGER.info(f"Synthesizing {count} Piper clips for {len(phrases)} phrases "
                 f"into {output_dir}")
    generate_samples(
        text=phrases,
        output_dir=str(output_dir),
        model=model,
        max_samples=count,
        batch_size=16,
        slerp_weights=(0.5,),
        length_scales=(0.75, 1.0, 1.25, 1.4),
        noise_scales=(0.667, 0.85, 1.0),
        noise_scale_ws=(0.8,),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phrase", default="hey limbo")
    parser.add_argument("--output-dir", default="hey_limbo_v3")
    parser.add_argument("--source-dataset", default="hey_limbo_v2",
                        help="Dataset holding the existing Deepgram clips")
    parser.add_argument("--original-phrases", default="neg_phrases.txt",
                        help="The 87-phrase list the Deepgram clips were generated from")
    parser.add_argument("--model", default="models/en-us-libritts-high.pt")
    parser.add_argument("--piper-positives", type=int, default=3200)
    parser.add_argument("--piper-negatives", type=int, default=3400)
    parser.add_argument("--skip-generate", action="store_true",
                        help="Reuse an existing raw dataset and only re-augment")
    args = parser.parse_args()

    positive, negative = generate_wakeword_variations(args.phrase)
    _LOGGER.info(f"{len(positive)} positive phrases, {len(negative)} negative phrases")

    out = Path(args.output_dir)
    source = Path(args.source_dataset)
    all_phrases = [
        p.strip() for p in Path(args.original_phrases).read_text().splitlines() if p.strip()
    ]

    # Each engine gets its own directory so the two can be augmented with
    # different copy counts. Deepgram contributes 53 voices but only 159
    # positive clips, against thousands from Piper; expanding the smaller set
    # is what keeps both engines equally represented inside each class.
    if not args.skip_generate:
        copy_deepgram_clips(source / "positive", out / "positive_dg", positive, positive)
        copy_deepgram_clips(source / "negative", out / "negative_dg", all_phrases, negative)

        generate_piper_clips(positive, out / "positive_piper", args.model,
                             args.piper_positives)
        generate_piper_clips(negative, out / "negative_piper", args.model,
                             args.piper_negatives)

    for kind in ("positive", "negative"):
        dst = out / f"{kind}_aug"
        counts = {
            engine: len(list((out / f"{kind}_{engine}").glob("*.wav")))
            for engine in ("dg", "piper")
        }
        target = max(counts.values())

        for engine, n in counts.items():
            if not n:
                continue
            # Round up, so the smaller engine reaches at least parity.
            copies = max(1, -(-target // n))
            _LOGGER.info(f"Augmenting {n} {kind} {engine} clips x{copies} -> {dst}")
            augment_directory(out / f"{kind}_{engine}", dst,
                              sample_rate=SAMPLE_RATE, copies=copies)

    for kind in ("positive", "negative"):
        files = list((out / f"{kind}_aug").glob("*.wav"))
        dg = sum(1 for f in files if f.name.startswith("dg_"))
        _LOGGER.info(f"{kind}_aug: {len(files)} clips "
                     f"({dg} deepgram, {len(files) - dg} piper)")


if __name__ == "__main__":
    main()
