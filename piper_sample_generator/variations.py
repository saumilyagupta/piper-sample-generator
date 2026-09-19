#!/usr/bin/env python3
"""Generate wake word training phrases: the target phrase and its confusables."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

# Written by analysis/probe_negative_separability.py --all.
DEFAULT_SEPARABILITY_FILE = "neg_sims_temporal.json"


def _doubled_character_variants(words: List[str]) -> Set[str]:
    """Spellings that repeat a character, e.g. "hey limbo" -> "hey limmbo".

    These are generated only so they can be excluded. Text-to-speech engines
    collapse repeated letters, so every one of them is rendered as the wake
    word itself while carrying a negative label. Measured against "hey limbo",
    all seven scored above the wake word's own self-similarity -- the most
    degenerate group in the whole negative set.
    """
    variants = set()
    for i, word in enumerate(words):
        for pos in range(len(word)):
            doubled = words.copy()
            doubled[i] = word[:pos] + word[pos] + word[pos:]
            variants.add(" ".join(doubled))
    return variants


def load_separability(
    path: Optional[Union[str, Path]] = None
) -> Optional[Tuple[Dict[str, float], float]]:
    """Load measured negative-phrase separability scores, if available.

    Returns (scores, baseline) or None when no measurement file exists. The
    baseline is the wake word's similarity to itself across different voices;
    a negative scoring at or above it cannot be told apart from the wake word
    by any model, so training on it only injects label noise.
    """
    path = Path(path or DEFAULT_SEPARABILITY_FILE)
    if not path.exists():
        return None

    data = json.loads(path.read_text())
    sims = data.get("sims")
    baseline = data.get("baseline")
    if not sims or baseline is None:
        return None
    return sims, float(baseline)


def filter_negatives_by_separability(
    negatives: Set[str], separability: Optional[Tuple[Dict[str, float], float]]
) -> Set[str]:
    """Drop negatives measured to be indistinguishable from the wake word.

    Phrases with no measurement are kept: absence of evidence is not evidence
    that a phrase is unlearnable.
    """
    if separability is None:
        return negatives

    sims, baseline = separability
    return {n for n in negatives if sims.get(n, 0.0) < baseline}


def generate_wakeword_variations(
    phrase: str,
    separability_file: Optional[Union[str, Path]] = None,
) -> Tuple[List[str], List[str]]:
    """Split wake word training text into positives and negatives.

    Positives are only renditions of the target phrase itself. Phonetic
    near-misses are negatives, because rejecting near-misses is precisely what
    a wake word detector is for.

    An earlier version split one-phoneme perturbations across both classes
    ("hey rimbo" positive, "hey nimbo" negative). Those are equally distant
    from the target, so the boundary existed only in the labels and not in the
    audio, and accuracy collapsed once the samples came from varied voices.

    Not every near-miss is usable either. Because the whole dataset is
    synthetic, a negative is only worth training on if the text-to-speech
    engine actually renders it differently from the wake word. Two filters
    apply: doubled-character spellings are always dropped, and if measured
    separability scores are available, anything at or above the wake word's own
    self-similarity is dropped too. For "hey limbo" that removes 18 of 87
    generated negatives -- including "hey linbo", "hey nimbo" and "hay limbo",
    which are spelled differently but synthesized identically.

    Args:
        phrase: Wake word phrase (e.g., "hey limbo")
        separability_file: JSON of measured similarity scores. Defaults to
            neg_sims_temporal.json when present; filtering is skipped if not.

    Returns:
        Tuple of (positive_phrases, negative_phrases)

    Raises:
        ValueError: If phrase is empty or has no words
    """
    if not phrase or not isinstance(phrase, str):
        raise ValueError("phrase must be a non-empty string")

    phrase = phrase.strip().lower()
    words = phrase.split()

    if not words:
        raise ValueError("phrase must contain at least one word")

    positive: Set[str] = {phrase}
    negative: Set[str] = set()

    # Spellings a text-to-speech engine pronounces the same way. They add no
    # acoustic variety on their own, but they keep the positive set from being
    # a single string, and cost nothing.
    homophones = {
        "o": ["ow", "oe"],
        "i": ["y"],
        "ee": ["ea"],
        "c": ["k"],
        "ph": ["f"],
    }

    for i, word in enumerate(words):
        for src, replacements in homophones.items():
            if word.endswith(src):
                for repl in replacements:
                    new_words = words.copy()
                    new_words[i] = word[: -len(src)] + repl
                    positive.add(" ".join(new_words))

    # --- Negatives: everything the detector must not fire on ---

    # Single-character substitutions: the hardest confusables.
    substitutions = {
        "a": "eou", "e": "aiou", "i": "aeou", "o": "aeiu", "u": "aeio",
        "b": "pmdv", "p": "bft", "d": "tbg", "t": "dpk", "g": "kdj",
        "k": "gtc", "m": "nbl", "n": "mlr", "l": "rnd", "r": "lwn",
        "s": "zf", "z": "sv", "f": "vps", "v": "fbw", "w": "vr",
        "h": "f", "j": "gy", "y": "ji", "c": "sk",
    }

    for i, word in enumerate(words):
        for pos, char in enumerate(word):
            for repl in substitutions.get(char, ""):
                new_words = words.copy()
                new_words[i] = word[:pos] + repl + word[pos + 1:]
                negative.add(" ".join(new_words))

    # Dropped characters. Doubled characters are deliberately not generated;
    # see _doubled_character_variants.
    for i, word in enumerate(words):
        if len(word) > 2:
            for pos in range(len(word)):
                new_words = words.copy()
                new_words[i] = word[:pos] + word[pos + 1:]
                negative.add(" ".join(new_words))

    # Trailing and leading vowels.
    for i, word in enumerate(words):
        for char in "aeiou":
            for candidate in (word + char, char + word):
                new_words = words.copy()
                new_words[i] = candidate
                negative.add(" ".join(new_words))

    # Partial phrases: each word alone, and the phrase minus one word.
    for word in words:
        negative.add(word)

    if len(words) > 1:
        negative.add(" ".join(words[:-1]))
        negative.add(" ".join(words[1:]))
        negative.add(" ".join(reversed(words)))

    # A different leading word, plus unrelated wake words and filler. These are
    # the most separable negatives measured, and the most realistic: a device
    # must not wake on a neighbouring assistant's name.
    alternates = ["hi", "hello", "hay", "yo", "okay", "oh", "huh", "uh", "the", "a"]
    for replacement in alternates:
        if replacement != words[0]:
            new_words = words.copy()
            new_words[0] = replacement
            negative.add(" ".join(new_words))

    negative.update([
        "okay google", "hey google", "alexa", "hey siri", "computer",
        "jarvis", "cortana", "hey assistant", "yes", "no", "stop", "play",
        "what time is it", "turn on the lights", "thank you", "hello there",
    ])

    negative -= positive
    negative -= _doubled_character_variants(words)
    negative = filter_negatives_by_separability(
        negative, load_separability(separability_file)
    )

    return sorted(positive), sorted(negative)


def main() -> None:
    """Print the phrase split for a sample wake word."""
    phrase = "hey limbo"
    positive, negative = generate_wakeword_variations(phrase)

    print(f"Wake word: {phrase}\n")
    print(f"POSITIVE ({len(positive)}):")
    for p in positive:
        print(f"  + {p}")

    print(f"\nNEGATIVE ({len(negative)}):")
    for n in negative[:30]:
        print(f"  - {n}")
    if len(negative) > 30:
        print(f"  ... and {len(negative) - 30} more")


if __name__ == "__main__":
    main()
