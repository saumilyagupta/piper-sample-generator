#!/usr/bin/env python3
"""Generate wake word training phrases: the target phrase and its confusables."""

from typing import List, Set, Tuple


def generate_wakeword_variations(phrase: str) -> Tuple[List[str], List[str]]:
    """Split wake word training text into positives and negatives.

    Positives are only renditions of the target phrase itself. Every phonetic
    near-miss is a negative, because rejecting near-misses is precisely what a
    wake word detector is for.

    An earlier version split one-phoneme perturbations across both classes
    ("hey rimbo" positive, "hey nimbo" negative). Those are equally distant
    from the target, so the boundary existed only in the labels and not in the
    audio, and accuracy collapsed once the samples came from varied voices.

    Args:
        phrase: Wake word phrase (e.g., "hey limbo")

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

    # Dropped and doubled characters.
    for i, word in enumerate(words):
        if len(word) > 2:
            for pos in range(len(word)):
                new_words = words.copy()
                new_words[i] = word[:pos] + word[pos + 1:]
                negative.add(" ".join(new_words))

        for pos in range(len(word)):
            new_words = words.copy()
            new_words[i] = word[:pos] + word[pos] + word[pos:]
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

    # A different leading word, plus unrelated wake words and filler.
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
