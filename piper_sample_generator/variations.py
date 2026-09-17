#!/usr/bin/env python3
"""Generate positive and negative wake word variations."""

from typing import List, Set, Tuple


def generate_wakeword_variations(phrase: str) -> Tuple[List[str], List[str]]:
    """Generate positive and negative variations of a wake word phrase.

    Positive: Minor phonetic differences (likely intended phrase)
    Negative: More significant phonetic/structural changes

    Args:
        phrase: Wake word phrase (e.g., "hey limbo")

    Returns:
        Tuple of (positive_variations, negative_variations)

    Raises:
        ValueError: If phrase is empty or None
    """
    if not phrase or not isinstance(phrase, str):
        raise ValueError("phrase must be a non-empty string")

    phrase = phrase.strip().lower()
    words = phrase.split()

    if not words:
        raise ValueError("phrase must contain at least one word")

    positive: Set[str] = set()
    negative: Set[str] = set()

    # POSITIVE VARIATIONS - Minor phonetic differences
    positive.add(phrase)
    positive.add(phrase.upper())
    positive.add(phrase.title())
    positive.add("".join(words))  # Remove spaces
    positive.add("  ".join(words))  # Extra spaces

    # Vowel substitutions (minor)
    vowel_variants = {
        "a": ["aa", "e"],
        "e": ["ee", "i"],
        "i": ["ee", "e"],
        "o": ["oo", "u"],
        "u": ["oo", "o"],
    }

    for i, word in enumerate(words):
        for pos, char in enumerate(word):
            if char in vowel_variants:
                for replacement in vowel_variants[char]:
                    new_word = word[:pos] + replacement + word[pos + 1:]
                    new_words = words.copy()
                    new_words[i] = new_word
                    positive.add(" ".join(new_words))

    # Consonant substitutions (minor - easily confused)
    consonant_variants = {
        "h": [""],
        "l": ["r"],
        "r": ["l"],
        "m": ["n"],
        "n": ["m"],
        "b": ["p"],
        "p": ["b"],
        "d": ["t"],
        "t": ["d"],
    }

    for i, word in enumerate(words):
        for pos, char in enumerate(word):
            if char in consonant_variants:
                for replacement in consonant_variants[char]:
                    new_word = word[:pos] + replacement + word[pos + 1:]
                    new_words = words.copy()
                    new_words[i] = new_word
                    positive.add(" ".join(new_words))

    # Final vowel variations
    for i, word in enumerate(words):
        if word.endswith("o"):
            new_words = words.copy()
            new_words[i] = word + "o"
            positive.add(" ".join(new_words))

    # NEGATIVE VARIATIONS - Significant changes

    # Partial phrases (remove words)
    for i in range(len(words)):
        negative.add(words[i])

    if len(words) > 1:
        negative.add(" ".join(words[:-1]))
        negative.add(" ".join(words[1:]))

    # Change first word (common wake words)
    first_word = words[0]
    first_word_variants = [
        "hi", "hello", "hay", "yo", "okay", "oh", "hey", "huh", "uh"
    ]

    for replacement in first_word_variants:
        if replacement != first_word:
            new_words = words.copy()
            new_words[0] = replacement
            negative.add(" ".join(new_words))

    # Phonetic-near substitutions (more distant)
    phonetic_subs = {
        "l": ["n", "d"],
        "r": ["n", "d"],
        "m": ["b"],
        "b": ["m", "p"],
        "p": ["b"],
        "i": ["a"],
        "e": ["a"],
        "o": ["a", "u"],
        "a": ["u"],
    }

    for i, word in enumerate(words):
        for pos, char in enumerate(word):
            if char in phonetic_subs:
                for replacement in phonetic_subs[char]:
                    new_word = word[:pos] + replacement + word[pos + 1:]
                    new_words = words.copy()
                    new_words[i] = new_word
                    negative.add(" ".join(new_words))

    # Remove/insert characters
    for i, word in enumerate(words):
        if len(word) > 2:
            for pos in range(len(word)):
                new_word = word[:pos] + word[pos + 1:]
                new_words = words.copy()
                new_words[i] = new_word
                negative.add(" ".join(new_words))

    # Add vowels
    for i, word in enumerate(words):
        for char in "aeiou":
            new_word = word + char
            new_words = words.copy()
            new_words[i] = new_word
            negative.add(" ".join(new_words))

    # Word order reversal
    if len(words) > 1:
        negative.add(" ".join(reversed(words)))

    # Remove positives from negatives
    negative -= positive

    return sorted(positive), sorted(negative)


def main() -> None:
    """Test variation generation."""
    phrase = "hey limbo"
    positive, negative = generate_wakeword_variations(phrase)

    print(f"Wake word: {phrase}\n")
    print(f"POSITIVE VARIATIONS ({len(positive)}):")
    for p in positive[:20]:
        print(f"  + {p}")
    if len(positive) > 20:
        print(f"  ... and {len(positive) - 20} more")

    print(f"\nNEGATIVE VARIATIONS ({len(negative)}):")
    for n in negative[:20]:
        print(f"  - {n}")
    if len(negative) > 20:
        print(f"  ... and {len(negative) - 20} more")


if __name__ == "__main__":
    main()
