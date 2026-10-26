"""
Word N-gram Builder

Generates a single merged n-gram fingerprint for any text string.

Process:
  1. Normalize: uppercase, remove special chars, split into word tokens
  2. Word 1-gram: each word token as-is
  3. Word 2-gram: adjacent token pairs, concatenated (no separator)
  4. Word 3-gram: adjacent token triples, concatenated
  5. Merge all, deduplicate, sort
"""

import re
from typing import List


def _to_words(s: str) -> List[str]:
    s = str(s).strip().upper()
    s = re.sub(r"[^A-Z0-9]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return [w for w in s.split() if w]


def _word_ngrams(words: List[str], n: int) -> List[str]:
    if len(words) < n:
        return []
    return ["".join(words[i: i + n]) for i in range(len(words) - n + 1)]


def build_ngram_fingerprint(value: str) -> List[str]:
    """
    Build combined word n-gram fingerprint (1+2+3-gram tokens).
    Returns a sorted, deduplicated list.
    """
    if not value or (isinstance(value, float)):
        return []
    words = _to_words(str(value))
    if not words:
        return []
    combined = set(_word_ngrams(words, 1)) | set(_word_ngrams(words, 2)) | set(_word_ngrams(words, 3))
    return sorted(combined)


def jaccard_similarity(a_grams: List[str], b_grams: List[str]) -> float:
    a, b = set(a_grams), set(b_grams)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)
