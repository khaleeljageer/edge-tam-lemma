"""Rule-based categories for lemmatization errors (for the paper's error analysis).

Strings are compared in Tamil graphemes (a base letter plus its combining vowel
sign / virama), so "the last letter differs" means து vs தை, not a lone sign.

Categories, checked in this order (word = input form, gold, pred):
  unchanged       pred == word: the model left the inflected form as it was
  gold_is_form    gold == word: the treebank keeps the form (adverbs, postpositions,
                  participles used as adjectives), the model reduced it anyway
  suffix_left     gold is a proper prefix of pred: part of the suffix was not removed
  over_stripped   pred is a proper prefix of gold: too much was removed
  stem_final      same stem, the last 1 gold grapheme differs (pred differs in at most
                  2): stem-final vowel or consonant not restored (sandhi, -உ / -அம்)
  wrong_stem      anything else: irregular or suppletive stems, wrong segmentation
"""

from __future__ import annotations

import unicodedata

CATEGORIES = ("unchanged", "gold_is_form", "suffix_left", "over_stripped", "stem_final",
              "wrong_stem")


def graphemes(s: str) -> list[str]:
    """Split into base characters with their combining marks attached."""
    out: list[str] = []
    for ch in s:
        if out and unicodedata.category(ch).startswith("M"):
            out[-1] += ch
        else:
            out.append(ch)
    return out


def common_prefix(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def categorize(word: str, gold: str, pred: str) -> str:
    if pred == gold:
        raise ValueError("not an error")
    if pred == word:
        return "unchanged"
    if gold == word:
        return "gold_is_form"
    g, p = graphemes(gold), graphemes(pred)
    k = common_prefix(g, p)
    if k == len(g):
        return "suffix_left"
    if k == len(p):
        return "over_stripped"
    if len(g) - k <= 1 and len(p) - k <= 2:
        return "stem_final"
    return "wrong_stem"
