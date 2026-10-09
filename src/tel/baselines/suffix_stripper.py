"""Rule baseline: iteratively strip common Tamil suffixes, then undo stem sandhi.

Deliberately simple. It exists to show how far hand-written rules get, not to
be a morphological analyzer.
"""

from __future__ import annotations

import unicodedata

from tel.tokenizer import graphemes

VIRAMA = "்"

# (suffix, replacement). Suffixes starting with a vowel sign attach to a
# consonant-final stem; after removing them the stem gets a virama back.
_RULES: list[tuple[str, str]] = [
    # plural
    ("ங்கள்", "ம்"),  # மரங்கள் -> மரம்
    ("க்கள்", ""),  # பூக்கள் -> பூ
    ("கள்", ""),
    # case markers and postpositions, vowel-final stems (glide + suffix)
    ("யிலிருந்து", ""), ("விலிருந்து", ""),
    ("யுடைய", ""), ("வுடைய", ""),
    ("யுடன்", ""), ("வுடன்", ""),
    ("யோடு", ""), ("வோடு", ""),
    ("யில்", ""), ("வில்", ""),
    ("யால்", ""), ("வால்", ""),
    ("யின்", ""), ("வின்", ""),
    ("யும்", ""), ("வும்", ""),
    ("யை", ""), ("வை", ""),
    ("க்கு", ""),
    # case markers and clitics, consonant-final stems
    ("ிலிருந்து", ""),
    ("ுடைய", ""),
    ("ுக்கு", ""),
    ("ுடன்", ""),
    ("ோடு", ""),
    ("ில்", ""),
    ("ால்", ""),
    ("ின்", ""),
    ("ும்", ""),
    ("ை", ""),
    # verb person/number endings with common tense markers
    ("க்கிறார்கள்", ""), ("கிறார்கள்", ""),
    ("க்கிறான்", ""), ("கிறான்", ""),
    ("க்கிறாள்", ""), ("கிறாள்", ""),
    ("க்கிறேன்", ""), ("கிறேன்", ""),
    ("ார்கள்", ""),
    ("ீர்கள்", ""),
    ("ான்", ""), ("ாள்", ""), ("ார்", ""), ("ேன்", ""), ("ோம்", ""), ("ாய்", ""),
]
_RULES.sort(key=lambda r: len(r[0]), reverse=True)

# Stem-final sandhi undone after stripping (oblique stems).
_STEM_FIXES: list[tuple[str, str]] = [
    ("த்த்", "ம்"),  # மரத்தில் -> மரத்த் -> மரம்
    ("ட்ட்", "டு"),  # வீட்டை -> வீட்ட் -> வீடு
    ("ற்ற்", "று"),  # ஆற்றில் -> ஆற்ற் -> ஆறு
]


def _is_vowel_sign(ch: str) -> bool:
    return unicodedata.category(ch) in ("Mn", "Mc")


def _has_syllable(stem: str) -> bool:
    # A stem needs at least one vowel-bearing grapheme: "பூ" is fine, "ந்" is not.
    return any(not g.endswith(VIRAMA) for g in graphemes(stem))


def strip_once(word: str) -> str | None:
    for suffix, repl in _RULES:
        if not word.endswith(suffix):
            continue
        stem = word[: -len(suffix)] + repl
        if _is_vowel_sign(suffix[0]):
            stem += VIRAMA
        if _has_syllable(stem):
            return stem
    return None


def lemmatize_word(word: str, max_steps: int = 3) -> str:
    for _ in range(max_steps):
        stripped = strip_once(word)
        if stripped is None:
            break
        word = stripped
    for end, repl in _STEM_FIXES:
        if word.endswith(end):
            return word[: -len(end)] + repl
    return word


class SuffixStripper:
    def lemmatize(self, words: list[str]) -> list[str]:
        return [lemmatize_word(w) for w in words]
