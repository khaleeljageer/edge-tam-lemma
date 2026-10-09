"""Unicode normalization and cleaning for Tamil text.

Every string entering the pipeline goes through `normalize()` first, so that
composed and decomposed spellings of the same word are identical.
"""

from __future__ import annotations

import re
import unicodedata

# Tamil letters, vowel signs, virama, OM and the AU length mark.
# Tamil digits (U+0BE6-0BEF) and symbols (U+0BF0-0BFA) are excluded.
_TAMIL_LETTER = "ஂ-்ௐௗ"
_TAMIL_WORD_RE = re.compile(f"^[{_TAMIL_LETTER}]+$")
_NON_TAMIL_RE = re.compile(f"[^{_TAMIL_LETTER}]")

# Invisible characters that appear in web text and change nothing visually.
_INVISIBLE = dict.fromkeys(map(ord, "​‌‍­﻿"))


def normalize(text: str) -> str:
    """NFC-normalize and drop invisible formatting characters."""
    return unicodedata.normalize("NFC", text.translate(_INVISIBLE))


def is_tamil_word(token: str) -> bool:
    """True if the (normalized) token consists only of Tamil letters and signs."""
    return bool(_TAMIL_WORD_RE.match(token))


def clean_word(token: str) -> str:
    """Normalize a token and remove every non-Tamil character (punctuation, Latin, digits).

    May return an empty string.
    """
    return _NON_TAMIL_RE.sub("", normalize(token))


def tamil_words(text: str) -> list[str]:
    """Split text on whitespace and return the cleaned, non-empty Tamil words."""
    words = (clean_word(t) for t in text.split())
    return [w for w in words if w and _starts_with_base(w)]


def _starts_with_base(word: str) -> bool:
    # A word cannot start with a combining vowel sign or virama.
    return unicodedata.category(word[0]) not in ("Mn", "Mc")
