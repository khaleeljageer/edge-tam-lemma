"""Character-level tokenizer with codepoint and grapheme modes.

codepoint: one token per Unicode codepoint after NFC (default, ~80 symbols).
grapheme:  one token per Tamil grapheme cluster: a base letter plus any
           following combining marks (vowel sign, virama), e.g. "கொ" or "க்".
"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

Mode = Literal["codepoint", "grapheme"]

PAD, BOS, EOS, UNK = "<pad>", "<s>", "</s>", "<unk>"
SPECIALS = (PAD, BOS, EOS, UNK)
PAD_ID, BOS_ID, EOS_ID, UNK_ID = range(4)


def graphemes(word: str) -> list[str]:
    """Split a word into grapheme clusters (base + combining marks)."""
    units: list[str] = []
    for ch in word:
        if units and unicodedata.category(ch) in ("Mn", "Mc"):
            units[-1] += ch
        else:
            units.append(ch)
    return units


def split_units(word: str, mode: Mode) -> list[str]:
    if mode == "codepoint":
        return list(word)
    if mode == "grapheme":
        return graphemes(word)
    raise ValueError(f"unknown tokenizer mode: {mode}")


class CharTokenizer:
    def __init__(self, symbols: Iterable[str], mode: Mode = "codepoint") -> None:
        self.mode: Mode = mode
        self.itos: list[str] = list(SPECIALS) + sorted(set(symbols) - set(SPECIALS))
        self.stoi: dict[str, int] = {s: i for i, s in enumerate(self.itos)}

    @classmethod
    def build(
        cls, words: Iterable[str], mode: Mode = "codepoint", min_count: int = 1
    ) -> CharTokenizer:
        counts = Counter(u for w in words for u in split_units(w, mode))
        return cls((s for s, c in counts.items() if c >= min_count), mode)

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, word: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [self.stoi.get(u, UNK_ID) for u in split_units(word, self.mode)]
        if add_bos:
            ids.insert(0, BOS_ID)
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        """Decode ids to a string, stopping at EOS and skipping PAD/BOS."""
        out = []
        for i in ids:
            if i == EOS_ID:
                break
            if i in (PAD_ID, BOS_ID):
                continue
            out.append(self.itos[i] if i != UNK_ID else "")
        return "".join(out)

    def save(self, path: str | Path) -> None:
        data = {"mode": self.mode, "symbols": self.itos[len(SPECIALS) :]}
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> CharTokenizer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["symbols"], data["mode"])
