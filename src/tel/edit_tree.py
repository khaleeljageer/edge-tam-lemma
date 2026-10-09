"""Edit scripts: word -> lemma as prefix/suffix rewrites around a kept stem.

The stem is the longest common substring of word and lemma. The script records
which strings are removed from / added to each side of it:

    சென்றார்கள் -> செல்   stem "செ"   suffix  -ன்றார்கள் +ல்

Removed strings are stored literally (not as lengths), so a script only applies
to words that actually start/end with them. This lets the edit classifier mask
out classes that cannot apply to the input word.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

_SEP = "|"


@dataclass(frozen=True)
class EditScript:
    pre_del: str = ""
    pre_add: str = ""
    suf_del: str = ""
    suf_add: str = ""

    def applies(self, word: str) -> bool:
        return (
            len(self.pre_del) + len(self.suf_del) <= len(word)
            and word.startswith(self.pre_del)
            and word.endswith(self.suf_del)
        )

    def apply(self, word: str) -> str | None:
        """Return the lemma, or None if the script does not apply to `word`."""
        if not self.applies(word):
            return None
        stem = word[len(self.pre_del) : len(word) - len(self.suf_del)]
        return self.pre_add + stem + self.suf_add

    def __str__(self) -> str:
        return _SEP.join((self.pre_del, self.pre_add, self.suf_del, self.suf_add))

    @classmethod
    def parse(cls, s: str) -> EditScript:
        parts = s.split(_SEP)
        if len(parts) != 4:
            raise ValueError(f"bad edit script: {s!r}")
        return cls(*parts)


IDENTITY = EditScript()


def _longest_common_substring(a: str, b: str) -> tuple[int, int, int]:
    """Return (start_a, start_b, length); ties go to the earliest start in `a`."""
    best = (0, 0, 0)
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best[2]:
                    best = (i - cur[j], j - cur[j], cur[j])
        prev = cur
    return best


def extract(word: str, lemma: str) -> EditScript:
    """Build the edit script that rewrites `word` into `lemma`."""
    i, j, k = _longest_common_substring(word, lemma)
    if k == 0:
        # No shared material: whole-word replacement (suppletive forms).
        return EditScript(suf_del=word, suf_add=lemma)
    return EditScript(word[:i], lemma[:j], word[i + k :], lemma[j + k :])


class ScriptVocab:
    """Maps edit scripts to class ids. Class 0 is always the identity script."""

    def __init__(self, scripts: Iterable[EditScript]) -> None:
        self.itos: list[EditScript] = [IDENTITY]
        self.itos += [s for s in dict.fromkeys(scripts) if s != IDENTITY]
        self.stoi: dict[EditScript, int] = {s: i for i, s in enumerate(self.itos)}

    @classmethod
    def build(
        cls, pairs: Iterable[tuple[str, str]], top_k: int | None = None, min_count: int = 1
    ) -> ScriptVocab:
        counts = Counter(extract(w, lem) for w, lem in pairs)
        ranked = [s for s, c in counts.most_common(top_k) if c >= min_count]
        return cls(ranked)

    def __len__(self) -> int:
        return len(self.itos)

    def id_of(self, script: EditScript) -> int:
        """Class id of a script; scripts outside the vocab map to identity (0)."""
        return self.stoi.get(script, 0)

    def applicable_mask(self, word: str) -> list[bool]:
        return [s.applies(word) for s in self.itos]

    def coverage(self, pairs: Iterable[tuple[str, str]]) -> float:
        """Fraction of pairs whose gold script is in the vocab (or is identity)."""
        pairs = list(pairs)
        if not pairs:
            return 0.0
        hit = sum(extract(w, lem) in self.stoi for w, lem in pairs)
        return hit / len(pairs)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(f"{s}\n" for s in self.itos)

    @classmethod
    def load(cls, path: str) -> ScriptVocab:
        with open(path, encoding="utf-8") as f:
            return cls(EditScript.parse(line.rstrip("\n")) for line in f if line.strip("\n"))
