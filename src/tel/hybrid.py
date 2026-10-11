"""Hybrid lemmatizer: a lookup table for frequent word forms, the model for the rest.

The table maps the top-N most frequent word forms (from a frequency list) to a
lemma. Labels come, in priority order, from gold training pairs, reviewed
annotations (data/annotation/) and an offline "silver" labeller
(scripts/hybrid/build_tables.py: majority vote of ThamizhiMorph, the teacher
and the distilled students). The expensive labellers
run once, offline; on the device frequent words cost one dict lookup and only
misses reach the neural model.

Pure Python (no torch, no numpy) so it runs inside the device benchmark.
Table files are TSV: word, lemma, source (gold / reviewed / silver), most
frequent first.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

HEADER = ["word", "lemma", "source"]


def read_table(path: str | Path) -> dict[str, str]:
    table: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        iw, il = header.index("word"), header.index("lemma")
        for line in f:
            row = line.rstrip("\n").split("\t")
            if len(row) > max(iw, il) and row[iw]:
                table.setdefault(row[iw], row[il])
    return table


def write_table(path: str | Path, rows: Iterable[tuple[str, str, str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(HEADER) + "\n")
        f.writelines(f"{w}\t{lemma}\t{src}\n" for w, lemma, src in rows)


def gold_lemmas(pairs: Iterable) -> dict[str, str]:
    """word -> its most frequent lemma in (word, lemma, ..., freq) pairs; ties -> first seen."""
    counts: dict[str, dict[str, int]] = defaultdict(dict)
    for p in pairs:
        c = counts[p.word]
        c[p.lemma] = c.get(p.lemma, 0) + max(1, int(p.freq))
    return {w: max(c, key=c.__getitem__) for w, c in counts.items()}


def build_table(ranked_words: Sequence[str], n: int,
                labels: Sequence[tuple[str, dict[str, str]]]) -> list[tuple[str, str, str]]:
    """Rows for the top-n words. `labels` is [(source name, word -> lemma), ...] in
    priority order (e.g. gold, reviewed, silver); words no source labels are skipped."""
    rows = []
    for w in ranked_words[:n]:
        for name, lemmas in labels:
            if w in lemmas:
                rows.append((w, lemmas[w], name))
                break
    return rows


def weighted_stream(items: Sequence, weights: Sequence[int], seed: int = 0) -> list:
    """Each item repeated by its weight (corpus frequency), in a fixed shuffled order:
    a stand-in for running text built from a type list."""
    out = [x for x, k in zip(items, weights, strict=True) for _ in range(max(1, int(k)))]
    random.Random(seed).shuffle(out)
    return out


class HybridLemmatizer:
    """Table first; misses go to `fallback.lemmatize` in one call per batch."""

    def __init__(self, table: dict[str, str], fallback) -> None:
        self.table, self.fallback = table, fallback
        self.lookups = self.hits = 0

    def lemmatize(self, words: list[str]) -> list[str]:
        out = [self.table.get(w) for w in words]
        miss = [i for i, lemma in enumerate(out) if lemma is None]
        self.lookups += len(words)
        self.hits += len(words) - len(miss)
        if miss:
            for i, lemma in zip(miss, self.fallback.lemmatize([words[i] for i in miss]),
                                strict=True):
                out[i] = lemma
        return out

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0
