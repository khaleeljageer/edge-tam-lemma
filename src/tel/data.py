"""Dataset records, TSV I/O and lemma-based splitting."""

from __future__ import annotations

import csv
import random
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from tel.normalize import normalize

POS_TAGS = ("NOUN", "VERB", "OTHER")

_UPOS_MAP = {"NOUN": "NOUN", "PROPN": "NOUN", "VERB": "VERB", "AUX": "VERB"}


def map_upos(upos: str) -> str:
    """Collapse UD UPOS tags to the NOUN / VERB / OTHER scheme."""
    return _UPOS_MAP.get(upos, "OTHER")


@dataclass
class Pair:
    word: str
    lemma: str
    pos: str = "OTHER"
    source: str = ""
    freq: int = 1
    uncertain: int = 0


COLUMNS = tuple(f.name for f in fields(Pair))


def read_tsv(path: str | Path) -> list[Pair]:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        return [
            Pair(
                word=normalize(r["word"]),
                lemma=normalize(r["lemma"]),
                pos=r.get("pos") or "OTHER",
                source=r.get("source") or "",
                freq=int(r.get("freq") or 1),
                uncertain=int(r.get("uncertain") or 0),
            )
            for r in reader
        ]


def write_tsv(path: str | Path, pairs: Iterable[Pair]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=COLUMNS, delimiter="\t", quoting=csv.QUOTE_NONE, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(asdict(p) for p in pairs)


def split_lemmas(
    lemmas: Iterable[str], fractions: Sequence[float] = (0.8, 0.1, 0.1), seed: int = 13
) -> list[list[str]]:
    """Shuffle unique lemmas with a fixed seed and cut them into len(fractions) groups."""
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    unique = sorted(set(lemmas))
    random.Random(seed).shuffle(unique)
    groups, start = [], 0
    for i, frac in enumerate(fractions):
        end = len(unique) if i == len(fractions) - 1 else start + round(frac * len(unique))
        groups.append(sorted(unique[start:end]))
        start = end
    return groups


def split_by_lemma(
    pairs: Sequence[Pair],
    names: Sequence[str] = ("train", "dev", "test"),
    fractions: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 13,
) -> tuple[dict[str, list[Pair]], dict[str, list[str]]]:
    """Split pairs so that each lemma (and all its word forms) lands in exactly one split."""
    groups = split_lemmas((p.lemma for p in pairs), fractions, seed)
    lemma_lists = dict(zip(names, groups, strict=True))
    owner = {lemma: name for name, group in lemma_lists.items() for lemma in group}
    splits: dict[str, list[Pair]] = {name: [] for name in names}
    for p in pairs:
        splits[owner[p.lemma]].append(p)
    return splits, lemma_lists


def holdout_seen_forms(
    train: Sequence[Pair], n_groups: int, frac: float, seed: int = 13
) -> tuple[list[Pair], list[list[Pair]]]:
    """Move word forms of training lemmas out of train into `n_groups` held-out groups.

    Each group gets about `frac` of the train pairs. The unit is a (word, lemma)
    key, so a word form never stays in train under the same lemma it is tested
    with. Every lemma keeps at least one form in train, so the held-out lemmas
    are still "seen".
    """
    by_key: dict[tuple[str, str], list[Pair]] = {}
    for p in train:
        by_key.setdefault((p.word, p.lemma), []).append(p)
    forms_left: dict[str, int] = {}
    for _, lemma in by_key:
        forms_left[lemma] = forms_left.get(lemma, 0) + 1

    keys = sorted(by_key)
    random.Random(seed).shuffle(keys)
    target = round(frac * len(train))
    groups: list[list[Pair]] = [[] for _ in range(n_groups)]
    held: set[tuple[str, str]] = set()
    g = 0
    for key in keys:
        if g == n_groups:
            break
        if forms_left[key[1]] < 2:
            continue
        forms_left[key[1]] -= 1
        held.add(key)
        groups[g].extend(by_key[key])
        if len(groups[g]) >= target:
            g += 1
    kept = [p for p in train if (p.word, p.lemma) not in held]
    return kept, groups


def make_splits(
    pairs: Sequence[Pair], seen_frac: float = 0.05, seed: int = 13
) -> tuple[dict[str, list[Pair]], dict[str, list[str]]]:
    """Build train / dev / test / test_unseen.

    Lemmas are split 80/10/10, so dev and test lemmas never occur in training
    (the unseen part). Then about `seen_frac` of the train pairs each go to dev
    and test as forms of training lemmas (the seen part). `test_unseen` is the
    unseen part of test on its own.
    """
    by_lemma, lemma_lists = split_by_lemma(pairs, seed=seed)
    train, (dev_seen, test_seen) = holdout_seen_forms(by_lemma["train"], 2, seen_frac, seed)
    splits = {
        "train": train,
        "dev": by_lemma["dev"] + dev_seen,
        "test": by_lemma["test"] + test_seen,
        "test_unseen": by_lemma["test"],
    }
    return splits, lemma_lists
