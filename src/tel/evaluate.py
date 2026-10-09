"""Exact-match evaluation, broken down by POS and by seen/unseen lemma."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from tel.data import POS_TAGS, Pair
from tel.lemmatizer import Lemmatizer


@dataclass
class _Count:
    correct: int = 0
    total: int = 0

    def add(self, ok: bool) -> None:
        self.correct += ok
        self.total += 1

    @property
    def acc(self) -> float | None:
        return self.correct / self.total if self.total else None


@dataclass
class EvalResult:
    overall: _Count = field(default_factory=_Count)
    by_pos: dict[str, _Count] = field(default_factory=lambda: {p: _Count() for p in POS_TAGS})
    seen: _Count = field(default_factory=_Count)
    unseen: _Count = field(default_factory=_Count)
    errors: list[tuple[Pair, str]] = field(default_factory=list)

    def metrics(self) -> dict[str, float | int | None]:
        """Flat dict whose keys match the bench/results CSV columns where they overlap."""
        out: dict[str, float | int | None] = {"acc": self.overall.acc, "n": self.overall.total}
        for pos, c in self.by_pos.items():
            out[f"acc_{pos.lower()}"] = c.acc
        out["acc_seen"] = self.seen.acc
        out["acc_unseen"] = self.unseen.acc
        return out


def score(
    gold: Sequence[Pair], predictions: Sequence[str], train_lemmas: Iterable[str] = ()
) -> EvalResult:
    if len(gold) != len(predictions):
        raise ValueError(f"{len(gold)} gold pairs but {len(predictions)} predictions")
    known = set(train_lemmas)
    res = EvalResult()
    for pair, pred in zip(gold, predictions, strict=True):
        ok = pred == pair.lemma
        res.overall.add(ok)
        res.by_pos.setdefault(pair.pos, _Count()).add(ok)
        (res.seen if pair.lemma in known else res.unseen).add(ok)
        if not ok:
            res.errors.append((pair, pred))
    return res


def evaluate(
    model: Lemmatizer, gold: Sequence[Pair], train_lemmas: Iterable[str] = ()
) -> EvalResult:
    predictions = model.lemmatize([p.word for p in gold])
    return score(gold, predictions, train_lemmas)


def write_errors(path: str | Path, result: EvalResult) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["word", "gold", "pred", "pos"])
        for pair, pred in result.errors:
            w.writerow([pair.word, pair.lemma, pred, pair.pos])
