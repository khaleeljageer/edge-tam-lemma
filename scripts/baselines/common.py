"""Shared harness for baseline scripts: load splits, score a word -> lemma function
(counting crashes), and write rows into bench/results/accuracy_baselines.csv.

Each script replaces only its own model's rows, so scripts can run in any order
and in different environments.

Extra split `test_clean`: test pairs whose word does not occur as a written word
in the UD Tamil-TTB *train* file. Stanza's Tamil model (and possibly other
tools) were trained on TTB train, so scores on the full test set can be inflated
for them; `test_clean` gives a leakage-free comparison for every model.
"""

from __future__ import annotations

import csv
import importlib.util
from collections.abc import Callable
from pathlib import Path

from tel.data import Pair, read_tsv
from tel.evaluate import score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
TTB_TRAIN = ROOT / "data" / "raw" / "ud" / "UD_Tamil-TTB" / "ta_ttb-ud-train.conllu"
RESULTS = ROOT / "bench" / "results" / "accuracy_baselines.csv"
SPLITS = ("dev", "test", "test_unseen", "test_clean")
COLUMNS = ["model", "split", "n", "acc", "acc_noun", "acc_verb", "acc_other", "acc_seen",
           "acc_unseen", "crashes", "notes"]


def _ttb_train_forms() -> set[str]:
    spec = importlib.util.spec_from_file_location("build_ud", ROOT / "scripts/data/build_ud.py")
    build_ud = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_ud)
    from tel.normalize import clean_word

    return {clean_word(form) for form, _, _ in build_ud.read_conllu(TTB_TRAIN)}


def load_splits() -> tuple[dict[str, list[Pair]], set[str]]:
    splits = {s: read_tsv(DATA / f"{s}.tsv") for s in SPLITS if (DATA / f"{s}.tsv").exists()}
    if TTB_TRAIN.exists():
        seen_by_ttb = _ttb_train_forms()
        splits["test_clean"] = [p for p in splits["test"] if p.word not in seen_by_ttb]
    train_lemmas = {p.lemma for p in read_tsv(DATA / "train.tsv")}
    return splits, train_lemmas


def run_baseline(name: str, lemmatize: Callable[[list[str]], list[str | None]],
                 notes: str = "") -> list[dict]:
    """Score `lemmatize` on every split. It returns one lemma per word, or None for a
    word it crashed on; a crash counts as an error and is reported in `crashes`."""
    splits, train_lemmas = load_splits()
    rows = []
    for split, pairs in splits.items():
        preds = lemmatize([p.word for p in pairs])
        crashes = sum(p is None for p in preds)
        m = score(pairs, [p if p is not None else "" for p in preds], train_lemmas).metrics()
        rows.append({"model": name, "split": split, **m, "crashes": crashes, "notes": notes})
        print(f"{name:22} {split:12} n={m['n']:4}  acc={m['acc']:.4f}  crashes={crashes}")
    write_rows(name, rows)
    return rows


def write_rows(name: str, rows: list[dict]) -> None:
    """Replace `name`'s rows in accuracy_baselines.csv, keeping all other models."""
    kept = []
    if RESULTS.exists():
        with open(RESULTS, newline="", encoding="utf-8") as f:
            kept = [r for r in csv.DictReader(f) if r["model"] != name]
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for r in kept + rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
