"""Error analysis on test: categorize every model error and pick examples for the paper.

    uv run python scripts/analysis/error_analysis.py

Reads runs/<run>/test_errors.tsv (written by tel.train, in test.tsv order) for
RUNS, and data/processed/{train,test}.tsv. Writes bench/results/m5_errors/:
  errors.csv      one row per test pair that any run gets wrong: gold, POS, corpus
                  frequency, whether the lemma occurs in train, whether the gold
                  edit script is in the edit classifier's vocabulary, and per run
                  the prediction and its category (tel.error_analysis)
  categories.csv  per run and category: errors, share of errors, share of test,
                  and the split by POS
  examples.csv    EXAMPLES errors of the deployed model (MAIN), drawn with a fixed
                  seed across categories, with the other runs' predictions
"""

from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path

from tel.data import POS_TAGS, Pair, read_tsv
from tel.edit_tree import ScriptVocab, extract
from tel.error_analysis import CATEGORIES, categorize
from tel.normalize import normalize

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
RUNS_DIR = ROOT / "runs"
OUT = ROOT / "bench" / "results" / "m5_errors"
RUNS = ["teacher_lstm", "lstm_500k_s13", "lstm_kd_500k_s13", "edit_500k_s13",
        "edit_kd_500k_s13"]
MAIN = "edit_kd_500k_s13"
EXAMPLES = 15
SEED = 0


def predictions(run: str, test: list[Pair]) -> list[str]:
    """Rebuild the run's full test predictions from its error file (same order)."""
    with open(RUNS_DIR / run / "test_errors.tsv", encoding="utf-8", newline="") as f:
        errors = [(normalize(r["word"]), normalize(r["gold"]), normalize(r["pred"]))
                  for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)]
    preds, i = [], 0
    for p in test:
        if i < len(errors) and errors[i][:2] == (p.word, p.lemma):
            preds.append(errors[i][2])
            i += 1
        else:
            preds.append(p.lemma)
    if i != len(errors):
        raise SystemExit(f"{run}: test_errors.tsv does not follow test.tsv order "
                         f"({i} of {len(errors)} errors matched)")
    return preds


def sample_examples(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Round-robin over categories (most frequent first), random within each."""
    rng = random.Random(seed)
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        by_cat.setdefault(r[f"cat_{MAIN}"], []).append(r)
    for v in by_cat.values():
        rng.shuffle(v)
    order = sorted(by_cat, key=lambda c: (-len(by_cat[c]), CATEGORIES.index(c)))
    picked: list[dict] = []
    while len(picked) < n and any(by_cat.values()):
        for c in order:
            if by_cat[c] and len(picked) < n:
                picked.append(by_cat[c].pop())
    return sorted(picked, key=lambda r: (CATEGORIES.index(r[f"cat_{MAIN}"]), r["word"]))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    test = read_tsv(DATA / "test.tsv")
    train_lemmas = {p.lemma for p in read_tsv(DATA / "train.tsv")}
    vocab = ScriptVocab.load(str(RUNS_DIR / MAIN / "scripts.txt"))
    preds = {run: predictions(run, test) for run in RUNS}

    rows = []
    for i, p in enumerate(test):
        wrong = [run for run in RUNS if preds[run][i] != p.lemma]
        if not wrong:
            continue
        row = {"word": p.word, "gold": p.lemma, "pos": p.pos, "freq": p.freq,
               "lemma_in_train": int(p.lemma in train_lemmas),
               "script_in_vocab": int(extract(p.word, p.lemma) in vocab.stoi),
               "n_wrong": len(wrong)}
        for run in RUNS:
            pred = preds[run][i]
            row[f"pred_{run}"] = pred
            row[f"cat_{run}"] = "" if pred == p.lemma else categorize(p.word, p.lemma, pred)
        rows.append(row)
    write_csv(OUT / "errors.csv", rows)

    summary = []
    for run in RUNS:
        cats = [(r[f"cat_{run}"], r["pos"]) for r in rows if r[f"cat_{run}"]]
        n_err = len(cats)
        by = Counter(cats)
        for c in CATEGORIES:
            n = sum(by[(c, pos)] for pos in POS_TAGS)
            summary.append({"run": run, "category": c, "errors": n,
                            "share_of_errors": f"{n / n_err:.4f}",
                            "share_of_test": f"{n / len(test):.4f}",
                            **{f"n_{pos.lower()}": by[(c, pos)] for pos in POS_TAGS}})
    write_csv(OUT / "categories.csv", summary)

    main_errors = [r for r in rows if r[f"cat_{MAIN}"]]
    examples = [{"category": r[f"cat_{MAIN}"], **{k: v for k, v in r.items()
                                                  if not k.startswith("cat_")}}
                for r in sample_examples(main_errors, EXAMPLES, SEED)]
    write_csv(OUT / "examples.csv", examples)

    all_wrong = sum(r["n_wrong"] == len(RUNS) for r in rows)
    print(f"{len(rows)} test pairs with an error in some run; {all_wrong} wrong in all "
          f"{len(RUNS)} runs")
    for run in RUNS:
        line = ", ".join(f"{s['category']} {s['errors']}" for s in summary if s["run"] == run)
        print(f"  {run}: {line}")
    print(f"wrote {OUT}/{{errors,categories,examples}}.csv")


if __name__ == "__main__":
    main()
