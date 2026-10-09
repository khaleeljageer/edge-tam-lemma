"""Score the in-repo baselines (identity, suffix stripper) and write
bench/results/accuracy_baselines.csv. External baselines append to the same file
from their own scripts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tel.baselines.identity import IdentityLemmatizer
from tel.baselines.suffix_stripper import SuffixStripper
from tel.data import read_tsv
from tel.evaluate import evaluate

SPLITS = ("dev", "test", "test_unseen")
COLUMNS = ["model", "split", "n", "acc", "acc_noun", "acc_verb", "acc_other", "acc_seen",
           "acc_unseen"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="bench/results/accuracy_baselines.csv")
    args = ap.parse_args()

    data = Path(args.data)
    train_lemmas = {p.lemma for p in read_tsv(data / "train.tsv")}
    models = {"identity": IdentityLemmatizer(), "suffix_stripper": SuffixStripper()}
    rows = []
    for name, model in models.items():
        for split in SPLITS:
            m = evaluate(model, read_tsv(data / f"{split}.tsv"), train_lemmas).metrics()
            rows.append({"model": name, "split": split, **m})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in r.items()})
    print(Path(args.out).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
