"""Second-annotator check of the reviewed expansion batch (inter-annotator agreement).

    uv run python scripts/data/agreement.py sample     # sheets for the second annotator
    # ... the annotator fills them in (Excel / LibreOffice, UTF-8 CSV) ...
    uv run python scripts/data/agreement.py score      # bench/results/m3_expansion/agreement.csv

`sample` draws --n rows the first annotator accepted (`checked` = y in
data/annotation/wiki_batch01_reviewed.tsv) with a fixed seed, in proportion to
the agree / majority / conflict rows, and writes
data/annotation/agreement/lemma_sheet.csv: id, word, and empty `lemma`, `pos`,
`note` columns. The sheet is blind: it shows neither the first annotator's
lemma nor the systems' suggestions, so the two annotations are independent.
It also writes error_examples_sheet.csv from bench/results/m5_errors/examples.csv
(word, gold lemma, the two deployed models' outputs, the automatic error
category) with columns for the annotator to judge the category and the gold
lemma.

`score` compares the filled lemma sheet with the first annotator's lemmas:
exact agreement (after NFC) overall and per status, plus Cohen's kappa on the
derived binary decision "the lemma is the word itself" vs "the word is
reduced". Rows the second annotator left empty are not counted.
Sheets are UTF-8 with a byte-order mark so spreadsheet programs show Tamil text.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

from tel.normalize import normalize

ROOT = Path(__file__).resolve().parents[2]
REVIEWED = ROOT / "data" / "annotation" / "wiki_batch01_reviewed.tsv"
OUT_DIR = ROOT / "data" / "annotation" / "agreement"
EXAMPLES = ROOT / "bench" / "results" / "m5_errors" / "examples.csv"
RESULT = ROOT / "bench" / "results" / "m3_expansion" / "agreement.csv"
STATUSES = ("agree", "majority", "conflict")
CATEGORY_NAMES = {
    "unchanged": "Form left unchanged",
    "gold_is_form": "Gold keeps the form",
    "suffix_left": "Suffix partly left",
    "over_stripped": "Too much removed",
    "stem_final": "Stem-final letter wrong",
    "wrong_stem": "Other / wrong stem",
}


def read_reviewed(path: Path = REVIEWED) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return [r for r in csv.DictReader(f, delimiter="\t") if r["checked"].strip() == "y"]


def stratified_sample(rows: list[dict[str, str]], n: int, seed: int) -> list[dict[str, str]]:
    """n rows, each status in proportion to its share (largest remainder), fixed seed."""
    by_status: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_status[r["status"]].append(r)
    shares = {s: n * len(by_status[s]) / len(rows) for s in STATUSES}
    take = {s: int(shares[s]) for s in STATUSES}
    by_remainder = sorted(STATUSES, key=lambda s: shares[s] - take[s], reverse=True)
    for s in by_remainder[: n - sum(take.values())]:
        take[s] += 1
    rng = random.Random(seed)
    picked = [r for s in STATUSES for r in rng.sample(by_status[s], take[s])]
    rng.shuffle(picked)  # mixed order, so the sheet does not reveal the status
    return picked


def write_sheet(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def cmd_sample(args: argparse.Namespace) -> None:
    rows = stratified_sample(read_reviewed(args.reviewed), args.n, args.seed)
    write_sheet(args.out_dir / "lemma_sheet.csv", ["id", "word", "lemma", "pos", "note"],
                [[r["id"], r["word"], "", "", ""] for r in rows])
    with EXAMPLES.open(encoding="utf-8") as f:
        examples = list(csv.DictReader(f))
    write_sheet(args.out_dir / "error_examples_sheet.csv",
                ["no", "word", "gold_lemma", "edit_kd_output", "lstm_kd_output", "category",
                 "category_ok", "better_category", "gold_ok", "correct_lemma", "note"],
                [[str(i), e["word"], e["gold"], e["pred_edit_kd_500k_s13"],
                  e["pred_lstm_kd_500k_s13"], CATEGORY_NAMES[e["category"]], "", "", "", "", ""]
                 for i, e in enumerate(examples, 1)])
    print(f"{len(rows)} rows ({dict(Counter(r['status'] for r in rows))}), "
          f"{len(examples)} error examples -> {args.out_dir}")


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def agreement_rows(first: dict[str, dict[str, str]],
                   second: list[dict[str, str]]) -> list[dict[str, str]]:
    pairs = [(first[r["id"]], normalize(r["lemma"].strip())) for r in second
             if r["lemma"].strip()]
    out = []
    for status in ("all", *STATUSES):
        sel = [(f, lem) for f, lem in pairs if status == "all" or f["status"] == status]
        if not sel:
            continue
        same = sum(normalize(f["lemma"]) == lem for f, lem in sel)
        kappa = cohen_kappa([normalize(f["lemma"]) == normalize(f["word"]) for f, _ in sel],
                            [lem == normalize(f["word"]) for f, lem in sel])
        out.append({"status": status, "n": str(len(sel)), "agreement": f"{same / len(sel):.4f}",
                    "kappa_reduce": f"{kappa:.4f}"})
    return out


def cmd_score(args: argparse.Namespace) -> None:
    first = {r["id"]: r for r in read_reviewed(args.reviewed)}
    with (args.out_dir / "lemma_sheet.csv").open(encoding="utf-8-sig") as f:
        second = list(csv.DictReader(f))
    unknown = [r["id"] for r in second if r["id"] not in first]
    if unknown:
        raise SystemExit(f"ids not in the reviewed batch: {unknown[:5]}")
    rows = agreement_rows(first, second)
    if not rows:
        raise SystemExit("no filled-in lemmas in the sheet yet")
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    with RESULT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['status']:9s} n={r['n']:>4s}  agreement {100 * float(r['agreement']):.1f}%"
              f"  kappa(reduce) {float(r['kappa_reduce']):.3f}")
    print(f"wrote {RESULT}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=200)
    s.add_argument("--seed", type=int, default=13)
    c = sub.add_parser("score")
    for p in (s, c):
        p.add_argument("--reviewed", type=Path, default=REVIEWED)
        p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    {"sample": cmd_sample, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
