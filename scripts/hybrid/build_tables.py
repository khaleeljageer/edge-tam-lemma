"""Build hybrid lookup tables (top-N Wikipedia word forms -> lemma) and their coverage.

    uv run python scripts/hybrid/build_tables.py                    # N = 1k 5k 10k 50k
    uv run python scripts/hybrid/build_tables.py \\
        --reviewed data/annotation/wiki_batch01_reviewed.tsv \\
        --tag reviewed     # -> table_<N>_reviewed.tsv, coverage_reviewed.csv

1. Silver labels for the top --max words of data/interim/wiki_freq.tsv: the
   majority vote of ThamizhiMorph, teacher_lstm and the two distilled students
   (scripts/data/expand_candidates.py), cached in
   data/interim/hybrid/silver_wiki<max>.tsv (slow: the teacher decodes with beam 4).
2. One table per N: data/interim/hybrid/table_<N>.tsv, labels in priority order
   gold train pair > reviewed annotation > silver.
3. bench/results/m4_hybrid/coverage.csv: per N, table rows by source, file size,
   share of Wikipedia running tokens covered, and on dev/test (types, and tokens
   weighted by UD corpus frequency) the hit rate and the table's accuracy on hits.

No train/dev/test word is excluded from the tables: a deployed table covers
whatever is frequent. Dev/test words get silver labels from models that never
saw them (and from the rule-based analyser), and test lemmas never enter
training, so the evaluation stays leakage-free.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path

from tel.data import read_tsv
from tel.hybrid import build_table, gold_lemmas, read_table, write_table

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
WIKI = ROOT / "data" / "interim" / "wiki_freq.tsv"
OUT = ROOT / "data" / "interim" / "hybrid"
RESULTS = ROOT / "bench" / "results" / "m4_hybrid" / "coverage.csv"


def _expand():
    path = ROOT / "scripts" / "data" / "expand_candidates.py"
    spec = importlib.util.spec_from_file_location("expand_candidates", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wiki_counts(top: int) -> tuple[list[tuple[str, int]], int]:
    """Top words (NFC, cleaned, deduplicated) with counts, and total Wikipedia tokens."""
    exp = _expand()
    words = exp.read_wiki(top, set())
    stats = dict(line.split("\t", 1) for line in
                 WIKI.with_suffix(".stats.tsv").read_text(encoding="utf-8").splitlines())
    return words, int(stats["tokens"])


def silver_labels(words: list[tuple[str, int]], path: Path) -> dict[str, str]:
    if not path.exists():
        exp = _expand()
        rows = exp.propose_rows(words)
        exp.write_rows(rows, path)
        print(f"silver labels -> {path}")
    return read_table_column(path, "suggested")


def read_table_column(path: Path, column: str) -> dict[str, str]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        return {r["word"]: r[column] for r in rows}


def reviewed_labels(paths: list[Path]) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in paths:
        with open(path, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\"):
                if (r.get("checked") or "").strip().lower() == "y" and r.get("lemma"):
                    out.setdefault(r["word"], r["lemma"])
    return out


def coverage(table: dict[str, str], pairs) -> dict[str, float]:
    """Hit rate and accuracy-on-hits over types and over frequency-weighted tokens."""
    hits = [p for p in pairs if p.word in table]
    tokens = sum(p.freq for p in pairs)
    hit_tokens = sum(p.freq for p in hits)
    return {
        "type_hit": len(hits) / len(pairs),
        "token_hit": hit_tokens / tokens,
        "type_acc_on_hits": sum(table[p.word] == p.lemma for p in hits) / len(hits)
        if hits else 0.0,
        "token_acc_on_hits": sum(p.freq * (table[p.word] == p.lemma) for p in hits) / hit_tokens
        if hit_tokens else 0.0,
    }


def label(n: int) -> str:
    return f"{n // 1000}k" if n % 1000 == 0 else str(n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000, 10000, 50000])
    ap.add_argument("--reviewed", type=Path, nargs="*", default=[])
    ap.add_argument("--tag", default="", help="suffix for table and coverage file names, so "
                    "earlier tables and results are kept")
    args = ap.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""
    results_path = RESULTS.with_name(f"{RESULTS.stem}{suffix}.csv")

    top = max(args.sizes)
    words, total_tokens = wiki_counts(top)
    ranked = [w for w, _ in words]
    count = dict(words)
    silver = silver_labels(words, OUT / f"silver_wiki{label(top)}.tsv")
    gold = gold_lemmas(read_tsv(DATA / "train.tsv"))
    reviewed = reviewed_labels(args.reviewed)
    splits = {s: read_tsv(DATA / f"{s}.tsv") for s in ("dev", "test")}

    results = []
    for n in sorted(args.sizes):
        rows = build_table(ranked, n, [("gold", gold), ("reviewed", reviewed),
                                       ("silver", silver)])
        path = OUT / f"table_{label(n)}{suffix}.tsv"
        write_table(path, rows)
        table = read_table(path)
        row = {"n": n, "rows": len(rows),
               **{f"{src}_rows": sum(r[2] == src for r in rows)
                  for src in ("gold", "reviewed", "silver")},
               "table_kb": round(path.stat().st_size / 1024, 1),
               "wiki_token_hit": sum(count[w] for w in table) / total_tokens}
        for split, pairs in splits.items():
            row.update({f"{split}_{k}": v for k, v in coverage(table, pairs).items()})
        results.append(row)
        print(f"N={n:>6}: {len(rows)} rows, {row['table_kb']} KB, wiki tokens "
              f"{row['wiki_token_hit']:.1%}, test tokens hit {row['test_token_hit']:.1%} "
              f"(acc on hits {row['test_token_acc_on_hits']:.1%})")

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in r.items()}
                    for r in results)
    print(f"wrote {results_path}")


if __name__ == "__main__":
    main()
