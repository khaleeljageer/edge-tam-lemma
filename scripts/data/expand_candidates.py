"""Data expansion: candidate lemmas for frequent Wikipedia words, for human review.

    uv run python scripts/data/expand_candidates.py calibrate              # on dev
    uv run python scripts/data/expand_candidates.py propose --top 3000 \\
        --out data/annotation/wiki_batch01.tsv
    # ... review the TSV (see below) ...
    uv run python scripts/data/expand_candidates.py import data/annotation/wiki_batch01.tsv \\
        --out data/expanded/train.tsv

Sources, each giving one lemma per word (the vote):
  thamizhimorph  rule-based FST analyser (first analysing FST, first analysis;
                 guessers only if no analyser knows the word; `tm_source` says which)
  teacher        teacher_lstm, beam 4, with its per-token probability
  lstm_kd        distilled 0.5M LSTM student
  edit_kd        distilled 0.5M edit-tree classifier
`suggested` is the majority lemma (ties go to the earlier source in that list)
and `status` is `agree` (all sources give it), `majority` or `conflict`.
`calibrate` runs the same pipeline on dev words and reports, per status, how
many words fall in it and how often `suggested` is right, so a reviewer knows
where to look hardest (and whether `agree` rows could be accepted unreviewed).

Reviewing a proposal TSV: `lemma` is prefilled with `suggested`. Correct it if
needed, optionally set `pos` (NOUN / VERB / OTHER), and set `checked` to `y`
(accept), or `x` (not a real word / skip). Rows left unchecked are ignored.

Leakage control (same rules as tel.distill): candidate words exclude every
train/dev/test word, and `import` drops pairs whose lemma is a dev/test lemma,
so the unseen-lemma test split stays unseen. Wikipedia text is CC BY-SA; see
docs/data_licenses.md.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path

from tel.data import Pair, read_tsv, write_tsv
from tel.export import load_run
from tel.inference import EditLemmatizer, Seq2SeqLemmatizer
from tel.normalize import clean_word, normalize

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
WIKI = ROOT / "data" / "interim" / "wiki_freq.tsv"
SOURCES = ("thamizhimorph", "teacher", "lstm_kd", "edit_kd")
RUNS = {"teacher": "teacher_lstm", "lstm_kd": "lstm_kd_500k_s13", "edit_kd": "edit_kd_500k_s13"}
COLUMNS = ["id", "word", "count", "suggested", "status", *SOURCES, "tm_source", "tm_all",
           "teacher_prob", "lemma", "pos", "checked", "note"]
POS = {"NOUN", "VERB", "OTHER"}


def _thamizhimorph():
    path = ROOT / "scripts" / "baselines" / "run_thamizhimorph.py"
    spec = importlib.util.spec_from_file_location("run_thamizhimorph", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def thamizhimorph_lemmas(words: list[str]) -> dict[str, tuple[str, str, list[str]]]:
    """word -> (lemma, "analyser" | "guesser" | "none", all distinct lemmas of that FST)."""
    if shutil.which("flookup") is None:
        sys.exit("foma's flookup not found: install foma first (sudo apt install foma)")
    tm = _thamizhimorph()
    tm.fetch_fsts()
    out: dict[str, tuple[str, str, list[str]]] = {}
    todo = sorted(set(words))
    for group, fsts in (("analyser", tm.ANALYSERS), ("guesser", tm.GUESSERS)):
        for fst in fsts:
            if not todo:
                break
            for w, analyses in tm.flookup(fst, todo).items():
                if analyses:
                    lemmas = list(dict.fromkeys(
                        clean_word(normalize(a.split("+", 1)[0])) or w for a in analyses))
                    out[w] = (lemmas[0], group, lemmas)
            todo = [w for w in todo if w not in out]
    out.update({w: (w, "none", []) for w in todo})
    return out


def model_lemmas(words: list[str]) -> dict[str, list]:
    """source -> per-word lemmas; also "teacher_prob" -> per-token probabilities."""
    out: dict[str, list] = {}
    for source, run in RUNS.items():
        model, model_type, _, tok, vocab, cfg = load_run(ROOT / "runs" / run)
        if model_type == "edit_classifier":
            out[source] = EditLemmatizer(model, tok, vocab, cfg.data.max_len).lemmatize(words)
            continue
        beam = 4 if source == "teacher" else cfg.decode.beam
        lem = Seq2SeqLemmatizer(model, tok, cfg.data.max_len, beam=beam)
        hyps = [h[0] for h in lem.nbest(words, n=1)]
        out[source] = [lemma for lemma, _ in hyps]
        if source == "teacher":
            out["teacher_prob"] = [math.exp(lp / (len(tok.encode(lemma)) + 1))
                                   for lemma, lp in hyps]
    return out


def vote(candidates: list[str]) -> tuple[str, str]:
    """Majority lemma (ties -> earliest source) and agree / majority / conflict."""
    counts = Counter(candidates)
    best = max(counts.values())
    suggested = next(c for c in candidates if counts[c] == best)
    if best == len(candidates):
        return suggested, "agree"
    return suggested, "majority" if best > len(candidates) / 2 else "conflict"


def propose_rows(words: list[tuple[str, int]]) -> list[dict]:
    ws = [w for w, _ in words]
    tm = thamizhimorph_lemmas(ws)
    models = model_lemmas(ws)
    rows = []
    for i, (w, count) in enumerate(words):
        cands = {"thamizhimorph": tm[w][0], **{s: models[s][i] for s in RUNS}}
        suggested, status = vote([cands[s] for s in SOURCES])
        rows.append({"id": i + 1, "word": w, "count": count, "suggested": suggested,
                     "status": status, **cands, "tm_source": tm[w][1],
                     "tm_all": "|".join(tm[w][2]),
                     "teacher_prob": f"{models['teacher_prob'][i]:.3f}",
                     "lemma": suggested, "pos": "", "checked": "", "note": ""})
    return rows


def held_out() -> tuple[set[str], set[str], set[str]]:
    """(train words, dev/test words, dev/test lemmas)."""
    train = {p.word for p in read_tsv(DATA / "train.tsv")}
    words = {p.word for s in ("dev", "test") for p in read_tsv(DATA / f"{s}.tsv")}
    lemmas = {line.strip() for s in ("dev", "test")
              for line in (DATA / "splits" / f"{s}.txt").read_text(encoding="utf-8").splitlines()
              if line.strip()}
    return train, words, lemmas


def read_wiki(top: int, exclude: set[str], max_len: int = 48) -> list[tuple[str, int]]:
    out = []
    with open(WIKI, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            w = clean_word(normalize(r["word"]))
            if w and w not in exclude and len(w) <= max_len:
                exclude.add(w)
                out.append((w, int(r["count"])))
                if len(out) >= top:
                    break
    return out


def write_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t", quoting=csv.QUOTE_NONE,
                           escapechar="\\")
        w.writeheader()
        w.writerows(rows)


def cmd_propose(args: argparse.Namespace) -> None:
    train, held_words, _ = held_out()
    words = read_wiki(args.top, train | held_words)
    rows = propose_rows(words)
    write_rows(rows, args.out)
    print(f"{len(rows)} words -> {args.out}:", dict(Counter(r["status"] for r in rows)))


def calibration(rows: list[dict], gold: dict[str, str]) -> dict:
    """Per status: share of words and accuracy of `suggested` and of each source."""
    out = {}
    for status in ("agree", "majority", "conflict", "all"):
        sel = [r for r in rows if status in ("all", r["status"])]
        if not sel:
            continue
        acc = {s: sum(r[s] == gold[r["word"]] for r in sel) / len(sel)
               for s in ("suggested", *SOURCES)}
        out[status] = {"n": len(sel), "share": len(sel) / len(rows),
                       **{f"acc_{k}": round(v, 4) for k, v in acc.items()}}
    return out


def cmd_calibrate(args: argparse.Namespace) -> None:
    dev = read_tsv(DATA / "dev.tsv")
    gold = {p.word: p.lemma for p in dev}
    rows = propose_rows([(w, 0) for w in dict.fromkeys(p.word for p in dev)])
    report = calibration(rows, gold)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["status", *next(iter(report.values()))])
        w.writeheader()
        w.writerows({"status": k, **{c: f"{x:.4f}" if isinstance(x, float) else x
                                     for c, x in v.items()}} for k, v in report.items())
    for status, r in report.items():
        print(f"{status:9} n={r['n']:4} share={r['share']:.0%} suggested={r['acc_suggested']:.1%}  "
              + "  ".join(f"{s}={r['acc_' + s]:.1%}" for s in SOURCES))


def reviewed_pairs(rows: list[dict], held_words: set[str], held_lemmas: set[str],
                   source: str) -> tuple[list[Pair], Counter]:
    stats: Counter = Counter()
    pairs = []
    for r in rows:
        mark = (r.get("checked") or "").strip().lower()
        if mark != "y":
            stats["skipped" if mark == "x" else "unchecked"] += 1
            continue
        word, lemma = clean_word(normalize(r["word"])), clean_word(normalize(r["lemma"] or ""))
        pos = (r.get("pos") or "").strip().upper()
        if not word or not lemma:
            stats["empty"] += 1
        elif word in held_words or lemma in held_lemmas:
            stats["dropped_heldout"] += 1
        else:
            pairs.append(Pair(word, lemma, pos if pos in POS else "OTHER", source,
                              int(r.get("count") or 1)))
            stats["kept"] += 1
            stats["corrected"] += lemma != r["suggested"]
    return pairs, stats


def cmd_import(args: argparse.Namespace) -> None:
    train, held_words, held_lemmas = held_out()
    pairs: list[Pair] = list(read_tsv(DATA / "train.tsv"))
    seen = set(train)
    report = {}
    for path in args.reviewed:
        with open(path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE,
                                       escapechar="\\"))
        new, stats = reviewed_pairs(rows, held_words, held_lemmas, f"wiki-reviewed:{path.stem}")
        new = [p for p in new if p.word not in seen]
        seen.update(p.word for p in new)
        pairs.extend(new)
        report[path.name] = dict(stats)
    write_tsv(args.out, pairs)
    report["total_pairs"] = len(pairs)
    args.out.with_suffix(".stats.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("propose", help="candidate lemmas for frequent Wikipedia words")
    p.add_argument("--top", type=int, default=3000)
    p.add_argument("--out", type=Path, default=ROOT / "data/annotation/wiki_batch01.tsv")
    c = sub.add_parser("calibrate", help="how reliable each status is, on dev")
    c.add_argument("--out", type=Path,
                   default=ROOT / "bench/results/m3_expansion/calibration_dev.csv")
    i = sub.add_parser("import", help="reviewed TSVs -> training file (gold train + reviewed)")
    i.add_argument("reviewed", nargs="+", type=Path)
    i.add_argument("--out", type=Path, default=ROOT / "data/expanded/train.tsv")
    args = ap.parse_args()
    {"propose": cmd_propose, "calibrate": cmd_calibrate, "import": cmd_import}[args.cmd](args)


if __name__ == "__main__":
    main()
