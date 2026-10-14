"""Sequence-level knowledge distillation data: teacher-labelled training sets.

    uv run python -m tel.distill --teacher runs/teacher_lstm \\
        --extra data/interim/wiki_freq.tsv --top 20000 --out data/distill/teacher_lstm_wiki20k

    uv run python -m tel.distill --teacher runs/teacher_lstm --gold data/expanded/train.tsv \\
        --extra data/interim/wiki_freq.tsv --top 20000 --out data/distill/teacher_lstm_wiki20k_exp

Writes <out>/train.tsv (tel Pair format) containing
  * the gold training pairs (mode "gold+teacher", default), or the teacher's
    own predictions for them (mode "teacher", pure sequence-level KD); --gold
    replaces the teacher's train.tsv, e.g. with the human-reviewed expanded set
    (scripts/data/expand_candidates.py import), whose words are then not
    teacher-labelled again, plus
  * teacher-labelled extra words (most frequent first from a word list),
and <out>/distill_stats.json. Students point `data.train_file` at it; dev and
test still come from `data.dir`.

Leakage control for the extra words:
  * words that occur in dev/test are excluded;
  * words whose teacher lemma is a dev/test lemma are dropped, so test lemmas
    never appear as training targets and the unseen-lemma split stays unseen;
  * low-confidence predictions are dropped: per-token probability
    exp(logprob / (tokens + EOS)) below --min-token-prob. A per-token threshold
    avoids the length bias of whole-sequence probability (with label smoothing,
    long lemmas can never reach a high sequence probability). On dev,
    teacher_lstm at 0.90 keeps 48% of words at 90% precision.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from tel.data import Pair, read_tsv, write_tsv
from tel.export import load_run
from tel.inference import Seq2SeqLemmatizer
from tel.normalize import is_tamil_word, normalize


def read_word_list(path: Path, top: int | None) -> list[tuple[str, int]]:
    """(word, count) from a TSV with a `word` column (and optional `count`/`freq`)."""
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    out, seen = [], set()
    for r in rows:
        w = normalize(r["word"])
        if w in seen or not is_tamil_word(w):
            continue
        seen.add(w)
        out.append((w, int(r.get("count") or r.get("freq") or 1)))
        if top and len(out) >= top:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", type=Path, required=True, help="teacher run directory")
    ap.add_argument("--extra", type=Path, nargs="*", default=[], help="word lists (TSV)")
    ap.add_argument("--top", type=int, default=20000, help="most frequent extra words to use")
    ap.add_argument("--mode", choices=["gold+teacher", "teacher"], default="gold+teacher")
    ap.add_argument("--min-token-prob", type=float, default=0.9)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--gold", type=Path, help="gold pairs (default: <teacher data.dir>/train.tsv)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    model, model_type, _, tok, _, cfg = load_run(args.teacher)
    if model_type == "edit_classifier":
        raise SystemExit("teacher must be a seq2seq model")
    teacher = Seq2SeqLemmatizer(model, tok, cfg.data.max_len, beam=args.beam)
    data = Path(cfg.data.dir)
    gold = read_tsv(args.gold or data / "train.tsv")
    held_words = {p.word for s in ("dev", "test") for p in read_tsv(data / f"{s}.tsv")}
    held_lemmas = {line.strip() for s in ("dev", "test")
                   for line in (data / "splits" / f"{s}.txt").read_text(encoding="utf-8")
                   .splitlines() if line.strip()}
    name = args.teacher.name
    stats: dict[str, int | float | str] = {"teacher": name, "mode": args.mode,
                                           "min_token_prob": args.min_token_prob,
                                           "gold": len(gold),
                                           "gold_file": str(args.gold or data / "train.tsv")}

    pairs: list[Pair] = []
    if args.mode == "gold+teacher":
        pairs.extend(gold)
    else:
        preds = teacher.lemmatize([p.word for p in gold])
        pairs.extend(Pair(p.word, lem, p.pos, f"distill:{name}", p.freq)
                     for p, lem in zip(gold, preds, strict=True))
        stats["teacher_agrees_with_gold"] = sum(
            a.lemma == b.lemma for a, b in zip(pairs, gold, strict=True)) / len(gold)

    train_words = {p.word for p in gold}
    candidates: list[tuple[str, int]] = []
    for path in args.extra:
        for w, c in read_word_list(path, None):
            if w not in train_words and w not in held_words and len(w) <= cfg.data.max_len:
                candidates.append((w, c))
                if len(candidates) >= args.top:
                    break
    stats["extra_candidates"] = len(candidates)

    kept = dropped_conf = dropped_held = 0
    nbest = teacher.nbest([w for w, _ in candidates], n=1)
    for (w, c), hyps in zip(candidates, nbest, strict=True):
        lemma, logprob = hyps[0]
        tokens = len(tok.encode(lemma)) + 1  # + EOS
        if not lemma or math.exp(logprob / tokens) < args.min_token_prob:
            dropped_conf += 1
        elif lemma in held_lemmas:
            dropped_held += 1
        else:
            pairs.append(Pair(w, lemma, "OTHER", f"distill:{name}", c))
            kept += 1
    stats.update(extra_kept=kept, extra_dropped_low_confidence=dropped_conf,
                 extra_dropped_heldout_lemma=dropped_held, total=len(pairs))

    args.out.mkdir(parents=True, exist_ok=True)
    write_tsv(args.out / "train.tsv", pairs)
    (args.out / "distill_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
