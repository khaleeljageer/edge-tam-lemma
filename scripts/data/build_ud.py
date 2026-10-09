"""Build word/lemma pairs from UD Tamil and split them by lemma.

Reads data/raw/ud/**/*.conllu, keeps written (surface) words with a Tamil form
and lemma, merges duplicates (counting frequency), and writes
data/processed/{train,dev,test,test_unseen}.tsv plus splits/*.txt lemma lists.

The original UD train/dev/test boundaries are ignored on purpose: the split is
by lemma so test lemmas never appear in training.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from tel.data import Pair, make_splits, map_upos, write_tsv
from tel.normalize import clean_word, is_tamil_word

# Function-word parts of a multiword token; the lemma comes from the first other part.
_FUNCTION_UPOS = {"PART", "AUX", "ADP", "CCONJ", "SCONJ", "DET", "PUNCT"}


def _head_part(parts: list[tuple[str, str, str]]) -> tuple[str, str, str]:
    for part in parts:
        if part[2] not in _FUNCTION_UPOS:
            return part
    return parts[0]


def read_conllu(path: Path) -> list[tuple[str, str, str]]:
    """Return (written form, lemma, upos) for every surface token.

    UD splits some written words into syntactic words, e.g.
    உறுதியளித்துள்ளார் = உறுதியளித்த் (VERB) + உள்ளார் (AUX). The split parts are
    not written words, so a multiword token is kept as one item whose lemma and
    POS come from its first content part (here உறுதியளி, VERB).
    """
    rows = []
    mwt_form, mwt_end, parts = "", 0, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            tok_id = cols[0]
            if "." in tok_id:
                continue  # empty node
            if "-" in tok_id:
                mwt_form, mwt_end, parts = cols[1], int(tok_id.split("-")[1]), []
                continue
            word = (cols[1], cols[2], cols[3])
            if mwt_end:
                parts.append(word)
                if int(tok_id) == mwt_end:
                    _, lemma, upos = _head_part(parts)
                    rows.append((mwt_form, lemma, upos))
                    mwt_end = 0
            else:
                rows.append(word)
    return rows


def collect_pairs(conllu_files: list[Path]) -> tuple[list[Pair], Counter[str]]:
    counts: Counter[tuple[str, str, str, str]] = Counter()
    dropped: Counter[str] = Counter()
    for path in conllu_files:
        source = path.parent.name.removeprefix("UD_Tamil-").lower()
        for form, lemma, upos in read_conllu(path):
            word, lem = clean_word(form), clean_word(lemma)
            if upos == "PUNCT" or not word:
                dropped["punct_or_empty"] += 1
            elif not (is_tamil_word(word) and is_tamil_word(lem)):
                dropped["non_tamil"] += 1
            else:
                counts[(word, lem, map_upos(upos), f"ud-{source}")] += 1
    pairs = [Pair(w, lem, pos, src, freq) for (w, lem, pos, src), freq in sorted(counts.items())]
    return pairs, dropped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw/ud")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    files = sorted(Path(args.raw).glob("*/*.conllu"))
    if not files:
        raise SystemExit(f"no .conllu files under {args.raw}; run fetch_ud.py first")
    pairs, dropped = collect_pairs(files)
    splits, lemma_lists = make_splits(pairs, seed=args.seed)

    out = Path(args.out)
    train_lemmas = set(lemma_lists["train"])
    for name, split_pairs in splits.items():
        write_tsv(out / f"{name}.tsv", split_pairs)
        n_seen = sum(p.lemma in train_lemmas for p in split_pairs)
        print(f"{name}: {len(split_pairs)} pairs ({n_seen} seen-lemma)")
    (out / "splits").mkdir(parents=True, exist_ok=True)
    for name, lemmas in lemma_lists.items():
        (out / "splits" / f"{name}.txt").write_text(
            "".join(f"{lemma}\n" for lemma in lemmas), encoding="utf-8"
        )
    print(f"dropped: {dict(dropped)}")


if __name__ == "__main__":
    main()
