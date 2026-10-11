"""Baseline: Stanza's Tamil lemmatizer (Stanford NLP), used as a black box.

Each word is processed as its own document (no sentence context, like our
models). Stanza may split a written word into syntactic words (MWT); the lemma
is taken from the first content word, the same rule used to build our data
(scripts/data/build_ud.py). Stanza's Tamil models are trained on UD Tamil-TTB
train, so compare on `test_clean`.

    uv run --project envs/baselines python scripts/baselines/run_stanza.py
"""

from __future__ import annotations

import sys
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, run_baseline  # noqa: E402

from tel.normalize import clean_word  # noqa: E402

MODEL_DIR = ROOT / "data" / "raw" / "stanza_resources"
FUNCTION_UPOS = {"PART", "AUX", "ADP", "CCONJ", "SCONJ", "DET", "PUNCT"}


def main() -> None:
    import stanza

    stanza.download("ta", model_dir=str(MODEL_DIR), processors="tokenize,mwt,pos,lemma",
                    verbose=False)
    nlp = stanza.Pipeline("ta", model_dir=str(MODEL_DIR), processors="tokenize,mwt,pos,lemma",
                          download_method=None, verbose=False)

    def lemmatize(words: list[str]) -> list[str | None]:
        out: list[str | None] = []
        docs = nlp([stanza.Document([], text=w) for w in words])
        for doc in docs:
            parts = [wd for sent in doc.sentences for wd in sent.words]
            head = next((wd for wd in parts if wd.upos not in FUNCTION_UPOS), parts[0] if parts
                        else None)
            lemma = clean_word(head.lemma) if head is not None and head.lemma else ""
            out.append(lemma or None)
        return out

    run_baseline("stanza", lemmatize,
                 f"stanza {version('stanza')} ta (trained on UD TTB train: use test_clean)")


if __name__ == "__main__":
    main()
