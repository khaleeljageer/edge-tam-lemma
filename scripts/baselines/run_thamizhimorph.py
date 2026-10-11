"""Baseline: ThamizhiMorph (Sarveswaran et al., Apache-2.0), a rule-based foma FST
morphological analyser, used as a black box through foma's `flookup`.

Needs foma installed (`sudo apt install foma`). The FST files are downloaded at
a pinned commit into data/raw/thamizhi-morph/.

Lookup order (no POS tags, unlike the tool's own pipeline, which picks a
guesser from a POS tagger): the analysers in a fixed priority order; the first
FST that returns an analysis wins and its first analysis gives the lemma (the
text before the first "+"). Words no analyser knows go to the guessers in a
fixed order. Words nothing analyses are left unchanged (identity) and counted
in `notes`, not as crashes.

    uv run python scripts/baselines/run_thamizhimorph.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, run_baseline  # noqa: E402

from tel.normalize import clean_word, normalize  # noqa: E402

COMMIT = "adbacceda5e8"  # sarves/thamizhi-morph master on 2026-05-06
BASE = f"https://raw.githubusercontent.com/sarves/thamizhi-morph/{COMMIT}/FST-Models"
FST_DIR = ROOT / "data" / "raw" / "thamizhi-morph" / COMMIT
ANALYSERS = ["noun.fst", "verb-c3.fst", "verb-c4.fst", "verb-c62.fst", "verb-c11.fst",
             "verb-c12.fst", "verb-c-rest.fst", "pronoun.fst", "adj.fst", "adv.fst", "part.fst"]
GUESSERS = ["noun-guess.fst", "verb-guess.fst", "adj-guess.fst", "adv-guess.fst"]


def fetch_fsts() -> None:
    FST_DIR.mkdir(parents=True, exist_ok=True)
    for name in ANALYSERS + GUESSERS:
        if not (FST_DIR / name).exists():
            urllib.request.urlretrieve(f"{BASE}/{name}", FST_DIR / name)


def flookup(fst: str, words: list[str]) -> dict[str, list[str]]:
    """Run all words through one FST -> {word: [analysis, ...]} (empty if unknown)."""
    proc = subprocess.run(["flookup", str(FST_DIR / fst)], input="\n".join(words) + "\n",
                          capture_output=True, text=True, check=True)
    out: dict[str, list[str]] = {w: [] for w in words}
    for line in proc.stdout.splitlines():
        if "\t" not in line:
            continue
        word, analysis = line.split("\t", 1)
        if analysis.strip() and not analysis.startswith("+?") and word in out:
            out[word].append(analysis.strip())
    return out


def main() -> None:
    if shutil.which("flookup") is None:
        sys.exit("foma's flookup not found: install foma first (sudo apt install foma)")
    fetch_fsts()
    stats = {"analyser": 0, "guesser": 0, "unanalysed": 0}

    def lemmatize(words: list[str]) -> list[str | None]:
        lemma: dict[str, str] = {}
        todo = sorted(set(words))
        for group, fsts in (("analyser", ANALYSERS), ("guesser", GUESSERS)):
            for fst in fsts:
                if not todo:
                    break
                found = flookup(fst, todo)
                for w, analyses in found.items():
                    if analyses:
                        lemma[w] = clean_word(normalize(analyses[0].split("+", 1)[0])) or w
                        stats[group] += 1
                todo = [w for w in todo if w not in lemma]
        stats["unanalysed"] += len(todo)
        return [lemma.get(w, w) for w in words]

    run_baseline("thamizhimorph", lemmatize, f"thamizhi-morph@{COMMIT} via flookup; "
                 "fixed FST order, first analysis, identity if unanalysed")
    total = sum(stats.values())
    print("coverage over all split lookups:",
          {k: f"{v} ({v / total:.0%})" for k, v in stats.items()})


if __name__ == "__main__":
    main()
