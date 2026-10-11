"""Add trained runs (runs/<name>/) to accuracy_baselines.csv, scored on the same
splits as the baselines, including the leakage-free `test_clean`.

Seq2seq runs are decoded with the beam size in their config.

    uv run python scripts/baselines/collect_runs.py teacher_lstm lstm_tiny edit_tiny
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, run_baseline  # noqa: E402

from tel.export import load_run  # noqa: E402
from tel.inference import EditLemmatizer, Seq2SeqLemmatizer  # noqa: E402
from tel.models.common import count_params  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    for name in args.runs:
        model, model_type, _, tok, vocab, cfg = load_run(ROOT / "runs" / name)
        if model_type == "edit_classifier":
            lem, decode = EditLemmatizer(model, tok, vocab, cfg.data.max_len), "single pass"
        else:
            lem = Seq2SeqLemmatizer(model, tok, cfg.data.max_len, beam=cfg.decode.beam,
                                    alpha=cfg.decode.alpha)
            decode = f"beam {cfg.decode.beam}" if cfg.decode.beam > 1 else "greedy"
        run_baseline(name, lem.lemmatize,
                     f"ours: {model_type}, {count_params(model) / 1e6:.2f}M params, {decode}")


if __name__ == "__main__":
    main()
