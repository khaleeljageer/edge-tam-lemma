"""Tables for the paper (LaTeX, booktabs), generated from bench/results/.

    uv run python paper/figures/tables.py

Writes to paper/figures/out/:
  tab_accuracy.tex  baselines vs teachers vs compact models (test, test_unseen)
  tab_quant.tex     INT8 effect on the Raspberry Pi 5 (size, accuracy, latency)
  tab_hybrid.tex    lookup table size N vs hit rate, accuracy, Pi latency and memory
  tab_errors.tex    error categories per model (scripts/analysis/error_analysis.py)
  tab_error_examples.tex  categorized test errors of the deployed model

Tamil text is wrapped in \\ta{...}; define it in the paper preamble for XeLaTeX or
LuaLaTeX, e.g. \\newfontfamily\\tamilfont{Noto Serif Tamil}[Script=Tamil]
\\newcommand{\\ta}[1]{{\\tamilfont #1}}.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    BASE_NAMES,
    OUT,
    QUANT_NAMES,
    RUNTIME_NAMES,
    latex_table,
    read_csv,
    seed_stats,
)

COMPACT_SIZE = "500000"  # the deployed size budget
COMPACT_BASES = ["lstm", "edit", "transformer", "lstm_kd", "edit_kd"]
BASELINES = [("identity", "Identity (copy the word)"),
             ("suffix_stripper", "Suffix rules"),
             ("thamizhimorph", "ThamizhiMorph (FST)"),
             ("tamil_lemmatizer_pkg", "tamil-lemmatizer (PyPI)$^a$"),
             ("stanza", "Stanza$^b$")]
TEACHERS = [("teacher_lstm", "Teacher LSTM, beam 4", "15.8M"),
            ("teacher_transformer", "Teacher transformer, beam 4", "7.4M")]

QUANT_RUNS = [("onnxruntime", "fp32"), ("onnxruntime", "int8-dyn"), ("onnxruntime", "int8"),
              ("litert", "fp32"), ("litert", "int8-dyn")]
HYBRID_DEVICE = "m4_quant_hybrid/rpi5-hybrid.csv"
HYBRID_RUNS = [("edit_kd_500k_s13", "litert", "fp32"), ("edit_kd_500k_s13", "litert", "int8-dyn"),
               ("lstm_kd_500k_s13", "onnxruntime", "fp32"),
               ("lstm_kd_500k_s13", "litert", "int8-dyn")]


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def params_label(n: float) -> str:
    return f"{n / 1e6:.2f}M"


def accuracy_table(out_dir: Path) -> Path:
    base = {(r["model"], r["split"]): float(r["acc"]) for r in read_csv("accuracy_baselines.csv")}
    rows: list[list[str]] = []
    for name, shown in BASELINES:
        rows.append([shown, "--", pct(base[(name, "test")]), pct(base[(name, "test_unseen")])])
    rows.append(["--"])
    for name, shown, params in TEACHERS:
        rows.append([shown, params, pct(base[(name, "test")]), pct(base[(name, "test_unseen")])])
    rows.append(["--"])
    sweep = [r for r in read_csv("m3_sweep/accuracy.csv") if r["target_params"] == COMPACT_SIZE]
    acc = seed_stats(sweep, ("base", "split"), "acc")
    params = seed_stats(sweep, ("base",), "params")
    for b in COMPACT_BASES:
        cells = [f"{100 * acc[(b, s)][0]:.1f} $\\pm$ {100 * acc[(b, s)][1]:.1f}"
                 for s in ("test", "test_unseen")]
        rows.append([f"{BASE_NAMES[b]}, greedy" if "lstm" in b or b == "transformer"
                     else BASE_NAMES[b], params_label(params[(b,)][0]), *cells])
    n_seeds = max(n for _, _, n in acc.values())
    path = out_dir / "tab_accuracy.tex"
    latex_table(
        path, ["System", "Params", "Test", "Test unseen"], rows, "lrrr",
        caption=("Exact-match lemma accuracy (\\%). Compact models: 500k-parameter budget, "
                 f"mean $\\pm$ std over {n_seeds} seeds; KD = distilled from the teacher LSTM "
                 "on 9k Wikipedia words. $^a$Training data unknown. $^b$Trained on UD TTB "
                 "train, which overlaps our test words."),
        label="tab:accuracy")
    return path


def quant_table(out_dir: Path) -> Path:
    pi = read_csv("m4_quant/rpi5.csv")
    desk = read_csv("m4_quant/desktop.csv")

    def pick(rows, model, runtime, quant, threads=None, batch=None):
        return [r for r in rows if r["model"] == model and r["runtime"] == runtime
                and r["quant"] == quant and (threads is None or r["threads"] == threads)
                and (batch is None or r["batch"] == batch)]

    rows: list[list[str]] = []
    for model, shown in (("edit_kd_500k_s13", "Edit + KD"), ("lstm_kd_500k_s13", "LSTM + KD")):
        if rows:
            rows.append(["--"])
        for runtime, quant in QUANT_RUNS:
            all_pi = pick(pi, model, runtime, quant)
            if not all_pi:
                continue
            one = pick(pi, model, runtime, quant, "1", "1")[0]
            best = min(all_pi, key=lambda r: float(r["lat_med_ms"]))
            d1 = pick(desk, model, runtime, quant, "1", "1")
            rows.append([shown, f"{RUNTIME_NAMES[runtime]} {QUANT_NAMES[quant]}",
                         f"{float(one['size_kb']):,.0f}", pct(float(one["acc"])),
                         pct(float(one["agreement"])),
                         f"{float(d1[0]['lat_med_ms']):.3f}" if d1 else "--",
                         f"{float(one['lat_med_ms']):.3f}",
                         f"{float(best['lat_med_ms']):.3f} ({best['threads']}t, b{best['batch']})",
                         f"{float(best['peak_rss_mb']):.0f}"])
    path = out_dir / "tab_quant.tex"
    latex_table(
        path, ["Model", "Variant", "Size (KB)", "Test acc", "Agree", "Desktop 1t",
               "Pi 1t", "Pi best", "RSS (MB)"],
        rows, "llrrrrrlr",
        caption=("Post-training INT8 quantization. Median latency per word in ms (batch 1 "
                 "unless noted); Agree = word-level agreement with the fp32 PyTorch model on "
                 "test. int8 dyn: int8 weights, activations quantized per call; int8 static: "
                 "calibrated QDQ on Conv/MatMul/Gemm."),
        label="tab:quant")
    return path


def hybrid_table(out_dir: Path) -> Path:
    cov = {f"table_{int(r['n']) // 1000}k": r for r in read_csv("m4_hybrid/coverage.csv")}
    dev = [r for r in read_csv(HYBRID_DEVICE) if r["batch"] == "1"]

    def ms(model: str, runtime: str, quant: str, table: str) -> dict[str, str]:
        """Batch-1 row at the best thread count (lowest mean ms per token)."""
        rows = [r for r in dev if (r["model"], r["runtime"], r["quant"], r["table"])
                == (model, runtime, quant, table)]
        return min(rows, key=lambda r: 1 / float(r["wps"]))

    rows: list[list[str]] = []
    for table in ["", *sorted(cov, key=lambda t: int(cov[t]["n"]))]:
        cells = [ms(m, rt, q, table) for m, rt, q in HYBRID_RUNS]
        c = cov.get(table)
        rows.append([f"{int(c['n']):,}" if c else "none",
                     f"{float(c['table_kb']):,.0f}" if c else "0",
                     pct(float(cells[0]["hit_rate"] or 0)),
                     pct(float(cells[0]["acc"])), pct(float(cells[2]["acc"])),
                     *(f"{1000 / float(r['wps']):.3f}" for r in cells),
                     f"{float(cells[1]['peak_rss_mb']):.0f}"])
    path = out_dir / "tab_hybrid.tex"
    latex_table(
        path, ["N", "Table (KB)", "Hit (%)", "Acc edit", "Acc LSTM", "Edit fp32",
               "Edit int8", "LSTM fp32", "LSTM int8", "RSS (MB)"],
        rows, "rrrrrrrrrr",
        caption=("Hybrid lookup: the N most frequent Wikipedia word forms map to a lemma "
                 "(gold train label, else the majority vote of four systems); misses go to "
                 "the neural model. Test words weighted by UD corpus frequency (running "
                 "tokens); accuracy of the fp32 models (int8 within 0.3 points). Raspberry "
                 "Pi 5, batch 1, best thread count, mean ms per token: Edit + KD on LiteRT "
                 "(fp32 / int8 dynamic), LSTM + KD on ONNX Runtime fp32 and LiteRT int8 "
                 "dynamic. RSS: Edit + KD LiteRT int8."),
        label="tab:hybrid")
    return path


ERROR_RUNS = [("teacher_lstm", "Teacher"), ("lstm_500k_s13", "LSTM"),
              ("lstm_kd_500k_s13", "LSTM+KD"), ("edit_500k_s13", "Edit"),
              ("edit_kd_500k_s13", "Edit+KD")]
CATEGORY_NAMES = {"unchanged": "Form left unchanged",
                  "gold_is_form": "Gold keeps the form",
                  "suffix_left": "Suffix partly left",
                  "over_stripped": "Too much removed",
                  "stem_final": "Stem-final letter wrong",
                  "wrong_stem": "Other / wrong stem"}


def errors_table(out_dir: Path) -> Path:
    summary = read_csv("m5_errors/categories.csv")
    n = {(r["run"], r["category"]): int(r["errors"]) for r in summary}
    rows = [[CATEGORY_NAMES[c], *(str(n[(run, c)]) for run, _ in ERROR_RUNS)]
            for c in CATEGORY_NAMES]
    rows += [["--"], ["Total errors", *(str(sum(n[(run, c)] for c in CATEGORY_NAMES))
                                        for run, _ in ERROR_RUNS)]]
    errors = read_csv("m5_errors/errors.csv")
    main = [r for r in errors if r["cat_edit_kd_500k_s13"]]
    oov = sum(r["script_in_vocab"] == "0" for r in main)
    all_wrong = sum(int(r["n_wrong"]) == len(ERROR_RUNS) for r in errors)
    path = out_dir / "tab_errors.tex"
    latex_table(
        path, ["Category", *(name for _, name in ERROR_RUNS)], rows, "l" + "r" * len(ERROR_RUNS),
        caption=("Test errors by category (rule-based, \\texttt{tel.error\\_analysis}); "
                 "500k models, seed 13. Distillation mainly removes ``form left unchanged'' "
                 f"errors of the edit classifier. {oov} of its {len(main)} errors need an edit "
                 f"script outside its vocabulary; {all_wrong} test pairs are wrong in all five "
                 "models."),
        label="tab:errors")
    return path


def error_examples_table(out_dir: Path) -> Path:
    rows = [[CATEGORY_NAMES[r["category"]], f"\\ta{{{r['word']}}}", f"\\ta{{{r['gold']}}}",
             f"\\ta{{{r['pred_edit_kd_500k_s13']}}}", f"\\ta{{{r['pred_lstm_kd_500k_s13']}}}",
             r["pos"].lower()]
            for r in read_csv("m5_errors/examples.csv")]
    path = out_dir / "tab_error_examples.tex"
    latex_table(
        path, ["Category", "Word", "Gold", "Edit+KD", "LSTM+KD", "POS"], rows, "llllll",
        caption=("Test errors of the deployed model (Edit+KD, 500k), sampled with a fixed seed "
                 "across categories, with the distilled LSTM's output for comparison."),
        label="tab:error-examples")
    return path


def main(out_dir: Path = OUT) -> list[Path]:
    return [accuracy_table(out_dir), quant_table(out_dir), hybrid_table(out_dir),
            errors_table(out_dir), error_examples_table(out_dir)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    for p in main(ap.parse_args().out):
        print(p)
