"""Figure: test accuracy vs parameters, with and without distillation.

    uv run python paper/figures/fig_params.py

Reads bench/results/m3_sweep/accuracy.csv (mean +- std over seeds per base model
and size budget) and bench/results/accuracy_baselines.csv (reference lines).
Writes paper/figures/out/acc_vs_params.{pdf,png}.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import BASE_NAMES, OUT, read_csv, save, seed_stats, setup_matplotlib
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from matplotlib.transforms import blended_transform_factory

SWEEP = "m3_sweep/accuracy.csv"
BASES = ["lstm", "lstm_kd", "edit", "edit_kd", "transformer"]
STYLE = {"lstm": ("C0", "o", "--"), "lstm_kd": ("C0", "o", "-"),
         "edit": ("C1", "s", "--"), "edit_kd": ("C1", "s", "-"),
         "transformer": ("C2", "^", "--")}
REFERENCES = {"teacher_lstm": "teacher (15.8M)",
              "tamil_lemmatizer_pkg": "tamil-lemmatizer",
              "thamizhimorph": "ThamizhiMorph",
              "suffix_stripper": "suffix rules"}


def series(split: str) -> dict[str, list[tuple[float, float, float]]]:
    """base -> [(mean params, mean acc, std acc)] sorted by size budget."""
    rows = [r for r in read_csv(SWEEP) if r["split"] == split]
    acc = seed_stats(rows, ("base", "target_params"), "acc")
    params = seed_stats(rows, ("base", "target_params"), "params")
    out: dict[str, list[tuple[float, float, float]]] = {}
    for (base, target), (mean, std, _) in sorted(acc.items(), key=lambda kv: int(kv[0][1])):
        out.setdefault(base, []).append((params[(base, target)][0], mean, std))
    return out


def main(out_dir: Path = OUT, split: str = "test") -> list[Path]:
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    data = series(split)
    for base in BASES:
        if base not in data:
            continue
        xs, ys, es = zip(*data[base], strict=True)
        color, marker, line = STYLE[base]
        ax.errorbar(xs, [100 * y for y in ys], yerr=[100 * e for e in es], color=color,
                    marker=marker, linestyle=line, markersize=3.5, linewidth=1, capsize=2,
                    label=BASE_NAMES[base])
    refs = {r["model"]: float(r["acc"]) for r in read_csv("accuracy_baselines.csv")
            if r["split"] == split}
    # Reference systems: dotted lines, labelled in the right margin.
    margin = blended_transform_factory(ax.transAxes, ax.transData)
    for name, label in REFERENCES.items():
        if name in refs:
            y = 100 * refs[name]
            ax.axhline(y, color="0.55", linewidth=0.6, linestyle=":")
            ax.text(1.02, y, label, transform=margin, fontsize=6, color="0.35", va="center")
    ax.set_xscale("log")
    ticks = sorted({int(r["target_params"]) for r in read_csv(SWEEP)})  # size budgets
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1e6:g}M"))
    ax.set_xlabel("Parameters")
    ax.set_ylabel(f"Exact-match accuracy, {split} (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, ncol=3)
    paths = save(fig, out_dir, "acc_vs_params")
    plt.close(fig)
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["dev", "test", "test_unseen"])
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    for p in main(args.out, args.split):
        print(p)
