"""Figure: accuracy vs median per-word latency (batch 1), Pareto front per device.

    uv run python paper/figures/fig_pareto.py

One point per (device, model, precision, runtime) at its best thread count, from
the device benchmark CSVs listed in common.DEVICE_RUNS (later files override
earlier ones). Plain neural models only: hybrid (model + lookup table) rows are
measured on a frequency-weighted token stream and are reported in the hybrid
table instead. Writes paper/figures/out/pareto.{pdf,png}.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from common import (
    BASE_NAMES,
    DEVICE_NAMES,
    OUT,
    QUANT_NAMES,
    RUNTIME_NAMES,
    best_by,
    device_rows,
    pareto_front,
    save,
    setup_matplotlib,
)
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

MARKERS = {"lstm": "o", "lstm_kd": "o", "edit": "s", "edit_kd": "s", "char_cnn": "D",
           "transformer": "^"}
COLORS = {"rpi5": "C3", "desktop": "C0"}


def base_of(model: str) -> str:
    """'edit_kd_500k_s13' -> 'edit_kd', 'lstm_tiny' -> 'lstm'."""
    return re.sub(r"_(tiny|\d+[kM])(_s\d+)?$", "", model)


def size_of(model: str) -> str:
    m = re.search(r"_(tiny|\d+[kM])(_s\d+)?$", model)
    return m.group(1) if m else ""


def label(r: dict[str, str]) -> str:
    return (f"{BASE_NAMES.get(base_of(r['model']), r['model'])} {size_of(r['model'])}, "
            f"{RUNTIME_NAMES.get(r['runtime'], r['runtime'])} {QUANT_NAMES[r['quant']]}")


def points(batch: str = "1") -> dict[str, list[dict[str, str]]]:
    rows = [r for r in device_rows() if r["batch"] == batch and r["input"] == "test.tsv"]
    best = best_by(rows, ("device", "model", "quant", "runtime"))
    out: dict[str, list[dict[str, str]]] = {}
    for r in sorted(best, key=lambda r: float(r["lat_med_ms"])):
        out.setdefault(r["device"], []).append(r)
    return out


def main(out_dir: Path = OUT, batch: str = "1") -> list[Path]:
    plt = setup_matplotlib()
    data = points(batch)
    devices = [d for d in ("rpi5", "desktop") if d in data]
    fig, axes = plt.subplots(1, len(devices), figsize=(3.4 * len(devices), 2.8), sharey=True,
                             squeeze=False)
    for ax, device in zip(axes[0], devices, strict=True):
        rows = data[device]
        xy = [(float(r["lat_med_ms"]), 100 * float(r["acc"])) for r in rows]
        front = pareto_front(xy)
        for i, r in enumerate(rows):
            on = i in front
            ax.scatter(*xy[i], marker=MARKERS.get(base_of(r["model"]), "x"), s=16 if on else 10,
                       facecolors=COLORS[device] if r["quant"].startswith("int8") else "none",
                       edgecolors=COLORS[device], linewidths=0.8, alpha=1.0 if on else 0.45,
                       zorder=3)
        fx, fy = zip(*(xy[i] for i in front), strict=True)
        ax.step(fx, fy, where="post", color=COLORS[device], linewidth=0.8, zorder=2)
        for k, i in enumerate(front):
            ax.annotate(label(rows[i]), xy[i], xytext=(4, -9 if k % 2 else 3),
                        textcoords="offset points", fontsize=5.5)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.set_xlabel(f"Median latency per word, batch {batch} (ms)")
        ax.set_title(DEVICE_NAMES.get(device, device))
        ax.grid(True, which="major", linewidth=0.3, alpha=0.5)
    axes[0][0].set_ylabel("Exact-match accuracy, test (%)")
    # Legend: marker = model family, fill = int8.
    handles = [plt.Line2D([], [], marker=m, linestyle="", color="0.3", markerfacecolor="none",
                          markersize=4, label=n)
               for m, n in (("o", "LSTM seq2seq"), ("s", "Edit classifier"), ("D", "Char-CNN"))]
    handles.append(plt.Line2D([], [], marker="s", linestyle="", color="0.3", markersize=4,
                              label="filled: int8"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    paths = save(fig, out_dir, "pareto")
    plt.close(fig)
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", default="1", choices=["1", "64"])
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    for p in main(args.out, args.batch):
        print(p)
