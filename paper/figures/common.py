"""Shared loading and selection helpers for the figure and table scripts.

Every number shown in the paper is read from bench/results/ here; nothing is typed in.
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "bench" / "results"
OUT = Path(__file__).resolve().parent / "out"

# Device benchmark files for the Pareto figure, oldest first. When the same
# (device, model, quant, runtime, threads, batch) appears in several files the
# later file wins (e.g. m4_quant re-ran the distilled fp32 exports).
DEVICE_RUNS = [
    "m2_export_check/desktop.csv",
    "m2_export_check/rpi5.run3.csv",
    "m2_export_check/rpi5-fp16.csv",
    "m3_ablation/desktop.csv",
    "m3_ablation/rpi5.csv",
    "m3_distill/desktop.csv",
    "m3_distill/rpi5.csv",
    "m4_quant/desktop.csv",
    "m4_quant/rpi5.csv",
]

DEVICE_NAMES = {"rpi5": "Raspberry Pi 5", "desktop": "Desktop (x86)"}
RUNTIME_NAMES = {"onnxruntime": "ONNX", "litert": "LiteRT", "ncnn": "ncnn",
                 "executorch": "ExecuTorch"}
QUANT_NAMES = {"fp32": "fp32", "fp16": "fp16", "int8-dyn": "int8 dyn", "int8": "int8 static"}
BASE_NAMES = {"lstm": "LSTM", "edit": "Edit classifier", "transformer": "Transformer",
              "lstm_kd": "LSTM + KD", "edit_kd": "Edit + KD", "char_cnn": "Char-CNN"}


def read_csv(path: Path | str) -> list[dict[str, str]]:
    path = RESULTS / path if not Path(path).is_absolute() else Path(path)
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def short_model(model: str) -> str:
    """'edit_kd_500k_s13' -> 'edit_kd_500k' (seed dropped)."""
    head, _, seed = model.rpartition("_s")
    return head if seed.isdigit() else model


def device_rows(files: Iterable[str] = DEVICE_RUNS) -> list[dict[str, str]]:
    """Plain (no lookup table) device rows, later files overriding earlier ones."""
    keyed: dict[tuple, dict[str, str]] = {}
    for name in files:
        if not (RESULTS / name).exists():
            continue
        for r in read_csv(name):
            if r.get("table"):
                continue
            key = (r["device"], r["model"], r["quant"], r["runtime"], r["threads"], r["batch"])
            keyed[key] = r
    return list(keyed.values())


def best_by(rows: Iterable[dict[str, str]], key: tuple[str, ...],
            metric: str = "lat_med_ms") -> list[dict[str, str]]:
    """For each value of `key`, the row with the lowest `metric`."""
    best: dict[tuple, dict[str, str]] = {}
    for r in rows:
        k = tuple(r[c] for c in key)
        if k not in best or float(r[metric]) < float(best[k][metric]):
            best[k] = r
    return list(best.values())


def pareto_front(points: list[tuple[float, float]]) -> list[int]:
    """Indices of points not dominated (lower x and higher y are better), by x."""
    order = sorted(range(len(points)), key=lambda i: (points[i][0], -points[i][1]))
    front, best_y = [], float("-inf")
    for i in order:
        if points[i][1] > best_y:
            front.append(i)
            best_y = points[i][1]
    return front


def seed_stats(rows: Iterable[dict[str, str]], key: tuple[str, ...],
               value: str) -> dict[tuple, tuple[float, float, int]]:
    """Mean, sample std (0 for one seed) and count of `value` per `key`."""
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        groups[tuple(r[c] for c in key)].append(float(r[value]))
    return {k: (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0, len(v))
            for k, v in groups.items()}


def latex_table(path: Path, header: list[str], rows: list[list[str]], align: str,
                caption: str, label: str) -> None:
    """Write a booktabs table (needs \\usepackage{booktabs})."""
    def esc(s: str) -> str:
        return s.replace("%", r"\%").replace("&", r"\&").replace("_", r"\_")

    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             rf"\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(esc(h) for h in header) + r" \\", r"\midrule"]
    for row in rows:
        lines.append(r"\midrule" if row == ["--"] else " & ".join(esc(c) for c in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}",
              rf"\label{{{label}}}", r"\end{table}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                         "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "figure.dpi": 150, "savefig.bbox": "tight",
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    return plt


def save(fig, out_dir: Path, name: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.pdf", out_dir / f"{name}.png"]
    for p in paths:
        fig.savefig(p, metadata={"CreationDate": None} if p.suffix == ".pdf" else None)
    return paths
