"""Benchmark exported lemmatizers on one device.

Needs Python 3.10+, numpy, the repo's src/ on PYTHONPATH and the runtime of
each export directory: onnxruntime, ncnn, ai-edge-litert, or executorch
(+ torch). Torch is not needed for ONNX, ncnn or LiteRT, so those run as-is on a Raspberry Pi
or Android (Termux).

    PYTHONPATH=src python bench/bench_runner.py --device desktop \\
        --models runs/edit_tiny/export/onnx runs/edit_tiny/export/tflite

Protocol (research plan section 8.2): a fixed word list in fixed order, 200
warm-up words not measured, each configuration repeated --reps times and the
median taken, 1 thread and all cores, batch 1 and batch 64. Each (model,
threads) configuration runs in a fresh subprocess so peak RSS and thread
settings do not leak between configurations.

Output: one CSV row per (model, runtime, threads, batch) appended to
bench/results/<device>.csv. For batch 64, latency columns are per word
(batch time / batch size).

Hybrid lookup (tel.hybrid): --tables runs every model once per lookup table
(`none` = model only); misses go to the model. --weighted repeats each input
word by its corpus frequency (`freq` column) in a fixed shuffled order, so
accuracy, hit rate and timing are over running-text tokens instead of types.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

COLUMNS = [
    "device", "model", "params", "quant", "runtime", "threads", "batch", "input", "n_words",
    "acc", "acc_noun", "acc_verb", "acc_unseen", "lat_med_ms", "lat_p95_ms", "wps",
    "peak_rss_mb", "size_kb", "cold_start_ms", "seed", "git_sha", "runtime_version",
    "python", "machine", "timestamp", "agreement", "table", "table_rows", "hit_rate",
]

WARMUP = 200


def read_gold(path: Path, weighted: bool = False) -> list[tuple[str, str, str]]:
    """(word, lemma, pos) rows from a processed TSV, in file order; with `weighted`,
    as a running-text token stream (each row repeated by its `freq`)."""
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        iw, il, ip = header.index("word"), header.index("lemma"), header.index("pos")
        rows = [line.rstrip("\n").split("\t") for line in f if line.strip()]
    gold = [(r[iw], r[il], r[ip]) for r in rows]
    if not weighted:
        return gold
    from tel.hybrid import weighted_stream

    return weighted_stream(gold, [int(r[header.index("freq")]) for r in rows])


def percentile(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024  # bytes vs KiB


def git_sha() -> str:
    """Commit of the code being benchmarked; on a device, read from the bundle."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        stamp = Path(__file__).with_name("BUNDLE_VERSION")
        return stamp.read_text().strip() if stamp.exists() else "unknown"


# --- worker: runs inside a fresh subprocess --------------------------------------


def accuracy(lem, gold, train_lemmas: set[str],
             reference: dict[str, str] | None = None) -> dict[str, float | None]:
    """Accuracy vs gold; `agreement` = fraction of words where the device's output
    equals PyTorch's (from the export's reference.tsv), if available."""
    preds = []
    words = [w for w, _, _ in gold]
    for i in range(0, len(words), 64):
        preds.extend(lem.lemmatize(words[i : i + 64]))

    def acc(rows):
        rows = list(rows)
        return sum(p == g for p, (_, g, _) in rows) / len(rows) if rows else None

    pairs = list(zip(preds, gold, strict=True))
    return {
        "acc": acc(pairs),
        "acc_noun": acc(r for r in pairs if r[1][2] == "NOUN"),
        "acc_verb": acc(r for r in pairs if r[1][2] == "VERB"),
        "acc_unseen": acc(r for r in pairs if r[1][1] not in train_lemmas)
        if train_lemmas else None,
        "agreement": sum(p == reference.get(w) for p, w in zip(preds, words, strict=True))
        / len(words) if reference else None,
    }


def read_reference(export_dir: str) -> dict[str, str] | None:
    path = Path(export_dir) / "reference.tsv"
    if not path.exists():
        return None
    rows = (line.rstrip("\n").split("\t") for line in path.open(encoding="utf-8"))
    return {r[0]: r[1] if len(r) > 1 else "" for r in rows if r and r[0]}


def time_batches(lem, words: list[str], batch: int, reps: int) -> tuple[float, float, float]:
    """-> (median per-word latency ms, p95 per-word latency ms, words per second)."""
    for i in range(0, min(WARMUP, len(words)), batch):
        lem.lemmatize(words[i : i + batch])
    meds, p95s, wps = [], [], []
    for _ in range(reps):
        lat = []
        t_start = time.perf_counter()
        for i in range(0, len(words), batch):
            chunk = words[i : i + batch]
            t0 = time.perf_counter()
            lem.lemmatize(chunk)
            lat.append((time.perf_counter() - t0) * 1000 / len(chunk))
        wps.append(len(words) / (time.perf_counter() - t_start))
        meds.append(statistics.median(lat))
        p95s.append(percentile(lat, 0.95))
    return statistics.median(meds), statistics.median(p95s), statistics.median(wps)


def worker(args: argparse.Namespace) -> None:
    from importlib.metadata import version

    from tel.runtime.backends import RUNTIME_PACKAGE, load_lemmatizer, load_manifest

    manifest = load_manifest(args.model)
    lem = load_lemmatizer(args.model, threads=args.threads)
    if args.table:
        from tel.hybrid import HybridLemmatizer, read_table

        lem = HybridLemmatizer(read_table(args.table), lem)
    gold = read_gold(Path(args.input), args.weighted)
    train_lemmas = set()
    if args.train_lemmas and Path(args.train_lemmas).exists():
        train_lemmas = set(Path(args.train_lemmas).read_text(encoding="utf-8").split())
    words = [gold[i % len(gold)][0] for i in range(args.n)]

    reference = None if args.table else read_reference(args.model)
    result = {"runtime_version": version(RUNTIME_PACKAGE[manifest["runtime"]]),
              **accuracy(lem, gold, train_lemmas, reference),
              "hit_rate": getattr(lem, "hit_rate", None)}
    for batch in args.batches:
        result[f"batch{batch}"] = time_batches(lem, words, batch, args.reps)
    result["peak_rss_mb"] = peak_rss_mb()
    print(json.dumps(result))


def cold_start(args: argparse.Namespace) -> None:
    """Load the model and lemmatize one word; the parent times the whole process."""
    from tel.runtime.backends import load_lemmatizer

    lem = load_lemmatizer(args.model, threads=args.threads)
    if args.table:
        from tel.hybrid import HybridLemmatizer, read_table

        lem = HybridLemmatizer(read_table(args.table), lem)
    lem.lemmatize(["தமிழ்"])


# --- driver ---------------------------------------------------------------------------


def run_sub(mode: str, args: argparse.Namespace, model: str, threads: int) -> str:
    cmd = [sys.executable, __file__, f"--{mode}", "--model", model, "--threads", str(threads),
           "--input", args.input, "--n", str(args.n), "--reps", str(args.reps),
           "--train-lemmas", args.train_lemmas or "", "--batches", *map(str, args.batches)]
    if args.table:
        cmd += ["--table", args.table]
    if args.weighted:
        cmd.append("--weighted")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
    if proc.returncode != 0:
        raise RuntimeError(f"{mode} failed for {model}:\n{proc.stderr}")
    return proc.stdout


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", help="device name, used for the output file")
    ap.add_argument("--models", nargs="+", help="export directories (with manifest.json)")
    ap.add_argument("--input", default="data/processed/test.tsv")
    ap.add_argument("--train-lemmas", default="data/processed/splits/train.txt")
    ap.add_argument("--n", type=int, default=10_000, help="words timed (input is cycled)")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, os.cpu_count() or 1])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 64])
    ap.add_argument("--quant", default="fp32", help="label if the manifest has no precision")
    ap.add_argument("--out", help="CSV path (default bench/results/<device>.csv)")
    ap.add_argument("--tables", nargs="+", default=["none"],
                    help="hybrid lookup tables (tel.hybrid TSV); 'none' = model only")
    ap.add_argument("--weighted", action="store_true",
                    help="frequency-weighted token stream from the input's freq column")
    ap.add_argument("--table", help=argparse.SUPPRESS)
    ap.add_argument("--model", help=argparse.SUPPRESS)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--cold-start", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        args.threads = args.threads[0]
        return worker(args)
    if args.cold_start:
        args.threads = args.threads[0]
        return cold_start(args)
    if not args.device or not args.models:
        ap.error("--device and --models are required")

    from tel.runtime.backends import deploy_size_bytes

    out = Path(args.out or f"bench/results/{args.device}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists()
    if not new_file:
        with open(out, encoding="utf-8") as f:
            if f.readline().rstrip("\r\n").split(",") != COLUMNS:
                sys.exit(f"{out} has an older column layout; write to a new --out file")
    sha = git_sha()
    failures: list[str] = []
    with open(out, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        if new_file:
            writer.writeheader()
        for model in args.models:
            manifest = json.loads((Path(model) / "manifest.json").read_text(encoding="utf-8"))
            model_kb = deploy_size_bytes(model, manifest) / 1024
            for table, threads in ((t, n) for t in args.tables for n in sorted(set(args.threads))):
                args.table = None if table == "none" else table
                table_rows = 0
                if args.table:
                    with open(args.table, encoding="utf-8") as tf:
                        table_rows = sum(1 for _ in tf) - 1
                size_kb = model_kb + (Path(args.table).stat().st_size / 1024 if args.table else 0)
                label = f"{manifest['runtime']}/{manifest.get('precision', args.quant)}"
                print(f"[{args.device}] {manifest['name']} {label} threads={threads}"
                      f" table={table} ...", flush=True)
                try:
                    res = json.loads(run_sub("worker", args, model, threads))
                    colds = []
                    for _ in range(args.reps):
                        t0 = time.perf_counter()
                        run_sub("cold-start", args, model, threads)
                        colds.append((time.perf_counter() - t0) * 1000)
                except RuntimeError as e:  # keep going; report at the end
                    failures.append(f"{model} threads={threads} table={table}")
                    print(f"   FAILED, skipping:\n{e}", file=sys.stderr, flush=True)
                    continue
                for batch in args.batches:
                    med, p95, wps = res[f"batch{batch}"]
                    row = {
                        "device": args.device, "model": manifest["name"],
                        "params": manifest.get("params"),
                        "quant": manifest.get("precision", args.quant),
                        "runtime": manifest["runtime"], "threads": threads, "batch": batch,
                        "input": Path(args.input).name + ("+weighted" if args.weighted else ""),
                        "n_words": args.n,
                        "acc": res["acc"], "acc_noun": res["acc_noun"],
                        "acc_verb": res["acc_verb"], "acc_unseen": res["acc_unseen"],
                        "lat_med_ms": med, "lat_p95_ms": p95, "wps": wps,
                        "peak_rss_mb": res["peak_rss_mb"], "size_kb": size_kb,
                        "cold_start_ms": statistics.median(colds),
                        "seed": manifest.get("seed"), "git_sha": sha,
                        "runtime_version": res["runtime_version"],
                        "python": platform.python_version(), "machine": platform.machine(),
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "agreement": res.get("agreement"),
                        "table": Path(table).stem if args.table else "",
                        "table_rows": table_rows, "hit_rate": res.get("hit_rate"),
                    }
                    writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v
                                     for k, v in row.items()})
                    f.flush()
                    print(f"   batch={batch:<3} med={med:.3f} ms/word  p95={p95:.3f}  "
                          f"{wps:,.0f} w/s  rss={res['peak_rss_mb']:.0f} MB  "
                          f"cold={statistics.median(colds):.0f} ms  acc={res['acc']:.3f}"
                          + (f"  agree={res['agreement']:.4f}" if res.get("agreement") is not None
                             else "")
                          + (f"  hit={res['hit_rate']:.3f}" if res.get("hit_rate") is not None
                             else ""))
    print(f"wrote {out}")
    if failures:
        sys.exit(f"{len(failures)} configuration(s) failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
