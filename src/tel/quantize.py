"""Post-training INT8 quantization of exported graphs (ONNX Runtime, LiteRT).

    uv run python -m tel.quantize runs/edit_kd_500k_s13/export/onnx --method dynamic
    uv run python -m tel.quantize runs/edit_kd_500k_s13/export/onnx --method static
    uv run --project envs/litert python -m tel.quantize runs/edit_kd_500k_s13/export/tflite \\
        --method dynamic

Methods:
  dynamic  int8 weights; activations quantized on the fly at each call
           (ONNX: integer MatMul/Conv/LSTM kernels; LiteRT: dynamic-range,
           `dynamic_wi8_afp32`).
  static   ONNX only: int8 weights and activations with fixed scales (QDQ,
           per-channel weights, Conv/MatMul/Gemm only so the additive padding
           mask stays float; the LSTM op itself stays float). Scales are calibrated on the real
           inputs every graph sees while the fp32 export lemmatizes --calib-words
           training words (train.tsv in a fixed shuffled order), so the decoder
           step is calibrated on genuine recurrent states, not random tensors.

Reads an fp32 export directory written by tel.export and writes a sibling
directory (<export>-int8dyn or <export>-int8) with the quantized graphs, a
manifest with precision "int8-dyn" / "int8", and the same reference.tsv
(PyTorch fp32 predictions), so `agreement` in the benchmark measures exactly
what quantization changes. Then verifies on dev (report.json); no minimum
agreement is enforced, the drop is the result.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tel.config import config_from_dict
from tel.data import read_tsv
from tel.runtime.backends import LOADERS, files_of, graph_files, load_lemmatizer, load_manifest

PRECISION = {"dynamic": "int8-dyn", "static": "int8"}
SUFFIX = {"dynamic": "-int8dyn", "static": "-int8"}
STATIC_OPS = ["Conv", "MatMul", "Gemm"]


def calibration_words(data_dir: Path, n: int, seed: int = 0) -> list[str]:
    words = sorted({p.word for p in read_tsv(data_dir / "train.tsv")})
    random.Random(seed).shuffle(words)
    return words[:n]


def record_inputs(export_dir: Path, words: list[str]) -> dict[str, list[list[np.ndarray]]]:
    """Run the fp32 export over `words` (in batches of 64, then the first 64 one by
    one) and record every graph call's inputs, keyed by graph file name."""
    import tel.runtime.backends as backends

    manifest = load_manifest(export_dir)
    loader = LOADERS[manifest["runtime"]]
    calls: dict[str, list[list[np.ndarray]]] = {}

    def recording_loader(path: Path, threads, io, **opts):
        run = loader(path, threads, io, **opts)
        seen = calls.setdefault(path.name, [])

        def wrapped(xs):
            seen.append([np.array(x, copy=True) for x in xs])
            return run(xs)

        return wrapped

    backends.LOADERS[manifest["runtime"]] = recording_loader
    try:
        lem = load_lemmatizer(export_dir, threads=1)
        for i in range(0, len(words), 64):
            lem.lemmatize(words[i : i + 64])
        for w in words[:64]:  # single words too, so fixed batch-1 graphs see data
            lem.lemmatize([w])
    finally:
        backends.LOADERS[manifest["runtime"]] = loader
    return calls


# --- ONNX Runtime -------------------------------------------------------------------------


def _onnx_input_names(path: Path) -> list[str]:
    import onnx

    model = onnx.load(str(path))
    inits = {i.name for i in model.graph.initializer}
    return [i.name for i in model.graph.input if i.name not in inits]


def quantize_onnx(src: Path, dst: Path, method: str, samples: list[list[np.ndarray]]) -> None:
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_dynamic,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    if method == "dynamic":
        # Fold constants first: the exporter reorders LSTM gate weights with
        # Slice/Concat inside the graph, and the quantizer only touches weights that
        # are plain initializers. The dynamo exporter's stored value_info shapes are
        # rejected by onnx shape inference on reload; drop them (they are re-inferred).
        import onnx
        import onnxruntime as ort

        clean = dst.with_suffix(".clean.onnx")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        opts.optimized_model_filepath = str(clean)
        ort.InferenceSession(str(src), opts, providers=["CPUExecutionProvider"])
        model = onnx.load(str(clean))
        del model.graph.value_info[:]
        onnx.save(model, str(clean))
        try:
            quantize_dynamic(str(clean), str(dst), weight_type=QuantType.QInt8)
        finally:
            clean.unlink(missing_ok=True)
        return
    # Static: shape inference + graph optimizations first (recommended for QDQ).
    prepped = dst.with_suffix(".prep.onnx")
    quant_pre_process(str(src), str(prepped), skip_symbolic_shape=True)
    try:
        names = _onnx_input_names(prepped)

        class Reader(CalibrationDataReader):
            def __init__(self) -> None:
                self.it = iter(samples)

            def get_next(self) -> dict[str, np.ndarray] | None:
                xs = next(self.it, None)
                return None if xs is None else dict(zip(names, xs, strict=True))

        # Only weight-carrying ops get int8. Elementwise mask arithmetic stays float:
        # the additive padding mask ((m - 1) * MASK_NEG) spans ~[-1e4, x], and one
        # per-tensor int8 scale over that range would erase the real activations.
        quantize_static(str(prepped), str(dst), Reader(), quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                        per_channel=True, op_types_to_quantize=STATIC_OPS)
    finally:
        prepped.unlink(missing_ok=True)


# --- LiteRT ---------------------------------------------------------------------------------


def quantize_litert(src: Path, dst: Path, method: str, samples: list[list[np.ndarray]]) -> None:
    """Dynamic-range INT8 (weights int8, activations float at the op boundaries)."""
    from ai_edge_quantizer import quantizer, recipe

    qt = quantizer.Quantizer(str(src), recipe.dynamic_wi8_afp32())
    qt.quantize().export_model(str(dst), overwrite=True)


QUANTIZERS = {"onnxruntime": quantize_onnx, "litert": quantize_litert}


def quantize_export(export_dir: Path, method: str, calib_words: int = 512,
                    out_dir: Path | None = None) -> Path:
    manifest = load_manifest(export_dir)
    runtime = manifest["runtime"]
    if runtime == "litert" and method == "static":
        # Tried (ai_edge_quantizer static_wi8_ai8): it also quantizes the additive
        # padding-mask arithmetic (range ~[-1e4, x]; edit classifier agreement fell to
        # 15%) and turns float graph inputs into int8 ones (LSTM). Full-int8 LiteRT
        # needs a quantization-friendly mask in the models; it matters for NPUs, not
        # for the CPU targets here.
        raise SystemExit("static INT8 for LiteRT is not supported here; use --method "
                         "dynamic (see the comment in tel.quantize)")
    if runtime not in QUANTIZERS:
        raise SystemExit(f"no INT8 path for {runtime} here (ONNX Runtime and LiteRT only)")
    if manifest.get("precision", "fp32") != "fp32":
        raise SystemExit(f"{export_dir} is already {manifest['precision']}; "
                         "quantize an fp32 export")
    out_dir = out_dir or export_dir.with_name(export_dir.name + SUFFIX[method])
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(export_dir, out_dir, ignore=shutil.ignore_patterns("report.json"))

    samples: dict[str, list[list[np.ndarray]]] = {}
    if method == "static":
        run_cfg = yaml.safe_load((export_dir.parents[1] / "config.yaml").read_text("utf-8"))
        data_dir = Path(config_from_dict(run_cfg).data.dir)
        samples = record_inputs(export_dir, calibration_words(data_dir, calib_words))
    for f in graph_files(manifest):
        for part in files_of(f):
            QUANTIZERS[runtime](export_dir / part, out_dir / part, method, samples.get(part, []))

    manifest["precision"] = PRECISION[method]
    manifest["quantization"] = {"method": method, "source": str(export_dir),
                                **({"calib_words": calib_words} if method == "static" else {})}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export_dir", type=Path, help="fp32 export (runs/<name>/export/<runtime>)")
    ap.add_argument("--method", choices=sorted(PRECISION), default="dynamic")
    ap.add_argument("--calib-words", type=int, default=512)
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()

    from tel.export import verify

    out_dir = quantize_export(args.export_dir, args.method, args.calib_words)
    report: dict[str, Any] = verify(args.export_dir.parents[1], out_dir, args.split)
    print(json.dumps({k: v for k, v in report.items() if k != "mismatches"}, indent=1))


if __name__ == "__main__":
    main()
