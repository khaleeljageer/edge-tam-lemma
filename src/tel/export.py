"""Export a trained run to an edge runtime and verify it against PyTorch.

    uv run python -m tel.export runs/edit_tiny                       # ONNX
    uv run --project envs/executorch python -m tel.export runs/edit_tiny --runtime executorch
    uv run --project envs/litert python -m tel.export runs/edit_tiny --runtime litert
    uv run python -m tel.export runs/edit_tiny --runtime ncnn

Writes runs/<name>/export/<runtime>/{manifest.json, tokenizer.json,
[scripts.txt], graph files, report.json}. Graphs take source ids padded to
exactly max_len. The padding mask (and reversal / end positions) are computed
inside the graph, except for ncnn, where the host passes them in because pnnx
cannot convert those ops. ONNX graphs have a dynamic batch dimension;
ExecuTorch and LiteRT graphs are exported for fixed batch sizes (default 1 and
64); ncnn graphs are batch 1. The host pads up to the graph's batch size.

The seq2seq model is exported as two graphs (encoder, decode_step); the greedy
loop runs in host code (`tel.runtime.host`), shared by all runtimes.

ExecuTorch and LiteRT pin their own torch versions, so they run from the
separate uv projects in envs/.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import tempfile
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from tel.config import config_from_dict
from tel.data import read_tsv
from tel.edit_tree import ScriptVocab
from tel.evaluate import score
from tel.inference import EditLemmatizer, Seq2SeqLemmatizer
from tel.lemmatizer import Lemmatizer
from tel.models.common import FP16_MAX, count_params
from tel.models.edit_classifier import EditClassifier
from tel.models.lstm_seq2seq import LSTMSeq2Seq, reversal_matrix
from tel.models.transformer import CharTransformer
from tel.runtime.backends import (
    RUNTIME_PACKAGE,
    deploy_size_bytes,
    graph_files,
    load_lemmatizer,
)
from tel.runtime.host import encode_fixed, end_positions
from tel.tokenizer import BOS_ID, PAD_ID, CharTokenizer
from tel.train import build_model

ONNX_OPSET = 18
SUFFIX = {"onnxruntime": "onnx", "executorch": "pte", "litert": "tflite", "ncnn": "ncnn"}
GRAPH_EXT = {"onnxruntime": "onnx", "executorch": "pte", "litert": "tflite",
             "ncnn": "ncnn.param"}


# --- export wrappers: plain-tensor signatures, mask computed in-graph ------------


class EditGraph(nn.Module):
    """Edit classifier with in-graph features: (src) -> logits."""

    def __init__(self, model: EditClassifier) -> None:
        super().__init__()
        self.model = model

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        return self.model(src, src != PAD_ID)


class EditFeaturesGraph(nn.Module):
    """Edit classifier with host features (ncnn): (src, mask, rpos) -> logits."""

    def __init__(self, model: EditClassifier) -> None:
        super().__init__()
        self.model = model

    def forward(self, src: torch.Tensor, mask: torch.Tensor, rpos: torch.Tensor) -> torch.Tensor:
        return self.model.forward_features(src, mask, rpos)


class EncoderGraph(nn.Module):
    """Encoder with in-graph features: (src) -> memory, h0."""

    def __init__(self, model: LSTMSeq2Seq) -> None:
        super().__init__()
        self.model = model

    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = (src != PAD_ID).to(torch.float32)
        return self.model.encode_features(src, mask, reversal_matrix(mask))


class EncoderFeaturesGraph(nn.Module):
    """Encoder with host features (ncnn): (src, mask, perm) -> memory, h0."""

    def __init__(self, model: LSTMSeq2Seq) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, src: torch.Tensor, mask: torch.Tensor, perm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.encode_features(src, mask, perm)


class CNNEncoderFeaturesGraph(nn.Module):
    """CNN encoder with host features (ncnn): (src, mask) -> memory, h0 (no perm)."""

    def __init__(self, model: LSTMSeq2Seq) -> None:
        super().__init__()
        self.model = model

    def forward(self, src: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.encode_features(src, mask)


class DecodeStepGraph(nn.Module):
    """(prev (B,1), attn, h, c, memory, mask) -> logits, attn, h, c. Same for all runtimes."""

    def __init__(self, model: LSTMSeq2Seq) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, prev: torch.Tensor, attn_prev: torch.Tensor, h: torch.Tensor, c: torch.Tensor,
        memory: torch.Tensor, mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, attn_h, (h, c) = self.model.decode_step(prev, attn_prev, (h, c), memory, mask)
        return logits, attn_h, h, c


class TransformerEncoderGraph(nn.Module):
    """Transformer encoder, in-graph mask: (src) -> memory."""

    def __init__(self, model: CharTransformer) -> None:
        super().__init__()
        self.model = model

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        return self.model.encode_features(src, (src != PAD_ID).to(torch.float32))


class TransformerEncoderFeaturesGraph(nn.Module):
    """Transformer encoder, host mask (ncnn): (src, mask) -> memory."""

    def __init__(self, model: CharTransformer) -> None:
        super().__init__()
        self.model = model

    def forward(self, src: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.model.encode_features(src, mask)


class TransformerDecodeGraph(nn.Module):
    """(prefix (B,T), memory, mask, sel (B,T)) -> logits. Same for all runtimes."""

    def __init__(self, model: CharTransformer) -> None:
        super().__init__()
        self.model = model

    def forward(self, prefix: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor,
                sel: torch.Tensor) -> torch.Tensor:
        return self.model.decode_step_fixed(prefix, memory, mask, sel)


# --- loading a run ------------------------------------------------------------------


def load_run(run_dir: Path) -> tuple[nn.Module, str, dict[str, Any], CharTokenizer,
                                     ScriptVocab | None, Any]:
    ckpt = torch.load(run_dir / "model.pt", weights_only=True)
    model = build_model(ckpt["type"], ckpt["kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    tok = CharTokenizer.load(run_dir / "tokenizer.json")
    scripts = run_dir / "scripts.txt"
    vocab = ScriptVocab.load(str(scripts)) if scripts.exists() else None
    cfg = config_from_dict(yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")))
    return model, ckpt["type"], ckpt["kwargs"], tok, vocab, cfg


def torch_lemmatizer(model: nn.Module, model_type: str, tok: CharTokenizer,
                     vocab: ScriptVocab | None, max_len: int) -> Lemmatizer:
    if model_type == "edit_classifier":
        return EditLemmatizer(model, tok, vocab, max_len)
    return Seq2SeqLemmatizer(model, tok, max_len)


# --- graphs and sample inputs -----------------------------------------------------------

@dataclass
class GraphSpec:
    module: nn.Module
    args: tuple
    inputs: list[tuple[str, int]]  # (name, batch axis)
    outputs: list[tuple[str, int]]


def graph_specs(model: nn.Module, model_type: str, tok: CharTokenizer, max_len: int,
                batch: int, host_features: bool) -> dict[str, GraphSpec]:
    words = (["தமிழ்", "மரங்கள்", "சென்றார்கள்"] * batch)[:batch]
    src = torch.tensor(encode_fixed(tok, words, max_len))
    mask = (src != PAD_ID).to(torch.float32)
    if model_type == "edit_classifier":
        if host_features:
            rpos = torch.tensor(end_positions(src.numpy(), max_len))
            return {"model": GraphSpec(EditFeaturesGraph(model), (src, mask, rpos),
                                       [("src", 0), ("mask", 0), ("rpos", 0)],
                                       [("logits", 0)])}
        return {"model": GraphSpec(EditGraph(model), (src,), [("src", 0)], [("logits", 0)])}

    if isinstance(model, CharTransformer):
        if host_features:
            encoder = GraphSpec(TransformerEncoderFeaturesGraph(model), (src, mask),
                                [("src", 0), ("mask", 0)], [("memory", 0)])
        else:
            encoder = GraphSpec(TransformerEncoderGraph(model), (src,), [("src", 0)],
                                [("memory", 0)])
        with torch.no_grad():
            memory = model.encode_features(src, mask)
        prefix = torch.full((batch, model.prefix_len), PAD_ID, dtype=torch.long)
        prefix[:, 0] = BOS_ID
        sel = torch.zeros(batch, model.prefix_len)
        sel[:, 0] = 1.0
        decode_step = GraphSpec(
            TransformerDecodeGraph(model), (prefix, memory, mask, sel),
            [("prefix", 0), ("memory", 0), ("mask", 0), ("sel", 0)], [("logits", 0)],
        )
        return {"encoder": encoder, "decode_step": decode_step}

    if host_features and model.needs_perm:
        encoder = GraphSpec(EncoderFeaturesGraph(model), (src, mask, reversal_matrix(mask)),
                            [("src", 0), ("mask", 0), ("perm", 0)],
                            [("memory", 0), ("h0", 0)])
    elif host_features:
        encoder = GraphSpec(CNNEncoderFeaturesGraph(model), (src, mask),
                            [("src", 0), ("mask", 0)], [("memory", 0), ("h0", 0)])
    else:
        encoder = GraphSpec(EncoderGraph(model), (src,), [("src", 0)],
                            [("memory", 0), ("h0", 0)])
    with torch.no_grad():
        memory, (h, c) = model.encode(src, src != PAD_ID)
    prev = torch.full((batch, 1), BOS_ID, dtype=torch.long)
    attn = torch.zeros(batch, model.hidden)
    decode_step = GraphSpec(
        DecodeStepGraph(model), (prev, attn, h, c, memory, mask),
        [("prev", 0), ("attn_prev", 0), ("h", 1), ("c", 1), ("memory", 0), ("mask", 0)],
        [("logits", 0), ("attn_h", 0), ("h_out", 1), ("c_out", 1)],
    )
    return {"encoder": encoder, "decode_step": decode_step}


# --- per-runtime exporters --------------------------------------------------------------


def _export_onnx(spec: GraphSpec, path: Path) -> None:
    batch = torch.export.Dim("batch", min=1, max=4096)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        program = torch.onnx.export(
            spec.module, spec.args, dynamo=True, opset_version=ONNX_OPSET,
            input_names=[n for n, _ in spec.inputs], output_names=[n for n, _ in spec.outputs],
            dynamic_shapes=tuple({ax: batch} for _, ax in spec.inputs), verbose=False,
        )
    program.save(str(path))


def _export_executorch(spec: GraphSpec, path: Path) -> None:
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower

    program = to_edge_transform_and_lower(
        torch.export.export(spec.module, spec.args), partitioner=[XnnpackPartitioner()]
    ).to_executorch()
    path.write_bytes(program.buffer)


def _export_litert(spec: GraphSpec, path: Path) -> None:
    import litert_torch

    litert_torch.convert(spec.module, spec.args).export(str(path))


def _export_ncnn(spec: GraphSpec, path: Path, precision: str = "fp32") -> None:
    """PyTorch -> pnnx -> ncnn (.param + .bin). `path` ends in .param.

    precision "fp32" stores fp32 weights; "fp16" stores fp16 weights (pnnx's
    default), and the runtime then also uses fp16 arithmetic where the CPU
    supports it (ARMv8.2+, e.g. Cortex-A76)."""
    import pnnx

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "graph"
        log = Path(tmp) / "pnnx.log"
        with open(log, "w") as f:  # pnnx prints a lot from native code; capture it
            saved = os.dup(1), os.dup(2)
            os.dup2(f.fileno(), 1)
            os.dup2(f.fileno(), 2)
            try:
                pnnx.export(spec.module, str(base) + ".pt", spec.args, fp16=precision == "fp16")
            finally:
                os.dup2(saved[0], 1)
                os.dup2(saved[1], 2)
        param, binf = Path(f"{base}.ncnn.param"), Path(f"{base}.ncnn.bin")
        if not param.exists():
            raise RuntimeError(f"pnnx failed for {path.name}:\n{log.read_text()[-2000:]}")
        unsupported = sorted({line.split()[0] for line in param.read_text().splitlines()[2:]
                              if "." in line.split()[0]})
        if unsupported:  # pnnx leaves torch ops it cannot map as e.g. "torch.gather"
            raise RuntimeError(f"ncnn has no layer for {unsupported} in {path.name}")
        if precision == "fp16":
            too_big = fp16_unsafe_constants(param.read_text())
            if too_big:
                raise RuntimeError(f"{path.name}: constants outside the fp16 range {too_big}; "
                                   "they become inf (and 0*inf = NaN) in fp16 arithmetic")
        shutil.copy(param, path)
        shutil.copy(binf, path.with_suffix(".bin"))


def fp16_unsafe_constants(param_text: str) -> list[float]:
    """Numeric layer parameters in an ncnn .param file that overflow fp16."""
    found = []
    for line in param_text.splitlines()[2:]:
        for token in line.split()[4:]:
            for value in token.split("=", 1)[-1].split(","):
                # Only float-typed params: ncnn writes floats with "." or an exponent;
                # plain integers are counts/sizes (e.g. weight_data_size), not constants.
                if not any(ch in value for ch in ".eE"):
                    continue
                try:
                    x = float(value)
                except ValueError:
                    continue
                # +-FLT_MAX is ncnn's "unbounded" sentinel (e.g. Clip max); it maps to
                # +-inf in fp16, which is harmless as a bound.
                if FP16_MAX < abs(x) < 3.4e38:
                    found.append(x)
    return sorted(set(found))


EXPORTERS = {"onnxruntime": _export_onnx, "executorch": _export_executorch,
             "litert": _export_litert, "ncnn": _export_ncnn}


def write_reference(run_dir: Path, out_dir: Path, split: str = "test") -> None:
    """Save PyTorch's predictions for a split (reference.tsv), so a device can report
    word-level agreement with PyTorch without having torch installed."""
    model, model_type, _, tok, vocab, cfg = load_run(run_dir)
    path = Path(cfg.data.dir) / f"{split}.tsv"
    if not path.exists():
        return
    words = [p.word for p in read_tsv(path)]
    preds = torch_lemmatizer(model, model_type, tok, vocab, cfg.data.max_len).lemmatize(words)
    (out_dir / "reference.tsv").write_text(
        "".join(f"{w}\t{p}\n" for w, p in zip(words, preds, strict=True)), encoding="utf-8"
    )


def export(run_dir: Path, runtime: str, out_dir: Path,
           batches: Sequence[int] = (1, 64), precision: str = "fp32") -> dict[str, Any]:
    if precision != "fp32" and runtime != "ncnn":
        raise ValueError("only ncnn supports --precision fp16 here; INT8/fp16 for other "
                         "runtimes belong to tel.quantize")
    model, model_type, kwargs, tok, vocab, cfg = load_run(run_dir)
    max_len = cfg.data.max_len
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    shutil.copy(run_dir / "tokenizer.json", out_dir / "tokenizer.json")
    if vocab is not None:
        shutil.copy(run_dir / "scripts.txt", out_dir / "scripts.txt")

    dynamic = runtime == "onnxruntime"
    host_features = runtime == "ncnn"
    if runtime == "ncnn":
        batches = [1]  # ncnn graphs drop the batch axis
    graphs: dict[str, dict[str, str]] = {}
    io: dict[str, dict[str, list]] = {}
    ext = GRAPH_EXT[runtime]
    for batch in [2] if dynamic else batches:
        key = "dynamic" if dynamic else str(batch)
        specs = graph_specs(model, model_type, tok, max_len, batch, host_features)
        for name, spec in specs.items():
            fname = f"{name}.{ext}" if dynamic else f"{name}_b{batch}.{ext}"
            if runtime == "ncnn":
                _export_ncnn(spec, out_dir / fname, precision)
            else:
                EXPORTERS[runtime](spec, out_dir / fname)
            graphs.setdefault(name, {})[key] = fname
            io[name] = {"inputs": spec.inputs, "outputs": spec.outputs}

    manifest: dict[str, Any] = {
        "name": cfg.name, "type": model_type, "runtime": runtime,
        "runtime_version": importlib.metadata.version(RUNTIME_PACKAGE[runtime]),
        "torch_version": torch.__version__, "params": count_params(model),
        "seed": cfg.train.seed, "max_len": max_len, "kwargs": kwargs, "precision": precision,
        "host_features": host_features, "graphs": graphs, "io": io,
    }
    if runtime == "onnxruntime":
        manifest["opset"] = ONNX_OPSET
    if isinstance(model, CharTransformer):
        manifest["prefix_len"] = model.prefix_len
    if isinstance(model, LSTMSeq2Seq):  # lstm_seq2seq and char_cnn
        manifest["hidden"] = model.hidden
        manifest["layers"] = model.layers
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    write_reference(run_dir, out_dir)
    return manifest


# --- verification -------------------------------------------------------------------


def verify(run_dir: Path, out_dir: Path, split: str = "dev") -> dict[str, Any]:
    """Compare exported vs PyTorch predictions on a split; write report.json."""
    model, model_type, _, tok, vocab, cfg = load_run(run_dir)
    pairs = read_tsv(Path(cfg.data.dir) / f"{split}.tsv")
    words = [p.word for p in pairs]
    ref = torch_lemmatizer(model, model_type, tok, vocab, cfg.data.max_len).lemmatize(words)
    got: list[str] = []
    lem = load_lemmatizer(out_dir)
    for i in range(0, len(words), 64):
        got.extend(lem.lemmatize(words[i : i + 64]))
    agree = float(np.mean([a == b for a, b in zip(ref, got, strict=True)]))
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    files = {f: (out_dir / f).stat().st_size for f in graph_files(manifest)}
    report = {
        "run": run_dir.name, "runtime": manifest["runtime"],
        "runtime_version": manifest["runtime_version"], "split": split, "n": len(words),
        "agreement": agree, "acc_torch": score(pairs, ref).overall.acc,
        "acc_export": score(pairs, got).overall.acc,
        "deploy_size_kb": round(deploy_size_bytes(out_dir, manifest) / 1024, 1),
        "files_kb": {k: round(v / 1024, 1) for k, v in files.items()},
        "mismatches": [(w, a, b) for w, a, b in zip(words, ref, got, strict=True) if a != b][:20],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False),
                                         encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--runtime", choices=sorted(EXPORTERS), default="onnxruntime")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 64],
                    help="fixed batch sizes (ExecuTorch, LiteRT; ncnn is always 1)")
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp32",
                    help="ncnn only: fp16 weights + fp16 arithmetic on capable CPUs")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--min-agreement", type=float, default=0.995)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    suffix = SUFFIX[args.runtime] + ("" if args.precision == "fp32" else f"-{args.precision}")
    out_dir = run_dir / "export" / suffix
    export(run_dir, args.runtime, out_dir, args.batches, args.precision)
    report = verify(run_dir, out_dir, args.split)
    print(json.dumps({k: v for k, v in report.items() if k != "mismatches"}, indent=1))
    if report["agreement"] < args.min_agreement:
        raise SystemExit(f"agreement {report['agreement']:.4f} below {args.min_agreement}")


if __name__ == "__main__":
    main()
