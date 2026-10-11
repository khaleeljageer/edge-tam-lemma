"""Load exported graphs for each runtime and wrap them as host lemmatizers.

Each runtime is imported lazily, so a device only needs the one it benchmarks:
onnxruntime, ncnn, ai-edge-litert, or executorch (its Python runtime also needs
torch).

Loaders take (path, threads, io), where io = {"inputs": [[name, batch_axis], ...],
"outputs": [...]} from the manifest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tel.edit_tree import ScriptVocab
from tel.runtime.host import EditHost, Graph, Graphs, Seq2SeqHost, TransformerHost
from tel.tokenizer import CharTokenizer

IO = dict[str, list[list[Any]]]


def _onnx_graph(path: Path, threads: int | None, io: IO, **_: Any) -> Graph:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    return lambda xs: sess.run(None, dict(zip(names, xs, strict=True)))


def _executorch_graph(path: Path, threads: int | None, io: IO, **_: Any) -> Graph:
    import torch
    from executorch.runtime import Runtime

    if threads:
        torch.set_num_threads(threads)
    method = Runtime.get().load_program(str(path)).load_method("forward")
    return lambda xs: [o.numpy() for o in method.execute([torch.from_numpy(x) for x in xs])]


def _litert_graph(path: Path, threads: int | None, io: IO, **_: Any) -> Graph:
    from ai_edge_litert.interpreter import Interpreter

    interp = Interpreter(model_path=str(path), num_threads=threads or None)
    runner = interp.get_signature_runner()
    in_names = sorted(runner.get_input_details(), key=lambda n: int(n.rsplit("_", 1)[-1]))
    out_names = sorted(runner.get_output_details(), key=lambda n: int(n.rsplit("_", 1)[-1]))

    def run(xs: list[np.ndarray]) -> list[np.ndarray]:
        out = runner(**dict(zip(in_names, xs, strict=True)))
        return [out[n] for n in out_names]

    return run


def _ncnn_graph(path: Path, threads: int | None, io: IO, precision: str = "fp32",
                **_: Any) -> Graph:
    """ncnn graphs are batch-1 and drop the batch axis; ids are int32.

    fp32: all reduced-precision paths off, to match the other runtimes exactly.
    fp16: ncnn's defaults, i.e. fp16 storage/arithmetic wherever the CPU has it.
    """
    import ncnn

    net = ncnn.Net()
    net.opt.num_threads = threads or ncnn.get_big_cpu_count()
    if precision == "fp32":
        for flag in ("use_fp16_packed", "use_fp16_storage", "use_fp16_arithmetic",
                     "use_bf16_storage"):
            setattr(net.opt, flag, False)
    if net.load_param(str(path)) or net.load_model(str(path.with_suffix(".bin"))):
        raise RuntimeError(f"ncnn could not load {path}")
    in_axes = [ax for _, ax in io["inputs"]]
    out_axes = [ax for _, ax in io["outputs"]]

    def run(xs: list[np.ndarray]) -> list[np.ndarray]:
        ex = net.create_extractor()
        for i, (x, ax) in enumerate(zip(xs, in_axes, strict=True)):
            x = np.squeeze(x, axis=ax)
            x = x.astype(np.int32) if x.dtype == np.int64 else x.astype(np.float32)
            ex.input(f"in{i}", ncnn.Mat(np.ascontiguousarray(x)).clone())
        outs = []
        for i, ax in enumerate(out_axes):
            ret, mat = ex.extract(f"out{i}")
            if ret != 0:
                raise RuntimeError(f"ncnn extract out{i} failed ({ret}) in {path.name}")
            outs.append(np.expand_dims(np.array(mat, dtype=np.float32), ax))
        return outs

    return run


# Distribution name of each runtime, for recording versions.
RUNTIME_PACKAGE = {"onnxruntime": "onnxruntime", "executorch": "executorch",
                   "litert": "ai-edge-litert", "ncnn": "ncnn"}

LOADERS = {"onnxruntime": _onnx_graph, "executorch": _executorch_graph,
           "litert": _litert_graph, "ncnn": _ncnn_graph}


def load_manifest(export_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(export_dir) / "manifest.json").read_text(encoding="utf-8"))


def files_of(graph_file: str) -> list[str]:
    """Files that make up one graph (ncnn: .param + .bin)."""
    if graph_file.endswith(".param"):
        return [graph_file, graph_file.removesuffix(".param") + ".bin"]
    return [graph_file]


def graph_files(manifest: dict[str, Any]) -> list[str]:
    return [p for by_batch in manifest["graphs"].values() for f in by_batch.values()
            for p in files_of(f)]


def deploy_size_bytes(export_dir: str | Path, manifest: dict[str, Any]) -> int:
    """Size of one deployable graph set: the dynamic or smallest-batch variant."""
    total = 0
    for by_batch in manifest["graphs"].values():
        key = "dynamic" if "dynamic" in by_batch else min(by_batch, key=int)
        total += sum((Path(export_dir) / p).stat().st_size for p in files_of(by_batch[key]))
    return total


def load_lemmatizer(export_dir: str | Path, threads: int | None = None):
    """Load an export directory written by `tel.export` as a Lemmatizer."""
    export_dir = Path(export_dir)
    manifest = load_manifest(export_dir)
    loader = LOADERS[manifest["runtime"]]

    def graphs(name: str) -> Graphs:
        io = manifest["io"][name]
        precision = manifest.get("precision", "fp32")
        return {(None if k == "dynamic" else int(k)):
                loader(export_dir / f, threads, io, precision=precision)
                for k, f in manifest["graphs"][name].items()}

    tok = CharTokenizer.load(export_dir / "tokenizer.json")

    def input_names(name: str) -> list[str]:
        return [n for n, _ in manifest["io"][name]["inputs"]]

    if manifest["type"] == "edit_classifier":
        vocab = ScriptVocab.load(str(export_dir / "scripts.txt"))
        return EditHost(tok, vocab, manifest["max_len"], graphs("model"), input_names("model"))
    if manifest["type"] == "transformer":
        return TransformerHost(tok, manifest["max_len"], manifest["prefix_len"],
                               graphs("encoder"), graphs("decode_step"), input_names("encoder"))
    return Seq2SeqHost(tok, manifest["max_len"], manifest["hidden"], manifest["layers"],
                       graphs("encoder"), graphs("decode_step"), input_names("encoder"))
