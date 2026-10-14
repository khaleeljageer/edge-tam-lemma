import pytest

from tel.config import config_from_dict
from tel.export import export, verify
from tests.test_train import _toy_data


def _int8_weights(path) -> int:
    import onnx

    model = onnx.load(str(path))
    return sum(i.data_type == onnx.TensorProto.INT8 for i in model.graph.initializer)


@pytest.mark.parametrize("method", ["dynamic", "static"])
@pytest.mark.parametrize("model_type", ["lstm_seq2seq", "edit_classifier"])
def test_onnx_int8(tmp_path, model_type, method):
    from tel.quantize import quantize_export
    from tel.train import train

    _toy_data(tmp_path)
    cfg = config_from_dict({
        "name": "q",
        "model": {"type": model_type, "emb": 16, "hidden": 32, "channels": 32, "layers": 1},
        "data": {"dir": str(tmp_path), "max_len": 24},
        "train": {"epochs": 2, "batch_size": 8, "seed": 1},
    })
    run = tmp_path / "run"
    train(cfg, run, verbose=False)
    export(run, "onnxruntime", run / "export" / "onnx")
    out = quantize_export(run / "export" / "onnx", method, calib_words=32)
    assert out.name == {"dynamic": "onnx-int8dyn", "static": "onnx-int8"}[method]
    manifest = (out / "manifest.json").read_text()
    assert '"precision": "int8' in manifest
    assert (out / "reference.tsv").exists()  # fp32 PyTorch reference kept
    assert sum(_int8_weights(f) for f in out.glob("*.onnx")) > 0
    # Toy 2-epoch models have near-tied logits, so int8 flips a few of the 20 dev
    # words; trained models agree 98-100% (bench/results/m4_quant/).
    assert verify(run, out)["agreement"] >= 0.6
    with pytest.raises(SystemExit):  # an int8 export is not re-quantized
        quantize_export(out, method)
