import pytest

from tel.config import config_from_dict
from tel.export import export, verify
from tel.runtime.backends import load_lemmatizer
from tests.test_train import _toy_data


@pytest.mark.parametrize("runtime", ["onnxruntime", "ncnn"])
@pytest.mark.parametrize("model_type",
                         ["lstm_seq2seq", "char_cnn", "transformer", "edit_classifier"])
def test_export_matches_torch(tmp_path, model_type, runtime):
    _toy_data(tmp_path)
    cfg = config_from_dict({
        "name": "exp",
        "model": {"type": model_type, "emb": 16, "hidden": 32, "channels": 32, "layers": 2},
        "data": {"dir": str(tmp_path), "max_len": 24},
        "train": {"epochs": 2, "batch_size": 8, "seed": 1},
    })
    from tel.train import train

    run = tmp_path / "run"
    train(cfg, run, verbose=False)
    out = run / "export" / runtime
    manifest = export(run, runtime, out)
    assert manifest["params"] > 0 and manifest["runtime"] == runtime
    assert verify(run, out)["agreement"] == 1.0
    # Words longer than max_len and empty input must not crash.
    lem = load_lemmatizer(out)
    assert lem.lemmatize([]) == []
    assert len(lem.lemmatize(["மரங்கள்" * 10, "abc"])) == 2


def test_ncnn_fp16_export_and_reference(tmp_path):
    _toy_data(tmp_path)
    cfg = config_from_dict({
        "name": "fp16",
        "model": {"type": "edit_classifier", "emb": 16, "channels": 32, "layers": 1},
        "data": {"dir": str(tmp_path), "max_len": 24},
        "train": {"epochs": 2, "batch_size": 8, "seed": 1},
    })
    from tel.train import train

    run = tmp_path / "run"
    train(cfg, run, verbose=False)
    fp32 = export(run, "ncnn", run / "export" / "ncnn")
    fp16 = export(run, "ncnn", run / "export" / "ncnn-fp16", precision="fp16")
    assert (fp32["precision"], fp16["precision"]) == ("fp32", "fp16")
    size = {p: sum(f.stat().st_size for f in (run / "export" / p).glob("*.bin"))
            for p in ("ncnn", "ncnn-fp16")}
    assert size["ncnn-fp16"] < 0.6 * size["ncnn"]  # fp16 weights: about half
    assert verify(run, run / "export" / "ncnn-fp16")["agreement"] >= 0.9
    ref = (run / "export" / "ncnn-fp16" / "reference.tsv").read_text(encoding="utf-8")
    assert len(ref.splitlines()) == 20  # one line per test word


def test_fp16_rejected_for_other_runtimes(tmp_path):
    with pytest.raises(ValueError):
        export(tmp_path, "onnxruntime", tmp_path / "x", precision="fp16")


def test_fp16_constant_guard():
    from tel.export import fp16_unsafe_constants

    param = ("7767517\n2 2\n"
             "Input  in0  0 1 in0\n"
             "Eltwise weighted_sum_0 2 1 a b out0 0=1 -23301=2,1.0,1.000000e9\n")
    assert fp16_unsafe_constants(param) == [1e9]
    assert fp16_unsafe_constants(param.replace("1.000000e9", "1.000000e4")) == []
    clip_line = "Clip clip_0 1 1 x y 0=1.0 1=3.40282347e+38\n"  # FLT_MAX = unbounded
    clip = param.replace("1.000000e9", "1.000000e4") + clip_line
    assert fp16_unsafe_constants(clip) == []
    sizes = "InnerProduct linear_4 1 1 x y 0=703 1=1 2=179968\n"  # integer weight size
    assert fp16_unsafe_constants(clip + sizes) == []
