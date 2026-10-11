import json

import pytest

from tel.config import config_from_dict
from tel.data import Pair, write_tsv
from tel.train import train

STEMS = ["மரம்", "நாய்", "பசு", "வீடு", "கல்", "பூ", "மாடு", "ஊர்", "பால்", "தேன்"]


def _toy_data(d):  # also used by test_export
    pairs = [Pair(s + "கள்", s, "NOUN") for s in STEMS] + [Pair(s, s, "NOUN") for s in STEMS]
    for name in ("train", "dev", "test", "test_unseen"):
        write_tsv(d / f"{name}.tsv", pairs)


@pytest.mark.parametrize("model_type",
                         ["lstm_seq2seq", "char_cnn", "transformer", "edit_classifier"])
def test_train_smoke(tmp_path, model_type):
    _toy_data(tmp_path)
    cfg = config_from_dict({
        "name": "smoke",
        "model": {"type": model_type, "emb": 16, "hidden": 32, "channels": 32, "layers": 1},
        "data": {"dir": str(tmp_path)},
        "train": {"epochs": 2, "batch_size": 8, "seed": 1, "label_smoothing": 0.1},
        "decode": {"beam": 3},
    })
    m1 = train(cfg, tmp_path / "run1", verbose=False)
    m2 = train(cfg, tmp_path / "run2", verbose=False)
    assert m1["test"] == m2["test"]  # same seed, same result
    for f in ("model.pt", "tokenizer.json", "config.yaml", "metrics.json"):
        assert (tmp_path / "run1" / f).exists()
    assert json.loads((tmp_path / "run1" / "metrics.json").read_text())["params"] > 0


def test_unknown_config_key_rejected():
    with pytest.raises(ValueError):
        config_from_dict({"name": "x", "model": {"hiden": 3}})
