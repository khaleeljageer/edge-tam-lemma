import csv
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "common", Path(__file__).parents[1] / "scripts/baselines/common.py"
)
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)


def test_write_rows_replaces_only_that_model(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "RESULTS", tmp_path / "acc.csv")
    common.write_rows("a", [{"model": "a", "split": "test", "acc": 0.5}])
    common.write_rows("b", [{"model": "b", "split": "test", "acc": 0.7}])
    common.write_rows("a", [{"model": "a", "split": "test", "acc": 0.6},
                            {"model": "a", "split": "dev", "acc": 0.4}])
    rows = list(csv.DictReader(open(tmp_path / "acc.csv", encoding="utf-8")))
    assert [(r["model"], r["split"], r["acc"]) for r in rows] == [
        ("b", "test", "0.7000"), ("a", "test", "0.6000"), ("a", "dev", "0.4000")]
