import csv
import json

import yaml

from tel import sweep


def test_parse_and_label_sizes():
    assert sweep.parse_size("1M") == 1_000_000
    assert sweep.parse_size("500k") == 500_000
    assert sweep.parse_size("0.5M") == 500_000
    assert sweep.size_label(500_000) == "500k" and sweep.size_label(2_000_000) == "2M"


def test_make_configs(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "GENERATED", tmp_path / "gen")
    base = tmp_path / "base.yaml"
    base.write_text("name: lstm\nmodel: {type: lstm_seq2seq}\ntrain: {epochs: 3}\n")
    paths = sweep.make_configs([base], [1_000_000], [13, 14])
    assert [p.stem for p in paths] == ["lstm_1M_s13", "lstm_1M_s14"]
    cfg = yaml.safe_load(paths[1].read_text())
    assert cfg["model"]["target_params"] == 1_000_000
    assert cfg["train"] == {"epochs": 3, "seed": 14}


def test_collect_rows_and_csv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sweep, "GENERATED", tmp_path / "gen")
    base = tmp_path / "base.yaml"
    base.write_text("name: lstm_kd\nmodel: {type: lstm_seq2seq}\n"
                    "data: {train_file: d/train.tsv}\n")
    done, missing = sweep.make_configs([base], [500_000], [13, 14])
    split = {"acc": 0.5, "n": 10, "acc_noun": 0.6, "acc_seen": None}
    (tmp_path / "runs" / done.stem).mkdir(parents=True)
    (tmp_path / "runs" / done.stem / "metrics.json").write_text(json.dumps(
        {"name": done.stem, "params": 123, "seed": 13, "git_sha": "abcdef123",
         "dev": split, "test": split, "test_unseen": split}))
    rows = sweep.collect_rows([done, missing])
    assert [r["split"] for r in rows] == ["dev", "test", "test_unseen"]
    assert rows[0]["base"] == "lstm_kd" and rows[0]["target_params"] == 500_000
    assert rows[0]["train_file"] == "d/train.tsv" and rows[0]["git_sha"] == "abcdef1"
    assert rows[0]["acc"] == "0.5000" and rows[0]["acc_seen"] == ""
    sweep.write_csv(rows, tmp_path / "out" / "a.csv")
    back = list(csv.DictReader(open(tmp_path / "out" / "a.csv")))
    assert len(back) == 3 and back[1]["run"] == "lstm_kd_500k_s13"
