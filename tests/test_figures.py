import sys
from pathlib import Path

import pytest

FIGURES = Path(__file__).resolve().parents[1] / "paper" / "figures"
sys.path.insert(0, str(FIGURES))

pytest.importorskip("matplotlib")
common = pytest.importorskip("common")


def test_pareto_front():
    pts = [(1.0, 70.0), (0.5, 65.0), (2.0, 72.0), (1.5, 69.0), (0.5, 60.0)]
    assert common.pareto_front(pts) == [1, 0, 2]


def test_best_by_and_seed_stats():
    rows = [{"m": "a", "lat": "2.0", "acc": "0.5"}, {"m": "a", "lat": "1.0", "acc": "0.7"},
            {"m": "b", "lat": "3.0", "acc": "0.6"}]
    best = {r["m"]: r["lat"] for r in common.best_by(rows, ("m",), "lat")}
    assert best == {"a": "1.0", "b": "3.0"}
    mean, std, n = common.seed_stats(rows, ("m",), "acc")[("a",)]
    assert (round(mean, 6), n) == (0.6, 2) and std > 0


def test_short_model():
    assert common.short_model("edit_kd_500k_s13") == "edit_kd_500k"
    assert common.short_model("lstm_tiny") == "lstm_tiny"


def test_all_outputs_build(tmp_path):
    import fig_params
    import fig_pareto
    import tables

    paths = fig_params.main(tmp_path) + fig_pareto.main(tmp_path) + tables.main(tmp_path)
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
    tex = (tmp_path / "tab_quant.tex").read_text(encoding="utf-8")
    assert r"\toprule" in tex and "LiteRT int8 dyn" in tex
    assert r"\ta{" in (tmp_path / "tab_error_examples.tex").read_text(encoding="utf-8")
