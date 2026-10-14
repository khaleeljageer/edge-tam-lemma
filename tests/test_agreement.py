import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "data"))

from agreement import agreement_rows, cohen_kappa, stratified_sample  # noqa: E402


def rows(n_agree, n_majority, n_conflict):
    out = []
    for status, n in (("agree", n_agree), ("majority", n_majority), ("conflict", n_conflict)):
        out += [{"id": f"{status}{i}", "status": status, "word": "மரங்கள்", "lemma": "மரம்"}
                for i in range(n)]
    return out


def test_sample_is_proportional_and_fixed():
    data = rows(50, 30, 20)
    a = stratified_sample(data, 10, seed=13)
    assert [r["status"] for r in a].count("agree") == 5
    assert [r["status"] for r in a].count("conflict") == 2
    assert a == stratified_sample(data, 10, seed=13)


def test_agreement_and_kappa():
    first = {r["id"]: r for r in rows(2, 1, 1)}
    second = [{"id": "agree0", "lemma": "மரம்"}, {"id": "agree1", "lemma": "மரங்கள்"},
              {"id": "majority0", "lemma": "மரம்"}, {"id": "conflict0", "lemma": ""}]
    out = {r["status"]: r for r in agreement_rows(first, second)}
    assert out["all"]["n"] == "3" and out["all"]["agreement"] == "0.6667"
    assert out["agree"]["agreement"] == "0.5000"
    assert "conflict" not in out  # left empty: not counted
    assert cohen_kappa([True, False], [True, False]) == 1.0
