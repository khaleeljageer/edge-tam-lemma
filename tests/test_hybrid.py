from tel.data import Pair
from tel.hybrid import (
    HybridLemmatizer,
    build_table,
    gold_lemmas,
    read_table,
    weighted_stream,
    write_table,
)


class Echo:
    def __init__(self):
        self.calls = []

    def lemmatize(self, words):
        self.calls.append(list(words))
        return [w + "?" for w in words]


def test_hybrid_uses_table_and_batches_misses():
    fallback = Echo()
    lem = HybridLemmatizer({"மரங்கள்": "மரம்"}, fallback)
    assert lem.lemmatize(["மரங்கள்", "வீடுகள்", "மரங்கள்", "கதைகள்"]) == [
        "மரம்", "வீடுகள்?", "மரம்", "கதைகள்?"]
    assert fallback.calls == [["வீடுகள்", "கதைகள்"]]
    assert lem.hit_rate == 0.5
    lem.lemmatize(["மரங்கள்"])
    assert fallback.calls == [["வீடுகள்", "கதைகள்"]]  # all hits: model not called


def test_build_table_priority_and_skips():
    ranked = ["அ", "ஆ", "இ", "ஈ"]
    rows = build_table(ranked, 3, [("gold", {"ஆ": "ஆ1"}), ("silver", {"அ": "அ2", "ஆ": "ஆ2"})])
    assert rows == [("அ", "அ2", "silver"), ("ஆ", "ஆ1", "gold")]  # இ unlabelled, ஈ beyond n


def test_gold_lemmas_most_frequent(tmp_path):
    pairs = [Pair("படி", "படி", "VERB", "x", 1), Pair("படி", "படு", "VERB", "x", 3)]
    assert gold_lemmas(pairs) == {"படி": "படு"}
    write_table(tmp_path / "t.tsv", [("படி", "படு", "gold")])
    assert read_table(tmp_path / "t.tsv") == {"படி": "படு"}


def test_weighted_stream_counts_and_is_fixed():
    s = weighted_stream(["a", "b"], [3, 1])
    assert sorted(s) == ["a", "a", "a", "b"] and s == weighted_stream(["a", "b"], [3, 1])
