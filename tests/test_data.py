from tel.data import Pair, map_upos, read_tsv, split_by_lemma, write_tsv


def _pairs(n_lemmas=50, forms=3):
    return [Pair(f"w{i}_{j}", f"l{i}") for i in range(n_lemmas) for j in range(forms)]


def test_split_has_no_lemma_overlap():
    splits, lemmas = split_by_lemma(_pairs())
    sets = {name: {p.lemma for p in ps} for name, ps in splits.items()}
    assert not sets["train"] & sets["dev"]
    assert not sets["train"] & sets["test"]
    assert not sets["dev"] & sets["test"]
    assert sum(len(ps) for ps in splits.values()) == 150
    assert {name: set(ls) for name, ls in lemmas.items()} == sets


def test_split_is_deterministic():
    a, _ = split_by_lemma(_pairs(), seed=7)
    b, _ = split_by_lemma(_pairs(), seed=7)
    c, _ = split_by_lemma(_pairs(), seed=8)
    assert a == b
    assert a != c


def test_split_proportions():
    _, lemmas = split_by_lemma(_pairs(n_lemmas=100))
    assert [len(lemmas[n]) for n in ("train", "dev", "test")] == [80, 10, 10]


def test_tsv_roundtrip(tmp_path):
    pairs = [Pair("வீட்டை", "வீடு", "NOUN", "ud-ttb", 3, 0), Pair("செல்", "செல்", "VERB")]
    write_tsv(tmp_path / "x.tsv", pairs)
    assert read_tsv(tmp_path / "x.tsv") == pairs


def test_map_upos():
    assert map_upos("PROPN") == "NOUN"
    assert map_upos("AUX") == "VERB"
    assert map_upos("ADJ") == "OTHER"


def test_make_splits_seen_and_unseen():
    from tel.data import make_splits

    splits, lemmas = make_splits(_pairs(n_lemmas=100, forms=4), seen_frac=0.05)
    train_lemmas = {p.lemma for p in splits["train"]}
    train_keys = {(p.word, p.lemma) for p in splits["train"]}
    assert train_lemmas == set(lemmas["train"])  # every train lemma keeps a form
    for name in ("dev", "test"):
        seen = [p for p in splits[name] if p.lemma in train_lemmas]
        assert seen and len(seen) < len(splits[name])
        assert not {(p.word, p.lemma) for p in splits[name]} & train_keys
    assert all(p.lemma not in train_lemmas for p in splits["test_unseen"])
    assert set(map(id, splits["test_unseen"])) <= set(map(id, splits["test"]))
    total = sum(len(splits[n]) for n in ("train", "dev", "test"))
    assert total == 400
