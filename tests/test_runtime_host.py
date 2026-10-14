import numpy as np

from tel.runtime.host import chunk_sizes


def test_chunk_sizes_dynamic():
    assert chunk_sizes(100, {None: None}) == [(100, None)]


def test_chunk_sizes_fixed_batches():
    graphs = {1: None, 64: None}
    assert chunk_sizes(1, graphs) == [(1, 1)]
    assert chunk_sizes(10, graphs) == [(1, 1)] * 10  # < half full: no padding to 64
    assert chunk_sizes(40, graphs) == [(40, 64)]  # >= half full: one padded call
    assert chunk_sizes(130, graphs) == [(64, 64), (64, 64), (1, 1), (1, 1)]
    assert sum(n for n, _ in chunk_sizes(1000, graphs)) == 1000
    three = {1: None, 8: None, 64: None}
    assert chunk_sizes(23, three) == [(8, 8), (8, 8), (7, 8)]
    assert chunk_sizes(3, three) == [(1, 1)] * 3


def test_edit_host_fixed_batch_matches_dynamic():
    from tel.edit_tree import ScriptVocab
    from tel.runtime.host import EditHost
    from tel.tokenizer import CharTokenizer

    pairs = [("மரங்கள்", "மரம்"), ("நாய்கள்", "நாய்"), ("வீடு", "வீடு")]
    tok = CharTokenizer.build(w for p in pairs for w in p)
    vocab = ScriptVocab.build(pairs)
    weights = np.random.default_rng(0).normal(size=(48, len(vocab))).astype(np.float32)

    def graph(xs):  # deterministic fake model; padded rows must not affect real ones
        return [(xs[0] % 7).astype(np.float32) @ weights]

    words = ["மரங்கள்", "வீடு", "நாய்கள்"] * 30
    dyn = EditHost(tok, vocab, 48, {None: graph}).lemmatize(words)
    fixed = EditHost(tok, vocab, 48, {1: graph, 64: graph}).lemmatize(words)
    assert dyn == fixed and len(dyn) == len(words)


def test_chunk_sizes_with_measured_cost():
    graphs = {1: None, 64: None}
    lstm_like = {1: 3.8, 64: 70.0}  # a batch-64 call costs ~18 batch-1 calls
    assert chunk_sizes(23, graphs, lstm_like) == [(23, 64)]
    assert chunk_sizes(10, graphs, lstm_like) == [(1, 1)] * 10
    edit_like = {1: 0.5, 64: 32.0}  # no batching gain: batch 1 always
    assert chunk_sizes(40, graphs, edit_like) == [(1, 1)] * 40
    split = chunk_sizes(150, graphs, lstm_like)
    assert sum(n for n, _ in split) == 150 and split[:2] == [(64, 64), (64, 64)]
    assert chunk_sizes(5, graphs, {1: 1.0}) == chunk_sizes(5, graphs)  # incomplete: heuristic


def test_calibrate_sets_cost_and_keeps_outputs():
    from tel.edit_tree import ScriptVocab
    from tel.runtime.host import EditHost
    from tel.tokenizer import CharTokenizer

    pairs = [("மரங்கள்", "மரம்"), ("நாய்கள்", "நாய்")]
    tok = CharTokenizer.build(w for p in pairs for w in p)
    vocab = ScriptVocab.build(pairs)
    weights = np.random.default_rng(1).normal(size=(48, len(vocab))).astype(np.float32)

    def graph(xs):
        return [(xs[0] % 5).astype(np.float32) @ weights]

    words = ["மரங்கள்", "நாய்கள்"] * 20
    host = EditHost(tok, vocab, 48, {1: graph, 64: graph})
    before = host.lemmatize(words)
    cost = host.calibrate(words)
    assert set(cost) == {1, 64} and all(c > 0 for c in cost.values())
    assert host.lemmatize(words) == before
    assert EditHost(tok, vocab, 48, {None: graph}).calibrate(words) is None
