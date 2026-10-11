import numpy as np

from tel.runtime.host import chunk_sizes


def test_chunk_sizes_dynamic():
    assert chunk_sizes(100, {None: None}) == [(100, None)]


def test_chunk_sizes_fixed_batches():
    graphs = {1: None, 64: None}
    assert chunk_sizes(1, graphs) == [(1, 1)]
    assert chunk_sizes(10, graphs) == [(10, 64)]
    assert chunk_sizes(130, graphs) == [(64, 64), (64, 64), (2, 64)]
    assert sum(n for n, _ in chunk_sizes(1000, graphs)) == 1000


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
