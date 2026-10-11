"""Runtime-independent host code for exported models (numpy only).

The exported graphs only compute tensors; tokenization, padding, the greedy
decoding loop and edit-script masking live here, so every runtime (ONNX
Runtime, ExecuTorch, LiteRT) shares exactly the same logic.

A graph is a callable `list[np.ndarray] -> list[np.ndarray]`. Graphs exported
with a dynamic batch use key None; fixed-batch graphs are keyed by batch size,
and inputs are padded up to the smallest size that fits.

Graph signatures (B = batch, S = max_len):
    edit model   (src)             or, host features, (src, mask, rpos) -> logits
    encoder      (src)             or, host features, (src, mask, perm) -> memory, h0
    decode_step  (prev (B,1), attn, h, c, memory, mask) -> logits, attn, h, c
transformer:
    encoder      (src)             or, host features, (src, mask) -> memory
    decode_step  (prefix (B,T), memory, mask, sel (B,T) one-hot) -> logits
With host features (ncnn) the padding mask, end-relative positions and the
reversal permutation are computed here instead of inside the graph. Inputs are
built by the names the manifest lists for each graph (`graph_inputs`), so a
graph that needs no permutation (CNN encoder) simply does not ask for one.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import numpy as np

from tel.edit_tree import ScriptVocab
from tel.tokenizer import BOS_ID, EOS_ID, PAD_ID, CharTokenizer

Graph = Callable[[list[np.ndarray]], list[np.ndarray]]
Graphs = dict[int | None, Graph]  # batch size (None = dynamic) -> graph


def encode_fixed(tok: CharTokenizer, words: Sequence[str], max_len: int) -> np.ndarray:
    """Encode words and pad/truncate to exactly max_len -> (B, max_len) int64."""
    ids = np.full((len(words), max_len), PAD_ID, dtype=np.int64)
    for i, w in enumerate(words):
        seq = tok.encode(w)[:max_len]
        ids[i, : len(seq)] = seq
    return ids


def padding_mask(src: np.ndarray) -> np.ndarray:
    """(B, S) ids -> (B, S) float32 0/1."""
    return (src != PAD_ID).astype(np.float32)


def end_positions(src: np.ndarray, max_pos: int) -> np.ndarray:
    """(B, S) ids -> (B, S) int64 positions counted from the last real character."""
    lengths = (src != PAD_ID).sum(1, keepdims=True)
    return np.clip(lengths - 1 - np.arange(src.shape[1])[None, :], 0, max_pos - 1)


def reversal_matrices(mask: np.ndarray) -> np.ndarray:
    """(B, S) float mask -> (B, S, S) permutations reversing each word in place.

    Same definition as `tel.models.lstm_seq2seq.reversal_matrix`.
    """
    b, s = mask.shape
    perm = np.zeros((b, s, s), dtype=np.float32)
    t = np.arange(s)
    for i, n in enumerate(mask.sum(1).astype(int)):
        idx = np.where(t < n, n - 1 - t, t)
        perm[i, t, idx] = 1.0
    return perm


def chunk_sizes(n: int, graphs: Graphs,
                cost: dict[int, float] | None = None) -> list[tuple[int, int | None]]:
    """Split n items into (chunk_len, graph_key) pieces that the graphs can take.

    With `cost` (seconds per call of each fixed-batch graph, from `calibrate`): the
    split with the lowest total cost, padding included. Without it, a heuristic that
    assumes a call costs about the same however much of the batch is padding: full
    batches of the largest graph first; a remainder r is padded up to the smallest
    graph that fits only if it fills at least half of it, otherwise it goes through
    the next smaller graph(s). With graphs {1, 64}, 23 leftover words (e.g. the
    table misses of a hybrid batch) take 23 batch-1 calls instead of one call
    padded to 64. That assumption fails for the LSTM on LiteRT, where a batch-64
    call costs only ~18 batch-1 calls; calibrate to get the cost-based split.
    """
    if None in graphs:
        return [(n, None)]
    sizes = sorted(k for k in graphs if k is not None)
    if cost is not None and all(s in cost for s in sizes):
        return _cheapest_split(n, sizes, cost)
    out = []
    while n > 0:
        fit = next((s for s in sizes if s >= n), sizes[-1])
        if fit > n and 2 * n < fit and fit != sizes[0]:
            fit = max(s for s in sizes if s < fit)  # too much padding: step down
        take = min(n, fit)
        out.append((take, fit))
        n -= take
    return out


def _cheapest_split(n: int, sizes: list[int], cost: dict[int, float]) -> list[tuple[int, int]]:
    """Dynamic programme over the number of items left: best[r] = cheapest way to
    process r items, each call taking up to `size` of them."""
    best = [0.0] + [float("inf")] * n
    choice = [0] * (n + 1)
    for r in range(1, n + 1):
        for s in sizes:
            c = cost[s] + best[max(0, r - s)]
            if c < best[r]:
                best[r], choice[r] = c, s
    out = []
    while n > 0:
        take = min(n, choice[n])
        out.append((take, choice[n]))
        n -= take
    return sorted(out, key=lambda x: -x[1])  # largest graphs first


class _ChunkedHost:
    """Shared batching for the hosts: `_call(src, key)` runs one graph call on a
    (key, max_len) id batch; `calibrate` measures each fixed-batch graph's cost."""

    tok: CharTokenizer
    max_len: int
    cost: dict[int, float] | None = None

    def _graphs(self) -> Graphs:
        raise NotImplementedError

    def _call(self, src: np.ndarray, key: int | None) -> np.ndarray:
        raise NotImplementedError

    def _chunks(self, words: Sequence[str]):
        """Yield (n, output of one graph call trimmed to n rows) over `words`."""
        src = encode_fixed(self.tok, words, self.max_len)
        start = 0
        for n, key in chunk_sizes(len(words), self._graphs(), self.cost):
            yield n, self._call(_pad_batch(src[start : start + n], key), key)[:n]
            start += n

    def calibrate(self, words: Sequence[str], reps: int = 3) -> dict[int, float] | None:
        """Time one call of every fixed-batch graph on real words (best of `reps`,
        after one warm-up call) so `chunk_sizes` can pick the cheapest split.
        Nothing to do for dynamic-batch graphs or a single fixed size."""
        sizes = sorted(k for k in self._graphs() if k is not None)
        if len(sizes) < 2 or not words:
            return None
        cost = {}
        for s in sizes:
            src = encode_fixed(self.tok, [words[i % len(words)] for i in range(s)],
                               self.max_len)
            self._call(src, s)
            times = []
            for _ in range(reps):
                t0 = time.perf_counter()
                self._call(src, s)
                times.append(time.perf_counter() - t0)
            cost[s] = min(times)
        self.cost = cost
        return cost


def _pad_batch(x: np.ndarray, size: int | None, axis: int = 0) -> np.ndarray:
    if size is None or x.shape[axis] == size:
        return x
    pad = [(0, 0)] * x.ndim
    pad[axis] = (0, size - x.shape[axis])
    return np.pad(x, pad)


def graph_inputs(names: Sequence[str], src: np.ndarray, max_len: int) -> list[np.ndarray]:
    """Build a graph's inputs by name (from the manifest) for a batch of source ids."""
    mask = padding_mask(src)
    make = {"src": lambda: src, "mask": lambda: mask,
            "rpos": lambda: end_positions(src, max_len), "perm": lambda: reversal_matrices(mask)}
    return [make[n]() for n in names]


class EditHost(_ChunkedHost):
    def __init__(self, tok: CharTokenizer, vocab: ScriptVocab, max_len: int,
                 model: Graphs, input_names: Sequence[str] = ("src",)) -> None:
        self.tok, self.vocab, self.max_len, self.model = tok, vocab, max_len, model
        self.input_names = list(input_names)

    def _graphs(self) -> Graphs:
        return self.model

    def _call(self, src: np.ndarray, key: int | None) -> np.ndarray:
        return self.model[key](graph_inputs(self.input_names, src, self.max_len))[0]

    def logits(self, words: Sequence[str]) -> np.ndarray:
        return np.concatenate([out for _, out in self._chunks(words)])

    def lemmatize(self, words: list[str]) -> list[str]:
        if not words:
            return []
        logits = self.logits(words)
        masked = np.full_like(logits, -1e9)
        for row, w in enumerate(words):
            ids = self.vocab.applicable_ids(w)
            masked[row, ids] = logits[row, ids]
        classes = masked.argmax(-1)
        return [self.vocab.itos[c].apply(w) or w for c, w in zip(classes, words, strict=True)]


class Seq2SeqHost(_ChunkedHost):
    def __init__(self, tok: CharTokenizer, max_len: int, hidden: int, layers: int,
                 encoder: Graphs, decode_step: Graphs,
                 encoder_inputs: Sequence[str] = ("src",)) -> None:
        self.tok, self.max_len, self.hidden, self.layers = tok, max_len, hidden, layers
        self.encoder, self.decode_step = encoder, decode_step
        self.encoder_inputs = list(encoder_inputs)

    def _greedy(self, src: np.ndarray, key: int | None) -> np.ndarray:
        encode, step = self.encoder[key], self.decode_step[key]
        mask = padding_mask(src)
        memory, h0 = encode(graph_inputs(self.encoder_inputs, src, self.max_len))
        b = src.shape[0]
        h = np.ascontiguousarray(
            h0.reshape(b, self.layers, self.hidden).transpose(1, 0, 2), dtype=np.float32
        )
        c = np.zeros_like(h)
        attn = np.zeros((b, self.hidden), dtype=np.float32)
        prev = np.full((b,), BOS_ID, dtype=np.int64)
        done = np.zeros(b, dtype=bool)
        out = []
        for _ in range(self.max_len):
            logits, attn, h, c = step([prev[:, None], attn, h, c, memory, mask])
            prev = np.where(done, PAD_ID, logits.argmax(-1)).astype(np.int64)
            out.append(prev)
            done |= prev == EOS_ID
            if done.all():
                break
        return np.stack(out, 1)

    def _graphs(self) -> Graphs:
        return self.encoder

    def _call(self, src: np.ndarray, key: int | None) -> np.ndarray:
        return self._greedy(src, key)

    def lemmatize(self, words: list[str]) -> list[str]:
        if not words:
            return []
        return [self.tok.decode(row.tolist()) for _, ids in self._chunks(words) for row in ids]


class TransformerHost(_ChunkedHost):
    """Greedy decoding for the exported character transformer (fixed-length prefix)."""

    def __init__(self, tok: CharTokenizer, max_len: int, prefix_len: int, encoder: Graphs,
                 decode_step: Graphs, encoder_inputs: Sequence[str] = ("src",)) -> None:
        self.tok, self.max_len, self.prefix_len = tok, max_len, prefix_len
        self.encoder, self.decode_step = encoder, decode_step
        self.encoder_inputs = list(encoder_inputs)

    def _greedy(self, src: np.ndarray, key: int | None) -> np.ndarray:
        encode, step = self.encoder[key], self.decode_step[key]
        mask = padding_mask(src)
        memory = encode(graph_inputs(self.encoder_inputs, src, self.max_len))[0]
        b = src.shape[0]
        prefix = np.full((b, self.prefix_len), PAD_ID, dtype=np.int64)
        prefix[:, 0] = BOS_ID
        done = np.zeros(b, dtype=bool)
        out = []
        for t in range(min(self.max_len, self.prefix_len - 1)):
            sel = np.zeros((b, self.prefix_len), dtype=np.float32)
            sel[:, t] = 1.0
            logits = step([prefix, memory, mask, sel])[0]
            tok = np.where(done, PAD_ID, logits.argmax(-1)).astype(np.int64)
            out.append(tok)
            done |= tok == EOS_ID
            if done.all():
                break
            prefix[:, t + 1] = tok
        return np.stack(out, 1)

    def _graphs(self) -> Graphs:
        return self.encoder

    def _call(self, src: np.ndarray, key: int | None) -> np.ndarray:
        return self._greedy(src, key)

    def lemmatize(self, words: list[str]) -> list[str]:
        if not words:
            return []
        return [self.tok.decode(row.tolist()) for _, ids in self._chunks(words) for row in ids]
