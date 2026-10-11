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


def chunk_sizes(n: int, graphs: Graphs) -> list[tuple[int, int | None]]:
    """Split n items into (chunk_len, graph_key) pieces that the graphs can take."""
    if None in graphs:
        return [(n, None)]
    sizes = sorted(k for k in graphs if k is not None)
    out = []
    while n > 0:
        fit = next((s for s in sizes if s >= n), sizes[-1])
        take = min(n, fit)
        out.append((take, fit))
        n -= take
    return out


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


class EditHost:
    def __init__(self, tok: CharTokenizer, vocab: ScriptVocab, max_len: int,
                 model: Graphs, input_names: Sequence[str] = ("src",)) -> None:
        self.tok, self.vocab, self.max_len, self.model = tok, vocab, max_len, model
        self.input_names = list(input_names)

    def logits(self, words: Sequence[str]) -> np.ndarray:
        src = encode_fixed(self.tok, words, self.max_len)
        parts, start = [], 0
        for n, key in chunk_sizes(len(words), self.model):
            chunk = _pad_batch(src[start : start + n], key)
            out = self.model[key](graph_inputs(self.input_names, chunk, self.max_len))[0]
            parts.append(out[:n])
            start += n
        return np.concatenate(parts)

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


class Seq2SeqHost:
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

    def lemmatize(self, words: list[str]) -> list[str]:
        if not words:
            return []
        src = encode_fixed(self.tok, words, self.max_len)
        result, start = [], 0
        for n, key in chunk_sizes(len(words), self.encoder):
            ids = self._greedy(_pad_batch(src[start : start + n], key), key)[:n]
            result.extend(self.tok.decode(row.tolist()) for row in ids)
            start += n
        return result


class TransformerHost:
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

    def lemmatize(self, words: list[str]) -> list[str]:
        if not words:
            return []
        src = encode_fixed(self.tok, words, self.max_len)
        result, start = [], 0
        for n, key in chunk_sizes(len(words), self.encoder):
            ids = self._greedy(_pad_batch(src[start : start + n], key), key)[:n]
            result.extend(self.tok.decode(row.tolist()) for row in ids)
            start += n
        return result
