"""Wrap trained PyTorch models in the `Lemmatizer` interface."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from tel.edit_tree import ScriptVocab
from tel.models.edit_classifier import EditClassifier
from tel.models.lstm_seq2seq import LSTMSeq2Seq
from tel.tokenizer import PAD_ID, CharTokenizer


def encode_batch(
    tok: CharTokenizer, words: Sequence[str], max_len: int, add_bos: bool = False,
    add_eos: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode and right-pad to the longest sequence (truncated at max_len) -> ids, mask."""
    seqs = [tok.encode(w, add_bos=add_bos, add_eos=add_eos)[:max_len] or [PAD_ID] for w in words]
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), PAD_ID, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
    return ids, ids != PAD_ID


class Seq2SeqLemmatizer:
    def __init__(
        self, model: LSTMSeq2Seq, tok: CharTokenizer, max_len: int, batch_size: int = 256
    ) -> None:
        self.model, self.tok, self.max_len, self.batch_size = model, tok, max_len, batch_size

    def lemmatize(self, words: list[str]) -> list[str]:
        self.model.eval()
        out: list[str] = []
        for i in range(0, len(words), self.batch_size):
            src, mask = encode_batch(self.tok, words[i : i + self.batch_size], self.max_len)
            ids = self.model.greedy(src, mask, self.max_len)
            out.extend(self.tok.decode(row.tolist()) for row in ids)
        return out


def applicable_masks(vocab: ScriptVocab, words: Sequence[str]) -> torch.Tensor:
    """(N, C) bool: which edit scripts can apply to each word."""
    return torch.tensor([vocab.applicable_mask(w) for w in words], dtype=torch.bool)


class EditLemmatizer:
    def __init__(
        self, model: EditClassifier, tok: CharTokenizer, vocab: ScriptVocab, max_len: int,
        batch_size: int = 256,
    ) -> None:
        self.model, self.tok, self.vocab = model, tok, vocab
        self.max_len, self.batch_size = max_len, batch_size

    def predict_scripts(self, words: list[str]) -> list[int]:
        self.model.eval()
        out: list[int] = []
        with torch.no_grad():
            for i in range(0, len(words), self.batch_size):
                chunk = words[i : i + self.batch_size]
                src, mask = encode_batch(self.tok, chunk, self.max_len)
                logits = self.model(src, mask)
                logits = logits.masked_fill(~applicable_masks(self.vocab, chunk), -1e9)
                out.extend(logits.argmax(-1).tolist())
        return out

    def lemmatize(self, words: list[str]) -> list[str]:
        ids = self.predict_scripts(words)
        # Masking guarantees the chosen script applies; identity (class 0) always does.
        return [self.vocab.itos[c].apply(w) or w for c, w in zip(ids, words, strict=True)]
