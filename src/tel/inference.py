"""Wrap trained PyTorch models in the `Lemmatizer` interface."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from tel.decoding import beam_decode, greedy_decode
from tel.edit_tree import ScriptVocab
from tel.models.edit_classifier import EditClassifier
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
    """Any model implementing the `tel.decoding` interface (LSTM, transformer)."""

    def __init__(
        self, model, tok: CharTokenizer, max_len: int, batch_size: int = 256,
        beam: int = 1, alpha: float = 0.0,
    ) -> None:
        self.model, self.tok, self.max_len, self.batch_size = model, tok, max_len, batch_size
        self.beam, self.alpha = beam, alpha

    def lemmatize(self, words: list[str]) -> list[str]:
        return [hyps[0][0] for hyps in self.nbest(words, n=1)]

    def nbest(self, words: list[str], n: int | None = None) -> list[list[tuple[str, float]]]:
        """Per word, up to n (lemma, logprob) candidates, best first. Greedy gives 1."""
        self.model.eval()
        out: list[list[tuple[str, float]]] = []
        for i in range(0, len(words), self.batch_size):
            src, mask = encode_batch(self.tok, words[i : i + self.batch_size], self.max_len)
            if self.beam <= 1:
                ids = greedy_decode(self.model, src, mask, self.max_len)
                out.extend([(self.tok.decode(row.tolist()), 0.0)] for row in ids)
                continue
            for hyps in beam_decode(self.model, src, mask, self.max_len, self.beam, self.alpha):
                out.append([(self.tok.decode(h.tokens), h.logprob) for h in hyps[:n]])
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
