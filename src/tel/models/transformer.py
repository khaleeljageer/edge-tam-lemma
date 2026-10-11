"""Character transformer encoder-decoder (pre-norm).

Attention is written out with additive float masks (`MASK_NEG`), no
masked_fill or boolean masks, so the graphs stay within the ops every runtime
converts (same rules as the LSTM, see docs/code_plan.md). Positions are learned
and counted from the start; padding sits at the end and is masked as keys, so
outputs at real positions do not depend on pad length.

Decoding recomputes the decoder over the prefix at each step (no KV cache):
lemmas are short (< 30 characters), and a cache-free step is simpler to export.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from tel.models.common import MASK_NEG
from tel.tokenizer import PAD_ID


class Attention(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float) -> None:
        super().__init__()
        if d % heads:
            raise ValueError(f"d_model {d} not divisible by heads {heads}")
        self.heads, self.dh = heads, d // heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mem: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """x (B, Tq, D), mem (B, Tk, D), bias (B or 1, Tq or 1, Tk) additive -> (B, Tq, D)."""
        b, tq, d = x.shape
        tk = mem.size(1)
        q = self.q(x).view(b, tq, self.heads, self.dh).transpose(1, 2)
        k = self.k(mem).view(b, tk, self.heads, self.dh).transpose(1, 2)
        v = self.v(mem).view(b, tk, self.heads, self.dh).transpose(1, 2)
        att = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dh) + bias.unsqueeze(1)
        y = torch.matmul(self.drop(F.softmax(att, -1)), v)
        return self.o(y.transpose(1, 2).reshape(b, tq, d))


class FeedForward(nn.Module):
    def __init__(self, d: int, ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ff, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d: int, heads: int, ff: int, dropout: float) -> None:
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = Attention(d, heads, dropout)
        self.ff = FeedForward(d, ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        x = x + self.drop(self.att(h, h, bias))
        return x + self.drop(self.ff(self.ln2(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d: int, heads: int, ff: int, dropout: float) -> None:
        super().__init__()
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.self_att = Attention(d, heads, dropout)
        self.cross_att = Attention(d, heads, dropout)
        self.ff = FeedForward(d, ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, y: torch.Tensor, mem: torch.Tensor, self_bias: torch.Tensor,
                cross_bias: torch.Tensor) -> torch.Tensor:
        h = self.ln1(y)
        y = y + self.drop(self.self_att(h, h, self_bias))
        y = y + self.drop(self.cross_att(self.ln2(y), mem, cross_bias))
        return y + self.drop(self.ff(self.ln3(y)))


class CharTransformer(nn.Module):
    def __init__(
        self, vocab_size: int, d_model: int = 128, heads: int = 4, layers: int = 2,
        ff_mult: int = 4, dropout: float = 0.1, max_len: int = 48,
    ) -> None:
        super().__init__()
        self.d_model, self.max_len = d_model, max_len
        self.src_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.tgt_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        # +2: target side has BOS and may run to max_len + 1 tokens.
        self.src_pos = nn.Embedding(max_len, d_model)
        self.tgt_pos = nn.Embedding(max_len + 2, d_model)
        ff = ff_mult * d_model
        self.enc = nn.ModuleList(EncoderLayer(d_model, heads, ff, dropout) for _ in range(layers))
        self.dec = nn.ModuleList(DecoderLayer(d_model, heads, ff, dropout) for _ in range(layers))
        self.enc_ln, self.dec_ln = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)
        self.drop = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)

    # --- graph-level pieces (exportable) ------------------------------------------

    def encode_features(self, src: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """src (B, S), mask (B, S) float -> memory (B, S, D)."""
        pos = torch.arange(src.size(1), device=src.device)
        x = self.drop(self.src_emb(src) * self.scale + self.src_pos(pos)[None])
        bias = ((mask - 1.0) * MASK_NEG).unsqueeze(1)  # (B, 1, S)
        for layer in self.enc:
            x = layer(x, bias)
        return self.enc_ln(x) * mask.unsqueeze(-1)

    def decode_features(self, prefix: torch.Tensor, memory: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
        """prefix (B, T) target ids starting with BOS -> hidden states (B, T, D)."""
        t = prefix.size(1)
        pos = torch.arange(t, device=prefix.device)
        y = self.drop(self.tgt_emb(prefix) * self.scale + self.tgt_pos(pos)[None])
        causal = torch.triu(torch.ones(t, t, device=prefix.device), diagonal=1) * -MASK_NEG
        cross = ((mask - 1.0) * MASK_NEG).unsqueeze(1)  # (B, 1, S)
        for layer in self.dec:
            y = layer(y, memory, causal.unsqueeze(0), cross)
        return self.dec_ln(y)

    # --- training ---------------------------------------------------------------------

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        return self.encode_features(src, src_mask.to(self.out.weight.dtype))

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor,
                tgt_in: torch.Tensor) -> torch.Tensor:
        """Teacher forcing, all positions in parallel. tgt_in (B, T) -> logits (B, T, V)."""
        mask = src_mask.to(self.out.weight.dtype)
        memory = self.encode_features(src, mask)
        return self.out(self.drop(self.decode_features(tgt_in, memory, mask)))

    @property
    def prefix_len(self) -> int:
        """Fixed prefix length for exported decode steps: BOS + up to max_len tokens."""
        return self.max_len + 1

    def decode_step_fixed(self, prefix: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor,
                          sel: torch.Tensor) -> torch.Tensor:
        """Exported decode-step graph with fixed shapes.

        prefix (B, T) ids padded to T = prefix_len, sel (B, T) one-hot float marking the
        current position -> logits (B, V) at that position. The causal mask hides the
        padding after the current position, and the position is picked with a
        multiply-and-sum (no gather), so the graph converts to every runtime.
        """
        h = self.decode_features(prefix, memory, mask)
        return self.out((h * sel.unsqueeze(-1)).sum(1))

    # --- decoding interface (tel.decoding) ------------------------------------------

    def decoder_start(self, src: torch.Tensor, src_mask: torch.Tensor):
        mask = src_mask.to(self.out.weight.dtype)
        prefix = src.new_zeros(src.size(0), 0)
        return self.encode_features(src, mask), mask, prefix

    def decoder_step(self, prev: torch.Tensor, prefix: torch.Tensor, memory: torch.Tensor,
                     mask: torch.Tensor):
        prefix = torch.cat([prefix, prev], 1)
        h = self.decode_features(prefix, memory, mask)
        return self.out(h[:, -1]), prefix

    @staticmethod
    def reorder_state(prefix: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return prefix[idx]
