"""Character BiLSTM encoder + attention + LSTM decoder.

Split into `encode` and `decode_step` so both can be exported as separate
graphs; the decoding loop (greedy here, beam for the teacher) stays in host code.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tel.tokenizer import BOS_ID, EOS_ID, PAD_ID

State = tuple[torch.Tensor, torch.Tensor]  # (h, c), each (layers, B, H)


class LSTMSeq2Seq(nn.Module):
    def __init__(
        self, vocab_size: int, emb: int = 64, hidden: int = 128, layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden, self.layers = hidden, layers
        self.src_emb = nn.Embedding(vocab_size, emb, padding_idx=PAD_ID)
        self.tgt_emb = nn.Embedding(vocab_size, emb, padding_idx=PAD_ID)
        self.encoder = nn.LSTM(
            emb, hidden, layers, batch_first=True, bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.mem_proj = nn.Linear(2 * hidden, hidden)
        self.bridge = nn.Linear(hidden, layers * hidden)
        self.decoder = nn.LSTM(
            emb + hidden, hidden, layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.attn_out = nn.Linear(2 * hidden, hidden)
        self.out = nn.Linear(hidden, vocab_size)
        self.drop = nn.Dropout(dropout)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> tuple[torch.Tensor, State]:
        """src (B, S) -> memory (B, S, H), initial decoder state."""
        x = self.drop(self.src_emb(src))
        enc, _ = self.encoder(x)
        memory = self.mem_proj(enc) * src_mask[..., None]
        pooled = memory.sum(1) / src_mask.sum(1, keepdim=True).clamp(min=1)
        h0 = torch.tanh(self.bridge(pooled))
        h0 = h0.view(-1, self.layers, self.hidden).transpose(0, 1).contiguous()
        return memory, (h0, torch.zeros_like(h0))

    def decode_step(
        self, prev: torch.Tensor, attn_prev: torch.Tensor, state: State,
        memory: torch.Tensor, src_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, State]:
        """One step. prev (B,), attn_prev (B, H) -> logits (B, V), attentional h (B, H), state.

        Uses input feeding: the previous attentional vector is fed with the token.
        """
        x = torch.cat([self.drop(self.tgt_emb(prev)), attn_prev], -1)[:, None, :]
        out, state = self.decoder(x, state)
        h = out[:, 0]
        scores = torch.einsum("bh,bsh->bs", h, memory).masked_fill(~src_mask, -1e9)
        ctx = torch.einsum("bs,bsh->bh", F.softmax(scores, -1), memory)
        attn_h = torch.tanh(self.attn_out(torch.cat([h, ctx], -1)))
        return self.out(self.drop(attn_h)), attn_h, state

    def forward(
        self, src: torch.Tensor, src_mask: torch.Tensor, tgt_in: torch.Tensor
    ) -> torch.Tensor:
        """Teacher forcing. tgt_in (B, T) starts with BOS -> logits (B, T, V)."""
        memory, state = self.encode(src, src_mask)
        attn_h = memory.new_zeros(src.size(0), self.hidden)
        logits = []
        for t in range(tgt_in.size(1)):
            step_logits, attn_h, state = self.decode_step(
                tgt_in[:, t], attn_h, state, memory, src_mask
            )
            logits.append(step_logits)
        return torch.stack(logits, 1)

    @torch.no_grad()
    def greedy(self, src: torch.Tensor, src_mask: torch.Tensor, max_len: int) -> torch.Tensor:
        """Greedy decoding -> (B, <=max_len) token ids (EOS-terminated, then PAD)."""
        memory, state = self.encode(src, src_mask)
        attn_h = memory.new_zeros(src.size(0), self.hidden)
        prev = torch.full((src.size(0),), BOS_ID, dtype=torch.long, device=src.device)
        done = torch.zeros_like(prev, dtype=torch.bool)
        out = []
        for _ in range(max_len):
            logits, attn_h, state = self.decode_step(prev, attn_h, state, memory, src_mask)
            prev = logits.argmax(-1).masked_fill(done, PAD_ID)
            out.append(prev)
            done |= prev == EOS_ID
            if done.all():
                break
        return torch.stack(out, 1)
