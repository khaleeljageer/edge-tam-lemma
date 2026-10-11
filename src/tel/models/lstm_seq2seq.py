"""Character BiLSTM encoder + attention + LSTM decoder.

Split into `encode` and `decode_step` so both can be exported as separate
graphs; the decoding loop (greedy here, beam for the teacher) stays in host code.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tel.models.common import MASK_NEG
from tel.tokenizer import PAD_ID

State = tuple[torch.Tensor, torch.Tensor]  # (h, c), each (layers, B, H)


def reversal_matrix(mask: torch.Tensor) -> torch.Tensor:
    """Permutation that reverses each sequence's real positions in place.

    mask (B, S) float 0/1 -> P (B, S, S) with (P @ x)[b, t] = x[b, idx[b, t]],
    where idx reverses the first `length` positions and leaves padding at the
    end. P is its own inverse. A matmul (not gather) so that it converts to
    every runtime, including ncnn.
    """
    lengths = mask.sum(1, keepdim=True)  # (B, 1)
    t = torch.arange(mask.size(1), device=mask.device, dtype=mask.dtype)[None, :]
    idx = torch.where(t < lengths, lengths - 1 - t, t)  # (B, S)
    return (idx[..., None] == t[:, None, :]).to(mask.dtype)


class LSTMSeq2Seq(nn.Module):
    """Attention LSTM decoder over a BiLSTM encoder (default) or a dilated CNN
    encoder (`encoder="cnn"`, the `char_cnn` ablation: same decoder, only the
    encoder changes)."""

    def __init__(
        self, vocab_size: int, emb: int = 64, hidden: int = 128, layers: int = 1,
        dropout: float = 0.1, encoder: str = "bilstm", conv_layers: int = 4, kernel: int = 3,
    ) -> None:
        super().__init__()
        self.hidden, self.layers, self.encoder_type = hidden, layers, encoder
        self.src_emb = nn.Embedding(vocab_size, emb, padding_idx=PAD_ID)
        self.tgt_emb = nn.Embedding(vocab_size, emb, padding_idx=PAD_ID)
        # Masks are float 0/1 and applied arithmetically (no masked_fill/gather)
        # so the same graphs convert to every runtime, including ncnn.
        if encoder == "bilstm":
            # Bidirectional encoder built from one-directional layers: the backward
            # LSTM runs on each word reversed in place, so neither direction reads
            # padding before real characters and results do not depend on pad length.
            self.enc_fwd = nn.ModuleList(
                nn.LSTM(emb if i == 0 else 2 * hidden, hidden, batch_first=True)
                for i in range(layers)
            )
            self.enc_bwd = nn.ModuleList(
                nn.LSTM(emb if i == 0 else 2 * hidden, hidden, batch_first=True)
                for i in range(layers)
            )
            enc_out = 2 * hidden
        elif encoder == "cnn":
            # Residual dilated convolutions (dilation 1, 2, 4, ...): with 4 layers and
            # kernel 3 each position sees 31 characters. Padding is zeroed after
            # every block, so outputs do not depend on pad length.
            self.enc_in = nn.Conv1d(emb, hidden, 1)
            self.enc_convs = nn.ModuleList(
                nn.Conv1d(hidden, hidden, kernel, padding=(kernel // 2) * 2**i, dilation=2**i)
                for i in range(conv_layers)
            )
            self.enc_norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(conv_layers))
            enc_out = hidden
        else:
            raise ValueError(f"unknown encoder: {encoder}")
        self.mem_proj = nn.Linear(enc_out, hidden)
        self.bridge = nn.Linear(hidden, layers * hidden)
        self.decoder = nn.LSTM(
            emb + hidden, hidden, layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.attn_out = nn.Linear(2 * hidden, hidden)
        self.out = nn.Linear(hidden, vocab_size)
        self.drop = nn.Dropout(dropout)

    @property
    def needs_perm(self) -> bool:
        """Whether the encoder graph takes the reversal permutation as an input."""
        return self.encoder_type == "bilstm"

    def encode_features(
        self, src: torch.Tensor, mask: torch.Tensor, perm: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exported encoder graph. src (B, S), mask (B, S) float, perm (B, S, S)
        (BiLSTM only) -> memory (B, S, H), h0 (B, layers * H)."""
        x = self.drop(self.src_emb(src))
        if self.encoder_type == "bilstm":
            for i, (fwd, bwd) in enumerate(zip(self.enc_fwd, self.enc_bwd, strict=True)):
                if i > 0:
                    x = self.drop(x)
                f, _ = fwd(x)
                b, _ = bwd(torch.matmul(perm, x))
                x = torch.cat([f, torch.matmul(perm, b)], -1)
        else:
            m = mask.unsqueeze(1)  # (B, 1, S)
            h = self.enc_in(x.transpose(1, 2)) * m
            for conv, norm in zip(self.enc_convs, self.enc_norms, strict=True):
                r = torch.relu(conv(self.drop(h)))
                h = norm((h + r).transpose(1, 2)).transpose(1, 2) * m
            x = h.transpose(1, 2)
        memory = self.mem_proj(x) * mask.unsqueeze(-1)
        pooled = memory.sum(1) / mask.sum(1, keepdim=True).clamp(min=1.0)
        return memory, torch.tanh(self.bridge(pooled))

    def init_state(self, h0: torch.Tensor) -> State:
        """h0 (B, layers * H) -> decoder state ((layers, B, H), zeros)."""
        h = h0.view(-1, self.layers, self.hidden).transpose(0, 1).contiguous()
        return h, torch.zeros_like(h)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> tuple[torch.Tensor, State]:
        """src (B, S), src_mask (B, S) bool -> memory (B, S, H), initial decoder state."""
        mask = src_mask.to(self.mem_proj.weight.dtype)
        perm = reversal_matrix(mask) if self.needs_perm else None
        memory, h0 = self.encode_features(src, mask, perm)
        return memory, self.init_state(h0)

    def decode_step(
        self, prev: torch.Tensor, attn_prev: torch.Tensor, state: State,
        memory: torch.Tensor, mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, State]:
        """Exported decode-step graph. prev (B, 1) token ids, attn_prev (B, H),
        mask (B, S) float -> logits (B, V), attentional h (B, H), state.

        Uses input feeding: the previous attentional vector is fed with the token.
        """
        x = torch.cat([self.drop(self.tgt_emb(prev)), attn_prev.unsqueeze(1)], -1)
        out, state = self.decoder(x, state)
        h = out[:, 0]
        scores = torch.einsum("bh,bsh->bs", h, memory) + (mask - 1.0) * MASK_NEG
        ctx = torch.einsum("bs,bsh->bh", F.softmax(scores, -1), memory)
        attn_h = torch.tanh(self.attn_out(torch.cat([h, ctx], -1)))
        return self.out(self.drop(attn_h)), attn_h, state

    def forward(
        self, src: torch.Tensor, src_mask: torch.Tensor, tgt_in: torch.Tensor
    ) -> torch.Tensor:
        """Teacher forcing. tgt_in (B, T) starts with BOS -> logits (B, T, V)."""
        memory, state = self.encode(src, src_mask)
        mask = src_mask.to(memory.dtype)
        attn_h = memory.new_zeros(src.size(0), self.hidden)
        logits = []
        for t in range(tgt_in.size(1)):
            step_logits, attn_h, state = self.decode_step(
                tgt_in[:, t : t + 1], attn_h, state, memory, mask
            )
            logits.append(step_logits)
        return torch.stack(logits, 1)

    # --- decoding interface (tel.decoding) ----------------------------------------

    def decoder_start(self, src: torch.Tensor, src_mask: torch.Tensor):
        memory, (h, c) = self.encode(src, src_mask)
        attn = memory.new_zeros(src.size(0), self.hidden)
        return memory, src_mask.to(memory.dtype), (attn, h, c)

    def decoder_step(self, prev: torch.Tensor, dstate, memory: torch.Tensor,
                     mask: torch.Tensor):
        attn, h, c = dstate
        logits, attn, (h, c) = self.decode_step(prev, attn, (h, c), memory, mask)
        return logits, (attn, h, c)

    @staticmethod
    def reorder_state(dstate, idx: torch.Tensor):
        attn, h, c = dstate
        return attn[idx], h[:, idx], c[:, idx]
