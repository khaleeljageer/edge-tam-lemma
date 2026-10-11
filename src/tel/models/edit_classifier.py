"""Edit-tree classifier: char CNN encoder -> one edit-script class per word.

Single forward pass, no decoder. Positions are counted from the end of the
word, since Tamil inflection is almost entirely suffixal.
"""

from __future__ import annotations

import torch
from torch import nn

from tel.models.common import MASK_NEG
from tel.tokenizer import PAD_ID


class EditClassifier(nn.Module):
    def __init__(
        self, vocab_size: int, n_classes: int, emb: int = 64, channels: int = 128,
        layers: int = 3, kernel: int = 3, max_len: int = 48, dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=PAD_ID)
        self.rpos = nn.Embedding(max_len, emb)
        self.inp = nn.Conv1d(emb, channels, 1)
        self.convs = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel, padding=kernel // 2) for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(channels) for _ in range(layers))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * channels, n_classes)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """src (B, S), src_mask (B, S) bool -> class logits (B, C)."""
        mask = src_mask.to(self.rpos.weight.dtype)
        lengths = src_mask.sum(1, keepdim=True)
        rpos = lengths - 1 - torch.arange(src.size(1), device=src.device)[None, :]
        return self.forward_features(src, mask, rpos.clamp(0, self.max_len - 1))

    def forward_features(
        self, src: torch.Tensor, mask: torch.Tensor, rpos: torch.Tensor
    ) -> torch.Tensor:
        """Exported graph. src (B, S), mask (B, S) float 0/1, rpos (B, S) positions
        counted from the word end -> logits (B, C). Masks are applied
        arithmetically so the graph converts to every runtime, including ncnn."""
        x = self.drop(self.emb(src) + self.rpos(rpos))
        m = mask.unsqueeze(1)  # (B, 1, S)
        h = self.inp(x.transpose(1, 2)) * m
        for conv, norm in zip(self.convs, self.norms, strict=True):
            r = torch.relu(conv(self.drop(h)))
            h = norm((h + r).transpose(1, 2)).transpose(1, 2) * m
        h_max = (h + (m - 1.0) * MASK_NEG).amax(-1)
        h_mean = h.sum(-1) / m.sum(-1).clamp(min=1.0)
        return self.head(self.drop(torch.cat([h_max, h_mean], -1)))
