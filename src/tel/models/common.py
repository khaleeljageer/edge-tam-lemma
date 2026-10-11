"""Helpers shared by all model families."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

# Additive mask for padded positions: x + (mask - 1) * MASK_NEG. Must stay inside
# the fp16 range (max 65504): with 1e9 the constant becomes inf in fp16 graphs and
# 0 * inf = NaN at every real position (seen with ncnn fp16 on Cortex-A76).
# 1e4 is still far below any real activation, so fp32 results are unchanged.
MASK_NEG = 1e4
FP16_MAX = 65504.0


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def width_for_params(
    build: Callable[[int], nn.Module], target: int, lo: int = 8, hi: int = 2048
) -> int:
    """Largest width w in [lo, hi] such that build(w) has at most `target` parameters.

    `build` must be monotonic in w. Used to hit the 0.5M / 1M / 2M / 5M targets.
    """
    if count_params(build(lo)) > target:
        return lo
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_params(build(mid)) <= target:
            lo = mid
        else:
            hi = mid - 1
    return lo


def lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """(B,) lengths -> (B, T) bool mask, True at real tokens."""
    return torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]
