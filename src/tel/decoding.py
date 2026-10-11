"""Greedy and beam-search decoding for any seq2seq model (PyTorch side).

A model plugs in by implementing:

    decoder_start(src, src_mask) -> (memory, mask, dstate)
    decoder_step(prev (B, 1), dstate, memory, mask) -> (logits (B, V), dstate)
    reorder_state(dstate, idx) -> dstate         # select / repeat batch rows

`mask` is the float (B, S) padding mask. Exported graphs keep their own host
loop (`tel.runtime.host`); this module serves training-time evaluation, the
teacher, and distillation targets (n-best lists with scores).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from tel.tokenizer import BOS_ID, EOS_ID, PAD_ID


@torch.no_grad()
def greedy_decode(model, src: torch.Tensor, src_mask: torch.Tensor, max_len: int) -> torch.Tensor:
    """-> (B, <=max_len) token ids, EOS-terminated, then PAD."""
    memory, mask, dstate = model.decoder_start(src, src_mask)
    prev = torch.full((src.size(0), 1), BOS_ID, dtype=torch.long, device=src.device)
    done = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
    out = []
    for _ in range(max_len):
        logits, dstate = model.decoder_step(prev, dstate, memory, mask)
        tok = logits.argmax(-1).masked_fill(done, PAD_ID)
        out.append(tok)
        done |= tok == EOS_ID
        if done.all():
            break
        prev = tok[:, None]
    return torch.stack(out, 1)


@dataclass
class Hypothesis:
    tokens: list[int]  # without BOS, without EOS
    logprob: float  # sum of token log-probs, including EOS
    score: float  # length-normalized score used for ranking


@torch.no_grad()
def beam_decode(
    model, src: torch.Tensor, src_mask: torch.Tensor, max_len: int, beam: int = 4,
    alpha: float = 0.0,
) -> list[list[Hypothesis]]:
    """Batched beam search. Returns, per input, `beam` hypotheses best first.

    Ranking score = logprob / (length + 1) ** alpha (alpha = 0: plain logprob).
    Hypotheses that never emit EOS within max_len are kept, ranked last.
    """
    b = src.size(0)
    k = beam
    memory, mask, dstate = model.decoder_start(src, src_mask)
    rep = torch.arange(b, device=src.device).repeat_interleave(k)  # (B*K,)
    memory, mask, dstate = memory[rep], mask[rep], model.reorder_state(dstate, rep)

    # Only the first beam of each input is live at the start.
    scores = torch.full((b, k), float("-inf"), device=src.device)
    scores[:, 0] = 0.0
    scores = scores.view(-1)
    prev = torch.full((b * k, 1), BOS_ID, dtype=torch.long, device=src.device)
    history = torch.zeros((b * k, 0), dtype=torch.long, device=src.device)
    finished = torch.zeros(b * k, dtype=torch.bool, device=src.device)

    for _ in range(max_len):
        logits, dstate = model.decoder_step(prev, dstate, memory, mask)
        logp = F.log_softmax(logits.float(), -1)
        # A finished beam can only continue with PAD at no cost.
        logp[finished] = float("-inf")
        logp[finished, PAD_ID] = 0.0
        total = (scores[:, None] + logp).view(b, -1)  # (B, K*V)
        top, idx = total.topk(k, dim=-1)
        vocab = logp.size(-1)
        src_beam = (idx // vocab) + torch.arange(b, device=src.device)[:, None] * k
        src_beam, token = src_beam.view(-1), (idx % vocab).view(-1)

        scores = top.view(-1)
        history = torch.cat([history[src_beam], token[:, None]], 1)
        finished = finished[src_beam] | (token == EOS_ID)
        dstate = model.reorder_state(dstate, src_beam)
        prev = token[:, None]
        if finished.all():
            break

    results: list[list[Hypothesis]] = []
    for i in range(b):
        hyps = []
        for j in range(i * k, (i + 1) * k):
            seq = history[j].tolist()
            ended = EOS_ID in seq
            toks = seq[: seq.index(EOS_ID)] if ended else [t for t in seq if t != PAD_ID]
            lp = float(scores[j])
            norm = lp / ((len(toks) + 1) ** alpha) if alpha else lp
            hyps.append(Hypothesis(toks, lp, norm if ended else float("-inf")))
        hyps.sort(key=lambda h: h.score, reverse=True)
        results.append(hyps)
    return results
