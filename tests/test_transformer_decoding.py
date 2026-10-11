import pytest
import torch

from tel.decoding import beam_decode, greedy_decode
from tel.inference import Seq2SeqLemmatizer, encode_batch
from tel.models.common import count_params, width_for_params
from tel.models.lstm_seq2seq import LSTMSeq2Seq
from tel.models.transformer import CharTransformer
from tel.tokenizer import CharTokenizer
from tel.train import Task
from tests.test_models import PAIRS, _overfit


def _tok():
    return CharTokenizer.build(w for p in PAIRS for w in p)


@pytest.mark.parametrize("target", [500_000, 2_000_000])
def test_transformer_width_hits_param_target(target):
    def make(u):
        return CharTransformer(80, d_model=4 * u, heads=4, layers=2)
    params = count_params(make(width_for_params(make, target, lo=1, hi=256)))
    assert 0.9 * target <= params <= target


def test_transformer_output_does_not_depend_on_padding():
    torch.manual_seed(0)
    model = CharTransformer(30, d_model=32, heads=4, layers=2).eval()
    src = torch.tensor([[5, 6, 7, 8, 9]])
    padded = torch.cat([src, torch.zeros(1, 20, dtype=torch.long)], 1)
    with torch.no_grad():
        a = model.encode(src, src != 0)
        b = model.encode(padded, padded != 0)
        torch.testing.assert_close(a, b[:, :5])
        tgt = torch.tensor([[1, 5, 6]])
        torch.testing.assert_close(model(src, src != 0, tgt), model(padded, padded != 0, tgt))


def test_transformer_overfits_small_set():
    torch.manual_seed(0)
    tok = _tok()
    model = CharTransformer(len(tok), d_model=64, heads=4, layers=2, dropout=0.0)
    assert _overfit(model, Task(tok, 48), steps=300) == [lem for _, lem in PAIRS]


@pytest.mark.parametrize("cls", ["lstm", "transformer"])
def test_beam1_equals_greedy(cls):
    torch.manual_seed(0)
    tok = _tok()
    model = (LSTMSeq2Seq(len(tok), emb=16, hidden=32) if cls == "lstm"
             else CharTransformer(len(tok), d_model=32, heads=4)).eval()
    src, mask = encode_batch(tok, [w for w, _ in PAIRS], 48)
    greedy = [tok.decode(r.tolist()) for r in greedy_decode(model, src, mask, 20)]
    beam = [tok.decode(h[0].tokens) for h in beam_decode(model, src, mask, 20, beam=1)]
    assert beam == greedy


@pytest.mark.parametrize("cls", ["lstm", "transformer"])
def test_beam_after_overfit(cls):
    torch.manual_seed(0)
    tok = _tok()
    model = (LSTMSeq2Seq(len(tok), emb=32, hidden=64, dropout=0.0) if cls == "lstm"
             else CharTransformer(len(tok), d_model=64, heads=4, dropout=0.0))
    _overfit(model, Task(tok, 48), steps=300)
    lem = Seq2SeqLemmatizer(model, tok, 48, beam=4)
    words = [w for w, _ in PAIRS]
    assert lem.lemmatize(words) == [lem_ for _, lem_ in PAIRS]
    for hyps in lem.nbest(words, n=4):
        assert len(hyps) == 4
        scores = [s for _, s in hyps]
        assert scores == sorted(scores, reverse=True)  # alpha=0: ranked by logprob


def test_decode_step_fixed_matches_decoder_step():
    torch.manual_seed(0)
    tok = _tok()
    model = CharTransformer(len(tok), d_model=32, heads=4, layers=2, max_len=24).eval()
    src, mask = encode_batch(tok, ["மரங்கள்", "வீடு"], 24)
    with torch.no_grad():
        memory, fmask, prefix = model.decoder_start(src, mask)
        prefix_fixed = torch.zeros(2, model.prefix_len, dtype=torch.long)
        prev = torch.full((2, 1), 1, dtype=torch.long)  # BOS
        for t in range(5):
            logits, prefix = model.decoder_step(prev, prefix, memory, fmask)
            prefix_fixed[:, t] = prev[:, 0]
            sel = torch.zeros(2, model.prefix_len)
            sel[:, t] = 1.0
            fixed = model.decode_step_fixed(prefix_fixed, memory, fmask, sel)
            torch.testing.assert_close(fixed, logits, rtol=1e-4, atol=1e-4)
            prev = logits.argmax(-1, keepdim=True)
