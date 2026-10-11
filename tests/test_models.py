import pytest
import torch

from tel.edit_tree import ScriptVocab
from tel.inference import EditLemmatizer, Seq2SeqLemmatizer
from tel.models.common import count_params, width_for_params
from tel.models.edit_classifier import EditClassifier
from tel.models.lstm_seq2seq import LSTMSeq2Seq
from tel.tokenizer import CharTokenizer
from tel.train import Task

PAIRS = [
    ("மரங்கள்", "மரம்"), ("மரத்தில்", "மரம்"), ("வீட்டை", "வீடு"), ("வீடு", "வீடு"),
    ("பூக்கள்", "பூ"), ("நாய்கள்", "நாய்"), ("சென்றார்கள்", "செல்"), ("வந்தான்", "வா"),
    ("படித்தாள்", "படி"), ("கொண்டு", "கொள்"), ("நாட்டில்", "நாடு"), ("அவனை", "அவன்"),
]


@pytest.mark.parametrize("target", [500_000, 1_000_000, 2_000_000])
def test_width_hits_param_target(target):
    def make(h):
        return LSTMSeq2Seq(vocab_size=80, emb=64, hidden=h, layers=2)
    params = count_params(make(width_for_params(make, target)))
    assert 0.9 * target <= params <= target


def test_edit_classifier_width_hits_param_target():
    def make(c):
        return EditClassifier(vocab_size=80, n_classes=600, channels=c)
    params = count_params(make(width_for_params(make, 1_000_000)))
    assert 0.9 * 1_000_000 <= params <= 1_000_000


def _overfit(model, task, steps=300):
    batch = [type("P", (), {"word": w, "lemma": lem})() for w, lem in PAIRS]
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(steps):
        loss = task.loss(model, batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return task.lemmatizer(model).lemmatize([w for w, _ in PAIRS])


def test_lstm_overfits_small_set():
    torch.manual_seed(0)
    tok = CharTokenizer.build(w for p in PAIRS for w in p)
    model = LSTMSeq2Seq(len(tok), emb=32, hidden=64, dropout=0.0)
    assert _overfit(model, Task(tok, 48)) == [lem for _, lem in PAIRS]


def test_edit_classifier_overfits_small_set():
    torch.manual_seed(0)
    tok = CharTokenizer.build(w for p in PAIRS for w in p)
    vocab = ScriptVocab.build(PAIRS)
    model = EditClassifier(len(tok), len(vocab), emb=32, channels=64, dropout=0.0)
    assert _overfit(model, Task(tok, 48, vocab)) == [lem for _, lem in PAIRS]


def test_lemmatizers_keep_length_and_handle_unknown_chars():
    tok = CharTokenizer.build(w for p in PAIRS for w in p)
    vocab = ScriptVocab.build(PAIRS)
    words = ["மரங்கள்", "ஔஷதம்", "abc"]
    s2s = Seq2SeqLemmatizer(LSTMSeq2Seq(len(tok)), tok, 48)
    edit = EditLemmatizer(EditClassifier(len(tok), len(vocab)), tok, vocab, 48)
    assert len(s2s.lemmatize(words)) == 3
    assert len(edit.lemmatize(words)) == 3


def test_reversal_matrix():
    from tel.models.lstm_seq2seq import reversal_matrix

    x = torch.arange(10.0).view(2, 5, 1)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.float)
    p = reversal_matrix(mask)
    r = torch.matmul(p, x)
    assert r[0, :, 0].tolist() == [2, 1, 0, 3, 4]
    assert r[1, :, 0].tolist() == [9, 8, 7, 6, 5]
    assert torch.equal(torch.matmul(p, r), x)


@pytest.mark.parametrize("model_type", ["lstm", "edit"])
def test_output_does_not_depend_on_padding(model_type):
    torch.manual_seed(0)
    if model_type == "lstm":
        model = LSTMSeq2Seq(30, emb=16, hidden=32, layers=2).eval()
    else:
        model = EditClassifier(30, 10, emb=16, channels=32).eval()
    src = torch.tensor([[5, 6, 7, 8, 9]])
    padded = torch.cat([src, torch.zeros(1, 20, dtype=torch.long)], 1)
    with torch.no_grad():
        if model_type == "lstm":
            a, _ = model.encode(src, src != 0)
            b, _ = model.encode(padded, padded != 0)
            torch.testing.assert_close(a, b[:, :5])
        else:
            torch.testing.assert_close(model(src, src != 0), model(padded, padded != 0))


def test_char_cnn_padding_invariance_and_overfit():
    torch.manual_seed(0)
    model = LSTMSeq2Seq(30, emb=16, hidden=32, layers=1, encoder="cnn").eval()
    assert not model.needs_perm
    src = torch.tensor([[5, 6, 7, 8, 9]])
    padded = torch.cat([src, torch.zeros(1, 20, dtype=torch.long)], 1)
    with torch.no_grad():
        a, _ = model.encode(src, src != 0)
        b, _ = model.encode(padded, padded != 0)
    torch.testing.assert_close(a, b[:, :5])

    tok = CharTokenizer.build(w for p in PAIRS for w in p)
    cnn = LSTMSeq2Seq(len(tok), emb=32, hidden=64, dropout=0.0, encoder="cnn")
    assert _overfit(cnn, Task(tok, 48)) == [lem for _, lem in PAIRS]


def test_char_cnn_width_hits_param_target():
    def make(h):
        return LSTMSeq2Seq(80, emb=64, hidden=h, layers=1, encoder="cnn")
    params = count_params(make(width_for_params(make, 1_000_000)))
    assert 0.9 * 1_000_000 <= params <= 1_000_000
