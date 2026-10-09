import pytest

from tel.baselines.identity import IdentityLemmatizer
from tel.data import Pair
from tel.evaluate import evaluate, score

GOLD = [
    Pair("மரங்கள்", "மரம்", "NOUN"),
    Pair("வீடு", "வீடு", "NOUN"),
    Pair("சென்றார்கள்", "செல்", "VERB"),
    Pair("மிக", "மிக", "OTHER"),
]


def test_score_breakdown():
    res = score(GOLD, ["மரம்", "வீடா", "செல்", "மிக"], train_lemmas={"மரம்", "செல்"})
    m = res.metrics()
    assert m["acc"] == pytest.approx(0.75)
    assert m["acc_noun"] == pytest.approx(0.5)
    assert m["acc_verb"] == 1.0
    assert m["acc_other"] == 1.0
    assert m["acc_seen"] == 1.0
    assert m["acc_unseen"] == pytest.approx(0.5)
    assert [(p.word, pred) for p, pred in res.errors] == [("வீடு", "வீடா")]


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        score(GOLD, ["x"])


def test_identity_baseline():
    assert evaluate(IdentityLemmatizer(), GOLD).metrics()["acc"] == pytest.approx(0.5)
