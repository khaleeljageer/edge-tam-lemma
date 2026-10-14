import pytest

from tel.error_analysis import categorize, graphemes


def test_graphemes_keep_vowel_signs_with_their_letter():
    assert graphemes("அந்தஸ்து") == ["அ", "ந்", "த", "ஸ்", "து"]
    assert graphemes("ab") == ["a", "b"]


@pytest.mark.parametrize("word, gold, pred, expected", [
    ("தாழ்வாரமோ", "தாழ்வாரம்", "தாழ்வாரமோ", "unchanged"),
    ("அத்துடன்", "அத்துடன்", "அம்", "gold_is_form"),
    ("வாழ்வதோ", "வாழ்", "வாழ்வது", "suffix_left"),
    ("விரோதச்", "விரோதம்", "விரோத", "over_stripped"),
    ("அந்தஸ்தைப்", "அந்தஸ்து", "அந்தஸ்தை", "stem_final"),
    ("உட்காருகின்றான்", "உட்கார்", "உட்காருகில்", "wrong_stem"),
])
def test_categories(word, gold, pred, expected):
    assert categorize(word, gold, pred) == expected


def test_correct_prediction_is_not_an_error():
    with pytest.raises(ValueError):
        categorize("மனைவியும்", "மனைவி", "மனைவி")
