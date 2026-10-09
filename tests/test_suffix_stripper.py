import pytest

from tel.baselines.suffix_stripper import SuffixStripper, lemmatize_word


@pytest.mark.parametrize(
    ("word", "lemma"),
    [
        ("மரங்கள்", "மரம்"),
        ("மரங்களை", "மரம்"),
        ("மரத்தில்", "மரம்"),
        ("வீட்டை", "வீடு"),
        ("பூக்கள்", "பூ"),
        ("புத்தகங்களை", "புத்தகம்"),
        ("நாய்கள்", "நாய்"),
        ("நான்", "நான்"),  # too short to strip
        ("வீடு", "வீடு"),
    ],
)
def test_lemmatize_word(word, lemma):
    assert lemmatize_word(word) == lemma


def test_batch_keeps_length():
    words = ["மரங்கள்", "வீடு", "நான்"]
    assert len(SuffixStripper().lemmatize(words)) == len(words)
