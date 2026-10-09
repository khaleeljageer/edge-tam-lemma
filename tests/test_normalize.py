import unicodedata

from tel.normalize import clean_word, is_tamil_word, normalize, tamil_words


def test_composed_and_decomposed_agree():
    composed = "கொ"  # U+0B95 U+0BCA
    decomposed = "கொ"  # க + ெ + ா
    assert normalize(decomposed) == composed
    assert unicodedata.is_normalized("NFC", normalize(decomposed))


def test_invisible_characters_removed():
    assert normalize("தமி‍ழ்") == "தமிழ்"


def test_is_tamil_word():
    assert is_tamil_word("தமிழ்")
    assert not is_tamil_word("தமிழ்2")
    assert not is_tamil_word("Tamil")
    assert not is_tamil_word("")
    assert not is_tamil_word("௧௨")  # Tamil digits are not letters


def test_clean_word_strips_punctuation_and_latin():
    assert clean_word("“வீடு,”") == "வீடு"
    assert clean_word("abcவீடுxyz") == "வீடு"
    assert clean_word("123") == ""


def test_tamil_words_mixed_script():
    text = "நான் school-க்கு சென்றேன். 2024 ஆண்டு!"
    assert tamil_words(text) == ["நான்", "க்கு", "சென்றேன்", "ஆண்டு"]
