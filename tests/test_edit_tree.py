import pytest

from tel.edit_tree import IDENTITY, EditScript, ScriptVocab, extract

PAIRS = [
    ("சென்றார்கள்", "செல்"),
    ("மரங்கள்", "மரம்"),
    ("மரத்தில்", "மரம்"),
    ("வீட்டை", "வீடு"),
    ("வீடு", "வீடு"),
    ("பூக்கள்", "பூ"),
    ("நாய்கள்", "நாய்"),
    ("அ", "இ"),  # nothing shared: whole-word replacement
]


@pytest.mark.parametrize(("word", "lemma"), PAIRS)
def test_extract_apply_roundtrip(word, lemma):
    assert extract(word, lemma).apply(word) == lemma


def test_identity_script():
    assert extract("வீடு", "வீடு") == IDENTITY


def test_script_generalizes_to_new_word():
    script = extract("நாய்கள்", "நாய்")
    assert script == EditScript(suf_del="கள்")
    assert script.apply("பசுக்கள்") == "பசுக்"  # applies, though wrong lemma here
    assert script.apply("வீடு") is None


def test_parse_roundtrip():
    s = extract("சென்றார்கள்", "செல்")
    assert EditScript.parse(str(s)) == s


def test_vocab_identity_first_and_unknown_maps_to_zero():
    vocab = ScriptVocab.build(PAIRS)
    assert vocab.itos[0] == IDENTITY
    assert vocab.id_of(EditScript(suf_del="xyz")) == 0


def test_vocab_top_k_and_coverage():
    pairs = [("நாய்கள்", "நாய்"), ("மாடுகள்", "மாடு"), ("மரங்கள்", "மரம்")]
    vocab = ScriptVocab.build(pairs, top_k=1)
    assert len(vocab) == 2  # identity + "-கள்"
    assert vocab.coverage(pairs) == pytest.approx(2 / 3)


def test_applicable_mask():
    vocab = ScriptVocab.build([("நாய்கள்", "நாய்")])
    assert vocab.applicable_mask("மாடுகள்") == [True, True]
    assert vocab.applicable_mask("வீடு") == [True, False]


def test_vocab_save_load(tmp_path):
    vocab = ScriptVocab.build(PAIRS)
    vocab.save(str(tmp_path / "scripts.txt"))
    assert ScriptVocab.load(str(tmp_path / "scripts.txt")).itos == vocab.itos


def test_applicable_ids_match_bruteforce():
    vocab = ScriptVocab.build(PAIRS)
    for word in ["மரங்கள்", "வீட்டை", "சென்றார்கள்", "அ", "பூ", "நாய்களை"]:
        expected = [i for i, s in enumerate(vocab.itos) if s.applies(word)]
        assert vocab.applicable_ids(word) == expected
