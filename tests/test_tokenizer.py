from tel.tokenizer import BOS_ID, EOS_ID, UNK_ID, CharTokenizer, graphemes

WORDS = ["சென்றார்கள்", "செல்", "கொண்டு", "வீட்டை", "வீடு", "மரங்கள்"]


def test_graphemes_keep_vowel_signs_and_virama():
    assert graphemes("கொ") == ["கொ"]
    assert graphemes("தமிழ்") == ["த", "மி", "ழ்"]
    assert graphemes("அம்மா") == ["அ", "ம்", "மா"]


def test_codepoint_roundtrip():
    tok = CharTokenizer.build(WORDS)
    for w in WORDS:
        assert tok.decode(tok.encode(w, add_bos=True, add_eos=True)) == w


def test_grapheme_roundtrip_and_smaller_sequences():
    cp = CharTokenizer.build(WORDS, mode="codepoint")
    gr = CharTokenizer.build(WORDS, mode="grapheme")
    for w in WORDS:
        assert gr.decode(gr.encode(w)) == w
        assert len(gr.encode(w)) <= len(cp.encode(w))


def test_bos_eos_and_unknown():
    tok = CharTokenizer.build(["அ"])
    ids = tok.encode("அஆ", add_bos=True, add_eos=True)
    assert ids[0] == BOS_ID and ids[-1] == EOS_ID
    assert ids[2] == UNK_ID


def test_decode_stops_at_eos():
    tok = CharTokenizer.build(["அஆ"])
    a, b = tok.encode("அஆ")
    assert tok.decode([BOS_ID, a, EOS_ID, b]) == "அ"


def test_save_load(tmp_path):
    tok = CharTokenizer.build(WORDS, mode="grapheme")
    tok.save(tmp_path / "vocab.json")
    loaded = CharTokenizer.load(tmp_path / "vocab.json")
    assert loaded.itos == tok.itos and loaded.mode == "grapheme"
