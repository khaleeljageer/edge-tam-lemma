import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "wiki_freq", Path(__file__).parents[1] / "scripts/data/wiki_freq.py"
)
wiki = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wiki)


def test_links_keep_shown_text_not_target():
    text = "[[மரம்|மரங்கள்]] மற்றும் [[வீடு]]"
    assert wiki.tamil_tokens(wiki.clean_wikitext(text)) == ["மரங்கள்", "மற்றும்", "வீடு"]


def test_templates_refs_files_removed():
    text = ("அவன் {{Infobox|பெயர்={{nested|x}}}} வந்தான்<ref>மூலம் நூல்</ref> "
            "[[படிமம்:படம்.jpg|thumb|விளக்கம்]] ஊர்<!-- குறிப்பு -->")
    assert wiki.tamil_tokens(wiki.clean_wikitext(text)) == ["அவன்", "வந்தான்", "ஊர்"]


def test_tokens_split_on_non_tamil_and_normalized():
    assert wiki.tamil_tokens("தமிழ்,English 2024 மொழி") == ["தமிழ்", "மொழி"]
    assert wiki.tamil_tokens("கொ") == ["கொ"]  # decomposed -> NFC
