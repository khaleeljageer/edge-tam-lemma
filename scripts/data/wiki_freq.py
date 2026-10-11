"""Word-frequency list from a pinned Tamil Wikipedia dump (CC BY-SA).

    uv run python scripts/data/wiki_freq.py            # download + count
    uv run python scripts/data/wiki_freq.py --limit 2000   # quick test on 2000 articles

Streams the pages-articles dump (no full decompression), keeps namespace-0
articles, strips the wiki markup that would otherwise glue words together
(link targets, templates, refs, tables, tags), and counts NFC-normalized words
made only of Tamil letters. Writes data/interim/wiki_freq.tsv (word, count),
most frequent first, plus a small stats file.
"""

from __future__ import annotations

import argparse
import bz2
import re
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from tel.normalize import normalize

DUMP_DATE = "20261001"
URL = (f"https://dumps.wikimedia.org/tawiki/{DUMP_DATE}/"
       f"tawiki-{DUMP_DATE}-pages-articles.xml.bz2")
RAW = Path("data/raw/wiki") / f"tawiki-{DUMP_DATE}-pages-articles.xml.bz2"
OUT = Path("data/interim/wiki_freq.tsv")
USER_AGENT = "tel-tamil-edge-lemmatizer/0.1 (research; word-frequency list) python-urllib"

_TAMIL_RUN = re.compile("[ஂ-்ௐௗ]+")
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_REF = re.compile(r"<ref[^>/]*/>|<ref[^>]*>.*?</ref>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")
_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")  # innermost; applied repeatedly
_TABLE = re.compile(r"\{\|.*?\|\}", re.S)
_FILE_LINK = re.compile(r"\[\[(?:படிமம்|கோப்பு|File|Image|பகுப்பு|Category):[^\[\]]*\]\]", re.I)
_LINK = re.compile(r"\[\[(?:[^\[\]|]*\|)?([^\[\]]*)\]\]")  # keep the shown text, drop the target
_EXT_LINK = re.compile(r"\[https?://\S+\s*([^\]]*)\]")


def clean_wikitext(text: str) -> str:
    text = _COMMENT.sub(" ", text)
    text = _REF.sub(" ", text)
    for _ in range(5):  # nested templates
        text, n = _TEMPLATE.subn(" ", text)
        if not n:
            break
    text = _TABLE.sub(" ", text)
    text = _FILE_LINK.sub(" ", text)
    text = _LINK.sub(r"\1", text)
    text = _EXT_LINK.sub(r"\1", text)
    return _TAG.sub(" ", text)


def tamil_tokens(text: str) -> list[str]:
    """Runs of Tamil letters after NFC; a run may not start with a combining sign."""
    words = _TAMIL_RUN.findall(normalize(text))
    return [w for w in words if not ("ா" <= w[0] <= "்" or w[0] == "ௗ")]


def iter_articles(path: Path, limit: int | None = None):
    """Yield the wikitext of namespace-0, non-redirect pages."""
    count = 0
    with bz2.open(path, "rb") as f:
        ns, title_is_redirect = None, False
        for _, elem in ET.iterparse(f, events=("end",)):
            tag = elem.tag.rsplit("}", 1)[-1]
            if tag == "ns":
                ns = elem.text
            elif tag == "redirect":
                title_is_redirect = True
            elif tag == "text":
                if ns == "0" and not title_is_redirect and elem.text:
                    yield elem.text
                    count += 1
                    if limit and count >= limit:
                        return
            elif tag == "page":
                ns, title_is_redirect = None, False
                elem.clear()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, help="only the first N articles (testing)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    if not RAW.exists():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {URL}")
        # Wikimedia rejects urllib's default User-Agent (HTTP 403); identify the client.
        req = urllib.request.Request(URL, headers={"User-Agent": USER_AGENT})
        tmp = RAW.with_suffix(".part")
        with urllib.request.urlopen(req) as resp, open(tmp, "wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
        tmp.rename(RAW)  # only a complete download gets the final name

    counts: Counter[str] = Counter()
    articles = 0
    for text in iter_articles(RAW, args.limit):
        counts.update(tamil_tokens(clean_wikitext(text)))
        articles += 1
        if articles % 20000 == 0:
            print(f"{articles} articles, {len(counts):,} word types", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("word\tcount\n")
        f.writelines(f"{w}\t{c}\n" for w, c in counts.most_common())
    tokens = sum(counts.values())
    stats = (f"dump\t{DUMP_DATE}\narticles\t{articles}\ntokens\t{tokens}\n"
             f"types\t{len(counts)}\nlicense\tCC BY-SA (Wikipedia)\n")
    args.out.with_suffix(".stats.tsv").write_text(stats, encoding="utf-8")
    print(stats)


if __name__ == "__main__":
    main()
