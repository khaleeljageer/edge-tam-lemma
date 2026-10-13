"""Put the manuscript's \\bibitem entries in order of first citation (Sadhana style).

    uv run python paper/order_refs.py            # rewrite paper/main.tex in place
    uv run python paper/order_refs.py --check    # exit 1 if the order is wrong

Fails on citations without a \\bibitem and on \\bibitem entries never cited.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

MAIN = Path(__file__).resolve().parent / "main.tex"
BIB = re.compile(r"(\\begin\{thebibliography\}\{[^}]*\}\n)(.*?)(\n\\end\{thebibliography\})", re.S)


def cited_keys(body: str) -> list[str]:
    """Citation keys in order of first use, ignoring commented-out text."""
    seen: list[str] = []
    for line in body.splitlines():
        line = re.sub(r"(?<!\\)%.*", "", line)
        for group in re.findall(r"\\cite\{([^}]*)\}", line):
            for key in (k.strip() for k in group.split(",")):
                if key not in seen:
                    seen.append(key)
    return seen


def split_items(block: str) -> dict[str, str]:
    """{key: entry text} for each \\bibitem in the bibliography block."""
    parts = re.split(r"(?=\\bibitem\{)", block.strip())
    items = {}
    for part in parts:
        m = re.match(r"\\bibitem\{([^}]*)\}", part)
        if m:
            items[m.group(1)] = part.strip()
    return items


def reorder(tex: str) -> str:
    m = BIB.search(tex)
    if not m:
        raise SystemExit("no thebibliography environment found")
    keys = cited_keys(tex[: m.start()])
    items = split_items(m.group(2))
    missing = [k for k in keys if k not in items]
    unused = [k for k in items if k not in keys]
    if missing or unused:
        raise SystemExit(f"cited without \\bibitem: {missing}; \\bibitem never cited: {unused}")
    block = "\n" + "\n\n".join(items[k] for k in keys) + "\n"
    return tex[: m.start()] + m.group(1) + block + m.group(3) + tex[m.end():]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tex", nargs="?", type=Path, default=MAIN)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    tex = args.tex.read_text(encoding="utf-8")
    new = reorder(tex)
    if args.check:
        sys.exit(0 if new == tex else 1)
    args.tex.write_text(new, encoding="utf-8")
    print(f"{len(cited_keys(new[: BIB.search(new).start()]))} references in citation order")


if __name__ == "__main__":
    main()
