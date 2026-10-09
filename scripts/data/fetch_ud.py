"""Download the UD Tamil treebanks (TTB, MWTT) into data/raw/ud/.

Pinned to a git ref so the data is reproducible; change UD_REF deliberately.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

UD_REF = "master"  # TODO: pin to a UD release tag (e.g. r2.15) once chosen
BASE = "https://raw.githubusercontent.com/UniversalDependencies"

FILES = {
    "UD_Tamil-TTB": [
        "ta_ttb-ud-train.conllu",
        "ta_ttb-ud-dev.conllu",
        "ta_ttb-ud-test.conllu",
        "LICENSE.txt",
        "README.md",
    ],
    "UD_Tamil-MWTT": ["ta_mwtt-ud-test.conllu", "LICENSE.txt", "README.md"],
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/raw/ud")
    ap.add_argument("--ref", default=UD_REF)
    args = ap.parse_args()

    for repo, names in FILES.items():
        dest = Path(args.out) / repo
        dest.mkdir(parents=True, exist_ok=True)
        for name in names:
            url = f"{BASE}/{repo}/{args.ref}/{name}"
            print(f"fetch {url}")
            urllib.request.urlretrieve(url, dest / name)


if __name__ == "__main__":
    main()
