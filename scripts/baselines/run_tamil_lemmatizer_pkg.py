"""Baseline: the `tamil-lemmatizer` PyPI package (Hemanth Kumar, MIT), used as a
black-box dependency. Each word is lemmatized separately; an exception or an
empty result counts as a crash (research plan section 4.1).

    uv run --project envs/baselines python scripts/baselines/run_tamil_lemmatizer_pkg.py
"""

from __future__ import annotations

import sys
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import run_baseline  # noqa: E402

from tel.normalize import normalize  # noqa: E402


def main() -> None:
    from tamil_lemmatizer import TamilLemmatizer

    lem = TamilLemmatizer()

    def lemmatize(words: list[str]) -> list[str | None]:
        out: list[str | None] = []
        for w in words:
            try:
                result = lem.lemmatize(w)
                out.append(normalize(result) if isinstance(result, str) and result else None)
            except Exception:  # noqa: BLE001 - any failure of the package is a crash
                out.append(None)
        return out

    run_baseline("tamil_lemmatizer_pkg", lemmatize,
                 f"pypi tamil-lemmatizer {version('tamil-lemmatizer')}; training data unknown")


if __name__ == "__main__":
    main()
