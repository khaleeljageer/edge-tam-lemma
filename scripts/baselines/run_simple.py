"""In-repo baselines: identity (lower bound) and the hand-written suffix stripper.

    uv run python scripts/baselines/run_simple.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import run_baseline  # noqa: E402

from tel.baselines.identity import IdentityLemmatizer  # noqa: E402
from tel.baselines.suffix_stripper import SuffixStripper  # noqa: E402

if __name__ == "__main__":
    run_baseline("identity", IdentityLemmatizer().lemmatize, "lower bound")
    run_baseline("suffix_stripper", SuffixStripper().lemmatize, "hand-written rules")
