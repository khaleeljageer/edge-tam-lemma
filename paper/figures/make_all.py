"""Regenerate every paper figure and table from bench/results/.

    uv run python paper/figures/make_all.py
"""

from __future__ import annotations

import fig_params
import fig_pareto
import tables
from common import OUT

if __name__ == "__main__":
    for make in (fig_params.main, fig_pareto.main, tables.main):
        for path in make(OUT):
            print(path.relative_to(OUT.parents[2]))
