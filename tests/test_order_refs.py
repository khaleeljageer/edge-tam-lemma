import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper"))

from order_refs import MAIN, reorder  # noqa: E402

TEX = r"""\cite{b} and \cite{a, b}. % \cite{c}
\begin{thebibliography}{99}
\bibitem{a} A 2020 Title.

\bibitem{b} B 2021 Title.
\end{thebibliography}
"""


def test_reorders_by_first_citation_ignoring_comments():
    out = reorder(TEX)
    assert out.index(r"\bibitem{b}") < out.index(r"\bibitem{a}")
    assert reorder(out) == out


def test_rejects_missing_and_unused_entries():
    with pytest.raises(SystemExit):
        reorder(TEX.replace(r"\cite{a, b}", r"\cite{b, z}"))
    with pytest.raises(SystemExit):
        reorder(TEX.replace(r"\cite{a, b}", r"\cite{b}"))


def test_manuscript_references_in_citation_order():
    tex = MAIN.read_text(encoding="utf-8")
    assert reorder(tex) == tex
