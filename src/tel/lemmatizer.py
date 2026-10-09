"""Common interface shared by models, baselines and exported runtimes."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Lemmatizer(Protocol):
    def lemmatize(self, words: list[str]) -> list[str]: ...
