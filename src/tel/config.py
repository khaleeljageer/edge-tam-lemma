"""Experiment configs: YAML files loaded into typed dataclasses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    type: str = "lstm_seq2seq"  # lstm_seq2seq | edit_classifier
    emb: int = 64
    hidden: int = 128  # lstm_seq2seq
    layers: int = 1
    channels: int = 128  # edit_classifier
    kernel: int = 3  # edit_classifier
    dropout: float = 0.1
    # If set, `hidden` / `channels` is chosen to get as close as possible below this.
    target_params: int | None = None


@dataclass
class DataConfig:
    dir: str = "data/processed"
    tokenizer: str = "codepoint"  # codepoint | grapheme
    max_len: int = 48
    max_train: int | None = None  # cap on train pairs (smoke tests)


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 64
    lr: float = 2e-3
    weight_decay: float = 0.01
    patience: int = 10
    seed: int = 13


@dataclass
class EditConfig:
    top_k: int | None = None  # keep the K most frequent scripts; None = all in train


@dataclass
class Config:
    name: str
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    edit: EditConfig = field(default_factory=EditConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build(cls: type, values: dict[str, Any] | None, where: str) -> Any:
    values = values or {}
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown keys in {where}: {sorted(unknown)}")
    return cls(**values)


def config_from_dict(d: dict[str, Any]) -> Config:
    d = dict(d)
    sections = {"model": ModelConfig, "data": DataConfig, "train": TrainConfig, "edit": EditConfig}
    unknown = set(d) - set(sections) - {"name"}
    if unknown:
        raise ValueError(f"unknown top-level keys: {sorted(unknown)}")
    return Config(
        name=d["name"],
        **{key: _build(cls, d.get(key), key) for key, cls in sections.items()},
    )


def load_config(path: str | Path) -> Config:
    return config_from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
