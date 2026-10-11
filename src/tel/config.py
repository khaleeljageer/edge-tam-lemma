"""Experiment configs: YAML files loaded into typed dataclasses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    type: str = "lstm_seq2seq"  # lstm_seq2seq | char_cnn | transformer | edit_classifier
    emb: int = 64
    hidden: int = 128  # lstm_seq2seq hidden size; transformer d_model
    layers: int = 1
    channels: int = 128  # edit_classifier
    kernel: int = 3  # edit_classifier, char_cnn
    conv_layers: int = 4  # char_cnn: dilated conv blocks in the encoder
    heads: int = 4  # transformer
    ff_mult: int = 4  # transformer: feed-forward width = ff_mult * d_model
    dropout: float = 0.1
    # If set, `hidden` / `channels` is chosen to get as close as possible below this.
    target_params: int | None = None


@dataclass
class DataConfig:
    dir: str = "data/processed"
    tokenizer: str = "codepoint"  # codepoint | grapheme
    max_len: int = 48
    max_train: int | None = None  # cap on train pairs (smoke tests)
    # Alternative training file (e.g. distilled data from tel.distill); dev/test and the
    # gold train lemmas used for seen/unseen scoring still come from `dir`.
    train_file: str | None = None


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 64
    lr: float = 2e-3
    weight_decay: float = 0.01
    patience: int = 10
    seed: int = 13
    label_smoothing: float = 0.0  # seq2seq cross-entropy
    warmup_steps: int = 0  # linear LR warmup over this many optimizer steps


@dataclass
class DecodeConfig:
    beam: int = 1  # final evaluation; dev selection during training is always greedy
    alpha: float = 0.0  # length normalization: score = logprob / (len + 1) ** alpha


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
    decode: DecodeConfig = field(default_factory=DecodeConfig)

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
    sections = {"model": ModelConfig, "data": DataConfig, "train": TrainConfig,
                "edit": EditConfig, "decode": DecodeConfig}
    unknown = set(d) - set(sections) - {"name"}
    if unknown:
        raise ValueError(f"unknown top-level keys: {sorted(unknown)}")
    return Config(
        name=d["name"],
        **{key: _build(cls, d.get(key), key) for key, cls in sections.items()},
    )


def load_config(path: str | Path) -> Config:
    return config_from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
