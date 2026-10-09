"""Train one model from a YAML config.

    uv run python -m tel.train configs/lstm_tiny.yaml

Writes runs/<name>/{config.yaml, tokenizer.json, [scripts.txt], model.pt,
metrics.json, test_errors.tsv}. Early stopping on dev exact-match accuracy.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch import nn

from tel.config import Config, load_config
from tel.data import Pair, read_tsv
from tel.edit_tree import ScriptVocab, extract
from tel.evaluate import score, write_errors
from tel.inference import EditLemmatizer, Seq2SeqLemmatizer, applicable_masks, encode_batch
from tel.lemmatizer import Lemmatizer
from tel.models.common import count_params, width_for_params
from tel.models.edit_classifier import EditClassifier
from tel.models.lstm_seq2seq import LSTMSeq2Seq
from tel.tokenizer import PAD_ID, CharTokenizer

EVAL_SPLITS = ("dev", "test", "test_unseen")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# --- model construction ------------------------------------------------------


def model_kwargs(cfg: Config, vocab_size: int, n_classes: int) -> dict[str, Any]:
    """Constructor kwargs for the configured model, resolving `target_params`."""
    m = cfg.model
    if m.type == "lstm_seq2seq":
        def make(hidden: int) -> dict[str, Any]:
            return dict(vocab_size=vocab_size, emb=m.emb, hidden=hidden, layers=m.layers,
                        dropout=m.dropout)
        width_key, cls = "hidden", LSTMSeq2Seq
    elif m.type == "edit_classifier":
        def make(channels: int) -> dict[str, Any]:
            return dict(vocab_size=vocab_size, n_classes=n_classes, emb=m.emb,
                        channels=channels, layers=m.layers, kernel=m.kernel,
                        max_len=cfg.data.max_len, dropout=m.dropout)
        width_key, cls = "channels", EditClassifier
    else:
        raise ValueError(f"unknown model type: {m.type}")

    width = getattr(m, width_key)
    if m.target_params:
        width = width_for_params(lambda w: cls(**make(w)), m.target_params)
    return make(width)


def build_model(model_type: str, kwargs: dict[str, Any]) -> nn.Module:
    return {"lstm_seq2seq": LSTMSeq2Seq, "edit_classifier": EditClassifier}[model_type](**kwargs)


# --- tasks: batching, loss and inference per model family -----------------------


@dataclass
class Task:
    tok: CharTokenizer
    max_len: int
    vocab: ScriptVocab | None = None

    def loss(self, model: nn.Module, batch: list[Pair]) -> torch.Tensor:
        words = [p.word for p in batch]
        src, mask = encode_batch(self.tok, words, self.max_len)
        if self.vocab is None:
            tgt, _ = encode_batch(
                self.tok, [p.lemma for p in batch], self.max_len + 1, add_bos=True, add_eos=True
            )
            logits = model(src, mask, tgt[:, :-1])
            return F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1), ignore_index=PAD_ID
            )
        labels = torch.tensor([self.vocab.id_of(extract(p.word, p.lemma)) for p in batch])
        logits = model(src, mask).masked_fill(~applicable_masks(self.vocab, words), -1e9)
        return F.cross_entropy(logits, labels)

    def lemmatizer(self, model: nn.Module) -> Lemmatizer:
        if self.vocab is None:
            return Seq2SeqLemmatizer(model, self.tok, self.max_len)
        return EditLemmatizer(model, self.tok, self.vocab, self.max_len)


# --- training loop ------------------------------------------------------------


def train(cfg: Config, out_dir: str | Path | None = None, verbose: bool = True) -> dict[str, Any]:
    set_seed(cfg.train.seed)
    out = Path(out_dir or Path("runs") / cfg.name)
    out.mkdir(parents=True, exist_ok=True)
    log = print if verbose else (lambda *_: None)

    data_dir = Path(cfg.data.dir)
    train_pairs = read_tsv(data_dir / "train.tsv")
    if cfg.data.max_train:
        train_pairs = train_pairs[: cfg.data.max_train]
    evals = {s: read_tsv(data_dir / f"{s}.tsv") for s in EVAL_SPLITS
             if (data_dir / f"{s}.tsv").exists()}
    train_lemmas = {p.lemma for p in train_pairs}

    tok = CharTokenizer.build((w for p in train_pairs for w in (p.word, p.lemma)),
                              mode=cfg.data.tokenizer)
    tok.save(out / "tokenizer.json")

    vocab = None
    if cfg.model.type == "edit_classifier":
        vocab = ScriptVocab.build(((p.word, p.lemma) for p in train_pairs), top_k=cfg.edit.top_k)
        vocab.save(str(out / "scripts.txt"))
        before = len(train_pairs)
        train_pairs = [p for p in train_pairs if extract(p.word, p.lemma) in vocab.stoi]
        log(f"edit scripts: {len(vocab)} classes; dropped {before - len(train_pairs)} "
            "train pairs with out-of-vocab scripts")

    task = Task(tok, cfg.data.max_len, vocab)
    kwargs = model_kwargs(cfg, len(tok), len(vocab) if vocab else 0)
    model = build_model(cfg.model.type, kwargs)
    n_params = count_params(model)
    log(f"{cfg.model.type}: {n_params:,} params ({kwargs})")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    gen = torch.Generator().manual_seed(cfg.train.seed)
    best_acc, best_state, best_epoch, bad = -1.0, None, 0, 0
    t0 = time.time()
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        order = torch.randperm(len(train_pairs), generator=gen).tolist()
        total = 0.0
        for i in range(0, len(order), cfg.train.batch_size):
            batch = [train_pairs[j] for j in order[i : i + cfg.train.batch_size]]
            loss = task.loss(model, batch)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(batch)

        dev = evals.get("dev")
        dev_acc = (score(dev, task.lemmatizer(model).lemmatize([p.word for p in dev]))
                   .overall.acc if dev else 0.0)
        log(f"epoch {epoch:3d}  loss {total / len(train_pairs):.4f}  dev_acc {dev_acc:.4f}")
        if dev_acc > best_acc:
            best_acc, best_epoch, bad = dev_acc, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.train.patience:
                log(f"early stop at epoch {epoch} (best {best_epoch})")
                break
    train_time = time.time() - t0

    model.load_state_dict(best_state)
    torch.save({"type": cfg.model.type, "kwargs": kwargs, "state_dict": best_state},
               out / "model.pt")

    lem = task.lemmatizer(model)
    metrics: dict[str, Any] = {
        "name": cfg.name, "model": cfg.model.type, "params": n_params, "seed": cfg.train.seed,
        "best_epoch": best_epoch, "train_time_s": round(train_time, 1), "git_sha": git_sha(),
    }
    if vocab is not None:
        metrics["script_coverage"] = {
            s: vocab.coverage((p.word, p.lemma) for p in pairs) for s, pairs in evals.items()
        }
    for split, pairs in evals.items():
        res = score(pairs, lem.lemmatize([p.word for p in pairs]), train_lemmas)
        metrics[split] = res.metrics()
        if split == "test":
            write_errors(out / "test_errors.tsv", res)

    (out / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), allow_unicode=True,
                                                    sort_keys=False), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    log(json.dumps({s: metrics[s] for s in evals}, indent=1))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--out", help="output dir (default runs/<name>)")
    args = ap.parse_args()
    train(load_config(args.config), args.out)


if __name__ == "__main__":
    main()
