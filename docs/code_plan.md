# Code plan — Tamil edge lemmatizer

Companion to `tamil-edge-lemmatizer-plan.md` (research plan). This file covers
how the code is structured and built. Work through milestones in order.

---

## A. Design decisions

1. **Character unit.** Default: Unicode codepoints after NFC (~80 symbols).
   Grapheme tokenization (consonant + vowel sign as one unit) is an option of
   the same tokenizer, used as an ablation.
2. **Exportable decoding.** Every seq2seq model is two graphs:
   `encode(src) -> memory` and
   `decode_step(prev_token, state, memory) -> logits, state`.
   The decode loop lives in host code (Python now; Kotlin/C++ on device).
   Greedy decoding on device; beam search only for the teacher.
3. **Fixed shapes.** A fixed max word length (set from the data's 99.9th
   percentile, default 48 codepoints); inputs are padded to it.
4. **Edit-tree classifier (model family).** A char-CNN (or small BiLSTM)
   encoder predicts one of K edit scripts (e.g. strip suffix `ார்கள்`,
   append `ல்`), plus an optional POS head. One forward pass, no decoder.
   Removed strings are stored literally, so each script only applies to words
   that end/start with them; inapplicable classes are masked in both training
   and inference. With `top_k`, train pairs whose script falls outside the
   top-K are dropped. Report coverage of dev/test by the script vocab.
5. **Config and seeds.** YAML loaded into typed dataclasses (no Hydra).
   Each run writes `runs/<name>/{config.yaml, metrics.json, model.pt, git_sha}`.

## B. Data contracts

`data/processed/{split}.tsv` (tab-separated, with header):

```
word  lemma  pos  source  freq  uncertain
```

Splits: `train`, `dev`, `test`, `test_unseen`, `test_news`. Lemma lists per
split in `data/processed/splits/*.txt`. Words are written (surface) words; UD
multiword tokens take the lemma of their first content part. Lemmas are split
80/10/10; then ~5% of train pairs each go to dev and test as seen-lemma forms
(`tel.data.make_splits`). `test_unseen` = unseen-lemma part of `test`.

`bench/results/*.csv`, one row per run:

```
device, model, params, quant, runtime, threads, batch, acc, acc_noun, acc_verb,
acc_unseen, lat_med_ms, lat_p95_ms, wps, peak_rss_mb, size_kb, cold_start_ms,
seed, git_sha
```

Common interface for all models and baselines:

```python
class Lemmatizer(Protocol):
    def lemmatize(self, words: list[str]) -> list[str]: ...
```

`evaluate.py`, `hybrid.py` and `bench_runner.py` only depend on this.

## C. Modules and build order

| # | Module | Contents | Tests |
|---|---|---|---|
| 1 | Scaffold | `pyproject.toml` (uv, py3.11), `CLAUDE.md`, ruff, pytest, `.gitignore`, git | `pytest` passes |
| 2 | `normalize.py` | NFC, Tamil-only filter, punctuation strip, `is_tamil_word()` | Composed/decomposed input agree; mixed-script input |
| 3 | `tokenizer.py` | `CharTokenizer` (codepoint / grapheme), specials, encode/decode, JSON save/load | Round-trip; `கொ` is one grapheme |
| 4 | `scripts/data/` | `fetch_ud.py`, `wiki_freq.py`, `make_candidates.py`, `merge.py`, `split.py` | No lemma overlap across splits; deterministic split |
| 5 | `data.py`, `edit_tree.py` | TSV I/O, torch `Dataset`, collate; edit-script extract/apply, script vocab | `apply(extract(w, l), w) == l` on all pairs |
| 6 | `evaluate.py` | Exact match overall / POS / seen-unseen; error dump | Hand-built predictions give expected numbers |
| 7 | `tel/baselines/`, `scripts/baselines/` | identity, suffix stripper (in package); Stanza, ThamizhiMorph, PyPI package (crash counter) | Output length equals input length |
| 8 | `models/` | `lstm_seq2seq`, `transformer`, `char_cnn`, `edit_classifier`; size-from-param-target helper | Params within ±10% of target; overfit 100 examples |
| 9 | `train.py` | Trainer: AdamW, early stop on dev acc, seeds, logging | 1-epoch smoke run on 1k examples |
| 10 | `distill.py` | Teacher-decoded targets (+ unlabeled wiki words); optional soft-label KL. For the edit classifier: teacher output → edit script | Deterministic targets file |
| 11 | `quantize.py` | Dynamic + static INT8 (dev calibration), accuracy delta | Quantized within N points on toy set |
| 12 | `export.py` | ONNX / LiteRT / ExecuTorch for `encode` + `decode_step` (or single graph for edit classifier); `verify` vs PyTorch | Agreement ≥ 99.5% |
| 13 | `hybrid.py` | Top-N lookup + neural fallback, hit-rate stats | Hit rate on toy frequency list |
| 14 | `bench/bench_runner.py` | Minimal deps; warm-up, 5 reps, median/p95, RSS, cold start | Valid CSV row on desktop |
| 15 | `paper/figures/*.py` | Pareto, acc-vs-params, quantization plots from CSVs | Runs on fixture CSVs |

## D. Milestones

- **M1 (week 1): end-to-end skeleton on UD data.** Steps 1–6, identity +
  suffix baselines, tiny LSTM and tiny edit classifier.
- **M2 (week 4): teacher + early Pi run.** Steps 7–9; steps 11–12 for the
  teacher and the edit classifier; `bench_runner` on the Pi.
- **M3 (weeks 5–6):** size sweep (3 seeds) for all four families,
  distillation; config-driven sweep launcher.
- **M4 (weeks 7–8):** full quantization/export matrix, hybrid.
- **M5 (weeks 9–10):** device runs, figure scripts.

## D2. Status

**M1 done (2026-10-02).** UD Tamil (TTB + MWTT): 3,153 train / 668 dev /
692 test pairs. Results (exact match, test / test_unseen):

| Model | Params | test | test_unseen |
|---|---|---|---|
| identity | – | 26.7 | 30.0 |
| suffix stripper | – | 38.7 | 41.6 |
| `lstm_tiny` | 459k | 68.6 | 62.7 |
| `edit_tiny` | 344k | 65.8 | 66.9 |

Edit-script vocab (703 classes from train) covers 86% of test pairs. Seen-lemma
accuracy: LSTM 86%, edit classifier 62% — the classifier generalizes better to
unseen lemmas but does not memorize; a hybrid lookup should close that gap.

## E. Risks

- LSTM operator support in ExecuTorch / LiteRT — test at M2; GRU or
  transformer fallback.
- Edit-script coverage — measure top-K coverage on dev before relying on it;
  Tamil sandhi at the stem boundary may need scripts that also edit the
  stem end (handled by edit scripts that remove/append at both ends).
- ThamizhiMorph install (FST tooling) — isolate in its own env/container.
