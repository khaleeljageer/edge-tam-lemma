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
   The BiLSTM encoder runs its backward direction on each word reversed in
   place, so outputs do not depend on padding length; this is what makes
   fixed-length exported graphs match training.
   **Graph ops are kept to what every runtime supports (incl. ncnn):** masks
   are float 0/1 applied arithmetically (`x + (mask - 1) * MASK_NEG`, not
   `masked_fill`; `MASK_NEG = 1e4` must stay inside the fp16 range — `1e9`
   became inf in ncnn fp16 on the Pi and 0·inf = NaN broke every output), the in-place reversal is a matmul with a permutation matrix
   (`reversal_matrix`, not `gather`), and decode_step takes `prev` as (B, 1).
   The encoder outputs a flat `h0` (B, layers·H); the host reshapes it into the
   decoder state.
3. **Fixed shapes.** A fixed max word length (set from the data's 99.9th
   percentile, default 48 codepoints); inputs are padded to it.
4. **Edit-tree classifier (model family).** A char-CNN (or small BiLSTM)
   encoder predicts one of K edit scripts (e.g. strip suffix `ார்கள்`,
   append `ல்`), plus an optional POS head. One forward pass, no decoder.
   Removed strings are stored literally, so each script only applies to words
   that end/start with them; inapplicable classes are masked in both training
   and inference. With `top_k`, train pairs whose script falls outside the
   top-K are dropped. Report coverage of dev/test by the script vocab.
5. **Decoding interface.** Every seq2seq model (LSTM, transformer) implements
   `decoder_start / decoder_step / reorder_state`; `tel/decoding.py` provides
   batched greedy and beam search (n-best with log-probs, optional length
   normalization `alpha`) on top. Dev selection during training is greedy;
   final evaluation uses `decode.beam` and also reports greedy.
   The **character transformer** (`models/transformer.py`, pre-norm, learned
   positions) writes attention out with additive float masks, so it follows
   the same export rules as the LSTM; decoding recomputes the decoder over the
   prefix (no KV cache) to keep the step graph simple.
6. **Runtime layering.** `tel/runtime/host.py` (numpy only) holds
   tokenization, padding, greedy decoding and edit-script masking;
   `tel/runtime/backends.py` adds one thin loader per runtime (ONNX Runtime,
   ExecuTorch, LiteRT, ncnn). ONNX graphs have a dynamic batch; ExecuTorch and
   LiteRT graphs are exported at fixed batch sizes (1, 64); ncnn graphs are
   batch 1. The host pads up to the graph's batch size. For ncnn the host also
   computes the padding mask, end positions and reversal matrix
   (`host_features` in the manifest), because pnnx cannot convert those ops.
   ncnn has a `--precision` option: `fp32` (fp32 weights, fp16/bf16 paths
   disabled, matches the other runtimes exactly) and `fp16` (fp16 weights,
   half the size, plus fp16 arithmetic on ARMv8.2+ CPUs such as Cortex-A76),
   exported to `export/ncnn-fp16/`.
   Every export also writes `reference.tsv` (PyTorch's predictions for the
   test words), so devices report word-level `agreement` with PyTorch without
   needing torch.
   ExecuTorch and LiteRT pin their own torch, so they live in separate uv
   projects under `envs/` (`uv run --project envs/<rt> ...`); ONNX and ncnn
   (pnnx) work with the main environment.
7. **Config and seeds.** YAML loaded into typed dataclasses (no Hydra).
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
seed, git_sha, runtime_version, python, machine, timestamp, input, n_words,
agreement
```

`quant` is taken from the export manifest's `precision` (fp32 / fp16 / later int8).

For batch 64, latency columns are per word (batch time / 64). `size_kb` is the
batch-1 (or dynamic) graph set. Exploratory runs go in a subfolder
(e.g. `bench/results/m2_export_check/`), final paper runs at the top level.

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
| 12 | `export.py` | ONNX / LiteRT / ExecuTorch / ncnn for `encode` + `decode_step` (or single graph for edit classifier); `verify` vs PyTorch | Agreement ≥ 99.5% |
| 13 | `hybrid.py` | Top-N lookup + neural fallback, hit-rate stats | Hit rate on toy frequency list |
| 14 | `bench/bench_runner.py` | Minimal deps; warm-up, 5 reps, median/p95, RSS, cold start | Valid CSV row on desktop |
| 15 | `paper/figures/*.py` | Pareto, acc-vs-params, quantization plots from CSVs | Runs on fixture CSVs |

## D. Milestones

- **M1 (week 1): end-to-end skeleton on UD data.** Steps 1–6, identity +
  suffix baselines, tiny LSTM and tiny edit classifier.
- **M2 (week 4): teacher + early Pi run.** Steps 7–9; steps 11–12 for the
  teacher and the edit classifier; `bench_runner` on the Pi.
- **M3 (weeks 5–6):** size sweep (3 seeds) for LSTM, transformer and edit
  classifier (`tel.sweep`, bases in `configs/sweep/`); `char_cnn` (CNN
  encoder + the LSTM decoder) as a 1M-parameter ablation only;
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

**M2 step 1 — export check done (2026-10-10).** `edit_tiny` and `lstm_tiny`
export to ONNX, ncnn (fp32 + fp16), LiteRT and ExecuTorch; all fp32 exports
agree 100% with PyTorch on dev and on-device (Pi 5). Raspberry Pi 5,
`bench/results/m2_export_check/` (`rpi5.run3.csv`, `rpi5-fp16.csv`):

| Model | Fastest interactive (b1) | Best throughput | Lowest RAM |
|---|---|---|---|
| `edit_tiny` | LiteRT 4 thr, 0.38 ms/word | LiteRT 4 thr b1, 2,638 w/s | ncnn, 38 MB |
| `lstm_tiny` | ONNX 4 thr, 2.3 ms/word | LiteRT 4 thr b64, 1,016 w/s | ncnn, 39 MB |

- Edit classifier is 5–7× faster than the LSTM at similar accuracy.
- More threads help on the Pi (4 real cores) but hurt on the 24-thread desktop.
- ncnn: lowest RAM and fastest cold start, but 15–60% slower than the best
  runtime; fp16 halves file size (agreement 100% / 99.7%) but gives ≤11%
  speed-up on Cortex-A76 for models this small.
- ExecuTorch numbers from its Python runtime include loading torch; measure
  with its C++ runtime on Android instead.

**M2 step 2 — teachers trained (2026-10-10).** Exact match, beam 4:

| Model | Params | test | test_unseen | seen (test) |
|---|---|---|---|---|
| `teacher_lstm` | 15.8M | 72.8 | 68.5 | 85.7 |
| `teacher_transformer` | 7.4M | 53.2 | 43.3 | 82.3 |
| `lstm_tiny` | 459k | 67.9 | 64.0 | 79.4 |
| `edit_tiny` | 344k | 65.8 | 66.9 | 62.3 |

- `teacher_lstm` is the teacher for M3 distillation. Beam 4 adds +1.6 test
  (+5.2 on verbs) over greedy; no gain on dev.
- 34× the parameters buys only ~5 points over `lstm_tiny`: data (3.1k train
  pairs) limits accuracy more than model size.
- The transformer memorizes (82% seen) but generalizes poorly (43% unseen) on
  this little data; it stays in the M3 size sweep at small sizes, with tuning
  (smaller d_model, shared embeddings, larger batch) if time allows.
- `edit_tiny` (344k) is within 1.6 points of the teacher on unseen lemmas.

**M2 step 3 — external baselines done (2026-10-11); M2 complete.**
`bench/results/accuracy_baselines.csv` (exact match; `test_clean` = 277 test
words not in UD TTB train, the only fair split for tools trained on TTB):

| Model | test | test_unseen | test_clean |
|---|---|---|---|
| identity | 26.7 | 30.0 | 21.3 |
| suffix stripper | 38.7 | 41.6 | 35.7 |
| ThamizhiMorph (FST) | 55.4 | 55.9 | 53.8 |
| Stanza (ta) | 84.1* | 84.1* | 65.3 |
| tamil-lemmatizer (PyPI) | 77.0 | 73.9 | 82.7 |
| `teacher_lstm` | 72.8 | 68.5 | 69.3 |
| `lstm_tiny` | 67.9 | 64.0 | 67.9 |
| `edit_tiny` | 65.8 | 66.9 | 64.3 |

\* inflated: Stanza scores ~96% on words seen in TTB train, ~65% otherwise.

- ThamizhiMorph is 87–88% accurate on the ~40% of words its lexicon covers and
  29–35% on the rest (guessers). Its analyser output is a high-precision
  candidate source for the data pipeline (Phase 1) and a possible
  lookup tier in the hybrid (Phase 5).
- No crashes from any tool on our (clean Tamil) inputs.
- tamil-lemmatizer leads on test_clean; our models are data-limited (3.1k
  train pairs).

**M3 — size sweep, ablation, distillation (2026-10-11).**
`bench/results/m3_sweep/accuracy.csv` (60 runs: 5 bases × 0.5/1/2/5M × seeds
13–15, written by `tel.sweep --csv`). Test accuracy, mean over 3 seeds
(test_unseen in brackets):

| Model | 500k | 1M | 2M | 5M |
|---|---|---|---|---|
| LSTM | 70.5 (65.9) | 69.4 (64.5) | 71.2 (66.9) | 69.7 (65.4) |
| edit classifier | 64.8 (65.4) | 64.7 (65.6) | 64.4 (64.6) | 63.0 (63.9) |
| transformer | 61.9 (55.6) | 63.3 (55.8) | 63.5 (56.5) | 58.3 (51.1) |
| LSTM, distilled | 73.4 (69.8) | 72.3 (69.2) | 72.9 (70.0) | 70.6 (66.9) |
| edit, distilled | 70.9 (71.4) | 71.1 (71.5) | 71.1 (71.6) | 70.7 (70.9) |

- Accuracy is flat (or falls) from 0.5M to 5M for every family: data-limited.
- Distillation (`data/distill/teacher_lstm_wiki20k`: gold + 9,012
  teacher-labelled Wikipedia words) helps every family and size: LSTM +1 to +4,
  edit classifier +6 to +8. The distilled 0.5M LSTM matches the 15.8M teacher;
  the distilled edit classifier is the best model on unseen lemmas.
- Ablation (1M, `bench/results/m3_ablation/`, desktop + Pi 5): `char_cnn`
  loses 6 points to the LSTM on test and unseen; its speed gain over the LSTM
  depends on the runtime (Pi, 1 thread b1: ~80% of the LSTM→edit gap on LiteRT,
  ~40% on ONNX, 0% on ncnn). The edit classifier's single-pass output, not its
  CNN encoder, explains both its speed everywhere and its unseen-lemma accuracy.
- Distilled 0.5M students (`lstm_kd_500k_s13`, `edit_kd_500k_s13`, best dev
  seed) exported to ONNX / ncnn / LiteRT, agreement 100%; Pi bundle pending.

**M3 step 5 — data expansion (in progress).** `scripts/data/expand_candidates.py`:
candidate lemmas for the top Wikipedia words from four sources (ThamizhiMorph,
teacher, both distilled students), majority vote, and a review TSV
(`data/annotation/wiki_batch01.tsv`, top 3,000 words not in train/dev/test).
Calibration on dev (`bench/results/m3_expansion/calibration_dev.csv`):

| Status | Share | `suggested` correct |
|---|---|---|
| agree (4/4) | 37% | 93.7% |
| majority (3/4) | 36% | 79.8% |
| conflict | 27% | 49.4% |
| all | 100% | 76.8% (best single source 73.9%) |

`import` turns reviewed rows into `data/expanded/train.tsv` (gold train +
reviewed pairs), dropping dev/test lemmas. Needs a human reviewer.

## E. Risks

- ~~LSTM operator support in ExecuTorch / LiteRT / ncnn~~ — resolved at M2:
  both models export to ONNX, ExecuTorch (XNNPACK), LiteRT and ncnn with 100%
  dev agreement (ncnn also tested with a 2-layer LSTM). Still to confirm on
  ARM devices.
- ~~LiteRT batch-64 LSTM encoder 6.7 MB~~ — caused by the gather-based
  reversal; the permutation-matmul version is 1.5 MB.
- Reduced-precision runtimes can fail silently on a device while the desktop
  check passes (x86 has no fp16 arithmetic). Mitigations in place: fp16 ncnn
  exports reject constants outside the fp16 range, and every device run reports
  word-level `agreement` with PyTorch.
- pnnx traces models with `torch.jit.trace`, which PyTorch has deprecated. A
  future torch release could break the ncnn export until pnnx moves to
  `torch.export`; pin torch for the paper's runs.
- ExecuTorch's Python runtime needs torch on the device; on Android use its
  C++/Java runtime instead.
- Edit-script coverage — measure top-K coverage on dev before relying on it;
  Tamil sandhi at the stem boundary may need scripts that also edit the
  stem end (handled by edit scripts that remove/append at both ends).
- ThamizhiMorph install (FST tooling) — isolate in its own env/container.
