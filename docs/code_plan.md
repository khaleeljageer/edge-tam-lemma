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
  seed) exported to ONNX / ncnn / LiteRT, agreement 100% on desktop and Pi 5
  (`bench/results/m3_distill/`). Pi 5, ms/word, best runtime:

  | Model | test | b1, 1 thr | b1, 4 thr | RAM (ncnn) |
  |---|---|---|---|---|
  | LSTM 1M (scratch) | 71.5 | 5.29 ONNX | 5.64 ncnn | 41 MB |
  | LSTM 0.5M (distilled) | 72.7 | 2.96 ONNX | 2.45 ONNX | 40 MB |
  | edit 1M (scratch) | 67.5 | 2.18 LiteRT | 1.14 LiteRT | 41 MB |
  | edit 0.5M (distilled) | 71.0 | 0.79 LiteRT | 0.49 LiteRT | 40 MB |

  Distilled 0.5M students are more accurate and 1.8–2.8× faster than the 1M
  scratch models; the distilled edit classifier is 4–5× faster than the
  distilled LSTM at −1.7 test / +3.9 unseen.

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
reviewed pairs), dropping dev/test lemmas.

Review of `data/annotation/wiki_batch01_reviewed.tsv` (rows 1–1000 on
2026-10-11, rows 1001–3000 on 2026-10-12): 2,830 accepted (413 corrected), 39
rejected, 131 left blank; `import` kept 2,583 (247 dropped: dev/test lemma) →
5,736 training pairs (+82% over the 3,153 gold). Retrained at 500k, seeds 13–15
(`*_exp` configs; KD data rebuilt with `tel.distill --gold data/expanded/train.tsv`
→ `data/distill/teacher_lstm_wiki20k_exp`, 5,736 gold + 8,777 teacher words).
`bench/results/m3_expansion/accuracy.csv` (now the 3,000-row runs; the
1,000-row column is from the earlier version of that file, commit 8cfcb71):

| Model | dev | test | test unseen | test, rows 1–1000 only |
|---|---|---|---|---|
| LSTM | 71.3 ± 1.6 | 70.5 ± 1.1 | 65.9 ± 2.0 | |
| LSTM + reviewed | 72.0 ± 0.3 | 70.9 ± 2.0 | 65.8 ± 1.6 | 71.3 ± 0.7 |
| LSTM + KD | 72.3 ± 1.6 | 73.4 ± 1.0 | 69.8 ± 1.8 | |
| LSTM + KD + reviewed | 73.3 ± 1.1 | 72.5 ± 2.8 | 68.5 ± 2.8 | 74.5 ± 2.3 |
| Edit | 64.3 ± 0.2 | 64.8 ± 1.3 | 65.4 ± 1.5 | |
| Edit + reviewed | 67.1 ± 1.4 | 66.8 ± 1.7 | 67.4 ± 1.7 | 65.6 ± 1.7 |
| Edit + KD | 68.5 ± 0.6 | 70.9 ± 1.0 | 71.4 ± 1.2 | |
| Edit + KD + reviewed | 69.5 ± 0.8 | 71.7 ± 0.9 | 72.1 ± 1.1 | 70.9 ± 1.2 |

- The edit classifier gains: +2.8 dev / +2.0 test from scratch, +1.0 / +0.8
  with KD; more reviewed pairs add edit scripts it could not express before.
  The LSTM gains on dev (+0.7 / +1.0) but not on test, and LSTM + KD varies
  more between seeds (test 69.5–75.0). Only the scratch-edit dev gain is clearly
  above seed noise. The reviewed words are frequent Wikipedia forms, the test
  words rarer treebank forms.
- Their main value is the hybrid table (frequent forms, human labels over
  silver). Rebuilt as `table_<N>_reviewed.tsv` (`build_tables.py --reviewed …
  --tag reviewed`, `bench/results/m4_hybrid/coverage_reviewed.csv`). Test
  accuracy on table hits is unchanged (86.1% at 1k) by construction: the review
  batch excludes every train/dev/test word. On Wikipedia running text the
  reviewed words cover 22.7% of tokens and the corrected labels 3.2% — about
  +3 points of table accuracy there (no gold to measure it).
- The review confirms the dev calibration: `suggested` kept for 98.7% of agree,
  83.0% of majority, 51.6% of conflict rows (dev: 93.7 / 79.8 / 49.4). Agree rows
  can be accepted with a quick skim. Reviewer guideline fix: the
  treebank reduces most -ஆக adverbs to the noun (66 of 77; kept: அதிகமாக,
  குறிப்பாக, வேகமாக, எதிராக ...); 13 reviewed rows in 1–1000 keep the -ஆக form.

**M4 step 1 — hybrid lookup (2026-10-11).** `tel.hybrid` (table first, model
on misses) and `scripts/hybrid/build_tables.py`: top-N Wikipedia forms, labels
gold train > reviewed > silver (offline 4-source vote). Evaluated with
`bench_runner --weighted --tables`: test/dev words repeated by UD corpus
frequency (stand-in for `test_news`, which needs news-text rights).
`bench/results/m4_hybrid/` (`coverage.csv`, `desktop.csv`, `desktop_dev.csv`).
Token accuracy, distilled 0.5M students, desktop ONNX 1 thread, mean ms/token:

| Table | Size | Test hit | edit test | edit dev | edit ms | LSTM test | LSTM dev | LSTM ms |
|---|---|---|---|---|---|---|---|---|
| none | – | – | 77.8 | 65.3 | 0.122 | 78.3 | 72.4 | 0.487 |
| 1k | 45 KB | 38% | 79.7 | 71.2 | 0.077 | 79.7 | 74.0 | 0.314 |
| 5k | 247 KB | 56% | 80.4 | 73.0 | 0.054 | 80.0 | 74.2 | 0.221 |
| 10k | 512 KB | 64% | 81.0 | 73.6 | 0.045 | 81.3 | 74.5 | 0.183 |
| 50k | 2.7 MB | 84% | 82.5 | 74.2 | 0.020 | 83.2 | 75.0 | 0.081 |

- The table helps on both splits, most for the edit classifier (+3 test, +8
  dev at 10k), and closes its gap to the LSTM; mean latency falls 2.7× at 10k.
- Gains come from frequent words where the offline vote beats the student; the
  table only ever runs the expensive labellers once, offline.
- Memory: +0 MB (≤10k) to +9 MB (50k) RSS with ncnn; table file is the cost.
- Pi 5 (`rpi5-hybrid.csv`, 120 rows, same accuracies as desktop): mean
  ms/token, batch 1, best runtime per model:

  | Table | edit (LiteRT 4 thr) | edit (ncnn 4 thr) | LSTM (ONNX 1 thr) |
  |---|---|---|---|
  | none | 0.488 | 0.828 | 2.96 |
  | 10k | 0.183 | 0.302 | 1.10 |
  | 50k | 0.078 | 0.130 | 0.48 |

  Speed-up ≈ 1 / (1 − hit rate): the lookup itself costs ~2 µs on the Pi.
- Fixed-batch runtimes (LiteRT; graphs for batch 1 and 64 only) get no
  speed-up at batch 64: the misses of a 64-word batch are padded back to 64.
  Fix: export intermediate batch sizes, or send small miss sets through the
  batch-1 graph.

**M4 step 2 — INT8 quantization (2026-10-11).** `tel.quantize` (fp32 export →
`<export>-int8dyn` / `-int8`, fp32 PyTorch reference kept, so `agreement` is the
quantization error). Host fix first: `chunk_sizes` no longer pads a small
remainder up to the batch-64 graph (hybrid misses on LiteRT: edit 2.9× faster
at b64). Desktop, 1 thread, batch 1 (`bench/results/m4_quant/desktop.csv`):

| Model | Variant | Size | Test acc | Agreement | ms/word |
|---|---|---|---|---|---|
| edit | ONNX fp32 | 1,986 KB | 70.95 | 100% | 0.121 |
| edit | ONNX int8 dynamic | 502 KB | 70.81 | 99.9% | 0.384 (slower) |
| edit | ONNX int8 static (QDQ) | 544 KB | 70.81 | 99.4% | 0.081 |
| edit | LiteRT fp32 | 1,962 KB | 70.95 | 100% | 0.111 |
| edit | LiteRT int8 dynamic | 538 KB | 70.95 | 100% | 0.060 |
| LSTM | ONNX fp32 | 2,063 KB | 72.69 | 100% | 0.484 |
| LSTM | ONNX int8 dynamic | 511 KB | 72.83 | 98.6% | 0.320 |
| LSTM | ONNX int8 static | 1,682 KB | 72.69 | 98.8% | 0.499 |
| LSTM | LiteRT fp32 | 2,208 KB | 72.69 | 100% | 0.535 |
| LSTM | LiteRT int8 dynamic | 796 KB | 72.69 | 100% | 0.223 |

- INT8 costs ≤0.15 points everywhere; files shrink 2.8–4×.
- Best: LiteRT dynamic int8 (edit 1.9×, LSTM 2.4× faster, 100% agreement).
- ONNX dynamic int8 slows the edit classifier 3× (ORT's ConvInteger path);
  static QDQ is the ONNX choice for the CNN. ONNX static leaves the LSTM op in
  float (no size or speed gain); dynamic is the ONNX choice for the LSTM.
- Not done: static (full-int8) LiteRT — the additive padding mask and float
  graph inputs break it (see `tel.quantize`); ncnn int8 needs ncnn's C++ tools.

Raspberry Pi 5 (`bench/results/m4_quant/rpi5.csv`, active cooling, throttled=0x0;
fp32 rows reproduce the M3 distilled run within 2%). ms/word, batch 1 unless noted:

| Model | Variant | 1 thr | 4 thr | Best (thr, batch) | Peak RSS |
|---|---|---|---|---|---|
| edit | ONNX fp32 | 1.042 | 0.622 | 0.446 (4, 64) | 85 MB |
| edit | ONNX int8 dynamic | 0.621 | 0.446 | 0.433 (4, 64) | 84 MB |
| edit | ONNX int8 static | 0.570 | 0.382 | 0.295 (4, 64) | 77 MB |
| edit | LiteRT fp32 | 0.810 | 0.478 | 0.478 (4, 1) | 60 MB |
| edit | LiteRT int8 dynamic | 0.337 | **0.288** | **0.288 (4, 1)** | 55 MB |
| edit | ncnn fp32 | 1.668 | 0.817 | 0.786 (4, 64) | 40 MB |
| LSTM | ONNX fp32 | 2.953 | 2.426 | 1.350 (4, 64) | 90 MB |
| LSTM | ONNX int8 dynamic | 1.872 | 1.864 | 1.051 (1, 64) | 88 MB |
| LSTM | ONNX int8 static | 2.907 | 2.409 | 1.295 (4, 64) | 90 MB |
| LSTM | LiteRT fp32 | 3.800 | 4.717 | 1.061 (4, 64) | 136 MB |
| LSTM | LiteRT int8 dynamic | 1.867 | 2.706 | **0.856 (4, 64)** | 96 MB |
| LSTM | ncnn fp32 | 4.626 | 3.584 | 3.584 (4, 1) | 41 MB |

- Accuracy and agreement match the desktop except LSTM LiteRT int8 dynamic:
  72.54 / 99.9% agreement on the Pi vs 72.69 / 100% on x86 (ARM integer
  kernels round differently). Loss stays ≤0.15 points. The LSTM ONNX dynamic
  1- vs 4-thread difference (72.83 vs 72.69) reproduces on ARM.
- LiteRT dynamic int8 is again the best: edit 2.4× (1 thr) / 1.7× (4 thr)
  faster, 0.288 ms/word = 3,450 words/s, the fastest configuration measured so
  far; LSTM 2.0× faster at batch 1 and 0.86 ms/word at 4 thr batch 64.
- Unlike x86, ONNX dynamic int8 speeds the edit classifier up on the A76
  (1.7×, integer dot-product kernels); static QDQ is still faster (1.8×,
  0.295 ms at 4 thr b64). ONNX static again gives the LSTM nothing.
- INT8 also cuts LiteRT memory (LSTM 136 → 96 MB). ncnn fp32 stays the lowest-RAM
  option (40 MB) but the slowest.
- Batch-1 LSTM on LiteRT is slower with 4 threads than 1 (per-step ops too
  small to split); report the best thread count per runtime.

**M4 step 3 — INT8 + hybrid on the Pi (2026-10-11).**
`bench/results/m4_quant_hybrid/rpi5-hybrid.csv` (all 12 exports × no table / 1k /
5k / 10k / 50k; 41.7 °C, throttled=0x0; fp32 batch-1 rows reproduce `m4_hybrid`
within 3%, except LSTM ONNX Runtime at 4 threads, up to 22% faster this time:
multi-thread LSTM timing on ORT is noisy, report its 1-thread numbers).
Mean ms per token, batch 1, best thread count:

| Table | Acc (edit / LSTM) | Edit LiteRT fp32 | Edit LiteRT int8 | LSTM ONNX fp32 | LSTM LiteRT int8 |
|---|---|---|---|---|---|
| none | 77.8 / 78.3 | 0.493 | 0.288 | 2.441 | 1.812 |
| 10k (512 KB) | 81.0 / 81.3 | 0.182 | 0.104 | 1.099 | 0.670 |
| 50k (2.7 MB) | 82.5 / 83.2 | 0.078 | **0.048** | 0.492 | 0.294 |

- The two gains multiply: Edit + KD LiteRT int8 + 50k table = 0.048 ms/token
  (≈21k tokens/s on one Pi 5), 10× faster than fp32 without a table and +4.7
  points; 56 MB RSS. Fastest batched: ONNX static int8 + 50k, 0.040 ms (4 thr, b64).
- int8 accuracy with a table equals fp32 to ≤0.3 points (table hits dominate).
- The `chunk_sizes` fix works for the edit classifier on LiteRT b64 (0.500 →
  0.066 ms at 50k) but hurt the LSTM on LiteRT b64 with 5k/10k tables
  (1.11 → 1.80 ms): a batch-1 LSTM call is not much cheaper than a batch-64 one
  (3.8 vs ~70 ms), so stepping down pays off only below ~18 misses, not 32.
  **Fixed:** hosts now `calibrate()` (time one call per fixed-batch graph, done
  by the bench worker before timing) and `chunk_sizes` picks the cheapest split
  by dynamic programming; without calibration the old heuristic applies.
  Desktop check (LiteRT, 1 thr, b64, ms/token; `bench/tmp`, not a result file):
  LSTM none / 10k / 50k 0.199 / 0.197 / 0.089, edit 0.118 / 0.039 / 0.017 — no
  table size is slower than no table any more.
  Pi rerun of the LiteRT exports (`bench/results/m4_chunk_cost/rpi5-hybrid.csv`,
  37.8 °C, throttled=0x0; accuracy identical, batch-1 within 3% of before).
  Mean ms/token, 4 threads, batch 64, none / 1k / 5k / 10k / 50k:

  | Export | Before (half-full rule) | After (calibrated) |
  |---|---|---|
  | LSTM fp32 | 1.106 / 1.123 / 1.804 / 1.674 / 0.741 | 1.117 / 1.121 / 1.095 / 1.071 / 0.704 |
  | LSTM int8 | 0.914 / 0.920 / 1.068 / 0.928 / 0.433 | 0.914 / 0.926 / 0.935 / 0.890 / 0.428 |
  | Edit fp32 | 0.500 / 0.493 / 0.235 / 0.163 / 0.066 | 0.409 / 0.492 / 0.181 / 0.154 / 0.066 |
  | Edit int8 | 0.417 / 0.417 / 0.165 / 0.099 / 0.042 | 0.253 / 0.158 / 0.108 / 0.090 / 0.041 |

  The LSTM slowdown is gone (5k: 1.80 → 1.10 ms). The edit classifier gains even
  without a table (int8 0.417 → 0.253): its batch-64 graph is no cheaper per
  word than batch 1, so the split now uses batch-1 calls. One miss: edit fp32
  1k at 4 threads still pads (0.492; batch-1 calls would give ~0.30) — the
  best-of-3 calibration timing is noisy with 4 threads on the Pi.

**M5 step 1 — figure and table scripts (2026-10-11).** `paper/figures/`
(`uv run python paper/figures/make_all.py`, matplotlib in the `paper` dependency
group), output in `paper/figures/out/` (git-ignored, regenerated):

- `fig_params.py` → `acc_vs_params`: test accuracy vs parameters, mean ± std over
  seeds from `m3_sweep/accuracy.csv`, with and without KD, baseline reference lines.
- `fig_pareto.py` → `pareto`: accuracy vs median batch-1 latency, best thread count
  per (model, precision, runtime), Pareto front per device (Pi 5, desktop), from
  the M2–M4 device CSVs (later runs override earlier ones). Front on the Pi:
  Edit + KD LiteRT int8 (0.29 ms, 71.0) → LSTM + KD int8 (1.87 ms, 72.5–72.8).
- `tables.py` → `tab_accuracy`, `tab_quant`, `tab_hybrid` (LaTeX, booktabs).

**M5 step 2 — error analysis (2026-10-11).** `scripts/analysis/error_analysis.py`
rebuilds each run's test predictions from `runs/<run>/test_errors.tsv` and
categorizes every error with `tel.error_analysis` (Tamil-grapheme rules) →
`bench/results/m5_errors/{errors,categories,examples}.csv`; `tables.py` renders
`tab_errors.tex` and `tab_error_examples.tex` (15 examples, fixed seed; Tamil
wrapped in `\ta{}`, needs XeLaTeX/LuaLaTeX). Errors on test (500k, seed 13):

| Category | Teacher | LSTM | LSTM+KD | Edit | Edit+KD |
|---|---|---|---|---|---|
| Form left unchanged | 57 | 61 | 51 | 89 | 62 |
| Gold keeps the form | 46 | 42 | 42 | 39 | 36 |
| Suffix partly left | 23 | 30 | 19 | 36 | 25 |
| Too much removed | 11 | 5 | 8 | 14 | 13 |
| Stem-final letter wrong | 26 | 34 | 36 | 50 | 42 |
| Other / wrong stem | 25 | 29 | 33 | 26 | 23 |
| Total | 188 | 201 | 189 | 254 | 201 |

- KD helps the edit classifier mostly by no longer copying the form (89 → 62)
  and restoring stem-final letters (50 → 42).
- 69 of Edit+KD's 201 errors need an edit script outside its vocabulary (a hard
  ceiling: script coverage 90% on test); the LSTM has no such limit.
- "Gold keeps the form" (~40 per model) is mostly a treebank convention for
  adverbs, postpositions, numerals and participles (அத்துடன், முப்பது); 86 test
  pairs are wrong in all five models — candidates for annotation noise
  (e.g. பங்கேற்க → gold பங்கேல்).
- The automatic categories still need a manual pass on the 15 examples before
  the paper.

**M5 step 3 — manuscript (2026-10-14).** Target: Sādhanā (Indian Academy of
Sciences / Springer, Scopus + WoS, regular paper ≤ 20 pages, numbered references
in citation order, ≤ 6 keywords, 300 dpi figures, AI-use disclosure required);
TALLIP if it is not accepted. `paper/main.tex` uses the journal's class
(`paper/sadhana.cls`, downloaded 2026-10-14) with XeLaTeX (TeX Gyre Termes +
Noto Serif Tamil) and `\input`s the generated tables and figures from
`paper/figures/out/`. Tables have captions above; wide ones are `table*`
(`latex_table(..., wide=True)`). New tables: `tab_sweep` (all sizes),
`tab_ablation` (1M encoder ablation, accuracy from
`bench/results/m3_ablation/accuracy.csv`, written by `tel.sweep --csv`, plus Pi
latency) and `tab_expansion`. Figures are also written as 300 dpi PNG/TIFF.
`paper/order_refs.py` keeps the `\bibitem`s in citation order (tested).
References checked online; `%VERIFY` marks three entries still to confirm.
Open (`\todo{}`): speaker-number source, more Tamil/Dravidian related work,
second annotator, manual check of the error examples, repository URL, code
licence, funding, AI-use statement. Not compiled yet (no TeX on the dev
machine). Test clean for the distilled students (seed 13) added to
`accuracy_baselines.csv` (`collect_runs.py`): LSTM+KD 72.9, Edit+KD 70.8,
teacher 69.3. `docs/reproduce.md` lists the commands behind every result.
Compiled 2026-10-14 (XeLaTeX): 9 pages, no overfull boxes (tables are scaled to
the column with `adjustbox`). Second annotator (a Tamil professor, Dr. Revathi):
`scripts/data/agreement.py sample` wrote a blind, status-stratified 200-word
sheet and the 15 error examples to `data/annotation/agreement/` (instructions in
its README, rules in `docs/annotation_guidelines.md`); `score` writes
`bench/results/m3_expansion/agreement.csv`. Code licence: GPL-3.0 (`LICENSE`).

**Devices (2026-10-13).** Final device set: Raspberry Pi 5 (ARM edge) and the
desktop (x86 reference); Android phone and Qualcomm board dropped (research plan
§8.1). The only phone at hand, a Galaxy A13, runs 32-bit Android (armeabi-v7a
only), for which the onnxruntime / LiteRT / ncnn Python packages have no builds
(details in `bench/devices.md`). Limitations paragraph for the paper: the Pi 5's
Cortex-A76 cores are common in mid-range phones, but Android runtime builds were
not measured; low-end phones still ship 32-bit Android, which needs native
runtime builds rather than the Python packages used here.

## E. Risks

- ~~LSTM operator support in ExecuTorch / LiteRT / ncnn~~ — resolved at M2:
  both models export to ONNX, ExecuTorch (XNNPACK), LiteRT and ncnn with 100%
  dev agreement (ncnn also tested with a 2-layer LSTM); confirmed on the Pi 5.
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
