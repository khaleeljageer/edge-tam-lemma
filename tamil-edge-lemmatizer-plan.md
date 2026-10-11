# Compact Neural Lemmatization for Tamil on Edge Devices

Research and implementation plan. Author: Khaleel Jageer.

This file is written to be used directly with Claude Code. Each phase has
tasks, outputs, and "done when" criteria. Work through the phases in order.

---

## 0. Ground rules (read first)

- **All code in this project is written from scratch.** Do not copy code from
  `Hemanthkumar2112/tamil-lemmatizer` or any other lemmatizer repo.
- Existing tools (the `tamil-lemmatizer` PyPI package, Stanza, ThamizhiMorph)
  are used **only as baselines**, installed as external dependencies and cited
  in the paper.
- Check and record the license of every dataset before using it
  (`docs/data_licenses.md`). Some treebanks are non-commercial (NC); that is
  fine for research but must be stated.
- Every number in the paper must be reproducible from a script in this repo
  with a fixed seed.

---

## 1. Research summary

**Research question:** How small and fast can a Tamil lemmatizer be on
low-power hardware while keeping accuracy close to a large model?

**Contributions:**

1. A gold-standard Tamil word → lemma dataset with train/dev/test splits,
   including a test split of lemmas never seen in training.
2. A family of compact lemmatization models (0.5M–5M parameters), produced
   with distillation and quantization, plus a hybrid lookup + neural scheme.
3. A benchmark of accuracy vs efficiency (latency, memory, size, energy) on
   real edge hardware.

**Working title:** Compact Neural Lemmatization for Tamil on Edge Devices

**Target venues:** DravidianLangTech workshop (ACL-family), ICON, SustaiNLP.
Short paper (4 pages + references) first.

---

## 2. Repository layout

```
tamil-edge-lemma/
├── CLAUDE.md                 # short project rules for Claude Code
├── README.md
├── pyproject.toml
├── configs/                  # YAML configs, one per experiment
├── data/
│   ├── raw/                  # downloaded sources (git-ignored)
│   ├── interim/              # candidate pairs before annotation
│   └── processed/            # final train/dev/test TSV files
├── docs/
│   ├── annotation_guidelines.md
│   ├── data_licenses.md
│   └── related_work.md
├── src/tel/
│   ├── normalize.py          # Unicode NFC, cleaning
│   ├── tokenizer.py          # character / grapheme tokenizer
│   ├── data.py               # dataset loading, splits
│   ├── models/
│   │   ├── lstm_seq2seq.py
│   │   ├── transformer.py
│   │   ├── char_cnn.py
│   │   └── edit_classifier.py
│   ├── train.py
│   ├── distill.py
│   ├── quantize.py
│   ├── export.py             # ONNX / LiteRT / ExecuTorch / ncnn
│   ├── hybrid.py             # lookup table + neural fallback
│   └── evaluate.py
├── baselines/
│   ├── suffix_stripper.py
│   ├── run_stanza.py
│   ├── run_thamizhimorph.py
│   └── run_tamil_lemmatizer_pkg.py
├── bench/
│   ├── bench_runner.py       # runs on each device
│   ├── devices.md            # hardware details
│   └── results/              # raw CSV results per device
├── paper/
│   ├── figures/
│   └── main.tex
└── tests/
```

---

## 3. Phase 1 — Data

### 3.1 Sources

| Source | Use | Notes |
|---|---|---|
| UD Tamil treebanks (TTB, MWTT) | Gold lemmas | Small; check license (may be NC) |
| ThamizhiMorph (rule-based FST) | Generate candidate lemmas | Output must be hand-corrected |
| Tamil Wikipedia dump | Word frequency list, raw words | CC BY-SA |
| Own news text (with permission) | Domain test set | Confirm usage rights first |

### 3.2 Pipeline

1. Use **written words**: UD multiword tokens (e.g. உறுதியளித்துள்ளார் =
   உறுதியளித்த் + உள்ளார்) are kept as one word whose lemma and POS come from
   the first content part (here உறுதியளி, VERB), not as the split parts.
2. Normalize all text to Unicode **NFC**. Remove non-Tamil characters from
   word tokens; strip punctuation.
3. Build a frequency list of word forms from the raw corpora.
4. Generate candidate lemmas for the top ~20,000 word forms using
   ThamizhiMorph, then hand-correct.
5. Merge with UD gold lemmas. Resolve conflicts manually and log them.
6. Record for each pair: `word`, `lemma`, `pos` (NOUN/VERB/OTHER),
   `source`, `frequency`.

### 3.3 Splits

- Split **by lemma**, not by word form, so the test set contains lemmas the
  model has never seen.
- `train` 80% / `dev` 10% / `test` 10% of lemmas.
- Additionally ~5% of train pairs each move to `dev` and `test` as held-out
  word forms of *training* lemmas (every lemma keeps ≥1 form in train), so
  dev/test report both seen-lemma and unseen-lemma accuracy.
- Extra split `test_unseen`: the unseen-lemma part of `test` on its own.
- Extra split `test_news`: words from news text, frequency-weighted, for the
  hybrid lookup experiment.
- Fixed seed; save split lemma lists to `data/processed/splits/`.

### 3.4 Annotation quality

- Write `docs/annotation_guidelines.md` (see Appendix A).
- Two annotators label the same random 500 pairs; report Cohen's kappa or
  simple agreement percentage.

**Done when:** processed TSV files exist, splits are frozen, agreement
is reported, and licenses are documented.

---

## 4. Phase 2 — Baselines and teacher model

### 4.1 Baselines

1. **Identity:** output the word unchanged (lower bound).
2. **Suffix stripper:** a hand-written rule list of common Tamil suffixes.
3. **ThamizhiMorph:** rule-based morphological analyzer.
4. **Stanza** Tamil lemmatizer.
5. **`tamil-lemmatizer` PyPI package** (existing neural model). Wrap calls in
   try/except — it can crash on characters outside its vocabulary; count
   crashes as errors and report them.

### 4.2 Teacher model

- Character-level BiLSTM encoder + attention + LSTM decoder, large
  (e.g. embed 256, hidden 512), or a 4-layer character transformer.
- Beam search (beam 4) at evaluation.
- Train on `train`, select on `dev`, report on `test` and `test_unseen`.

**Done when:** a results table with all baselines + teacher exists in
`bench/results/accuracy_baselines.csv`.

---

## 5. Phase 3 — Compact models

### 5.1 Size sweep

| Model | Parameter targets |
|---|---|
| LSTM seq2seq | ~0.5M, 1M, 2M, 5M |
| Character transformer | ~0.5M, 1M, 2M, 5M |
| Character CNN encoder + small decoder (ablation only) | ~1M |
| Edit-tree classifier (char encoder → edit-script class) | ~0.5M, 1M, 2M |

Train each from scratch (3 seeds each). Report mean ± std accuracy.

The char-CNN model is an **ablation**, not a full sweep family: it keeps the
LSTM's attention decoder and swaps only the encoder for a dilated CNN. At 1M
parameters it separates the two ways the edit classifier differs from the LSTM
(CNN encoder vs single-pass edit-script output): is the edit classifier fast
and robust because of its encoder or because it does not decode character by
character?

### 5.2 Knowledge distillation

- Teacher outputs (soft labels or teacher-decoded sequences) used to train
  each small student.
- Compare: student from scratch vs distilled student, same size.

**Done when:** accuracy vs parameter count is plotted for all model types,
with and without distillation.

---

## 6. Phase 4 — Quantization and export

1. Post-training quantization: fp32 → INT8 (dynamic and static).
2. Optional: INT4 weights where the runtime supports it.
3. Quantization-aware training only if INT8 loses more than ~1 point.
4. Export each candidate to:
   - ONNX Runtime
   - LiteRT (TFLite)
   - ExecuTorch
   - ncnn (via pnnx) — ARM-optimized, no dependencies, small footprint;
     a strong fit for Raspberry Pi and Android CPU inference
5. Verify exported model outputs match PyTorch outputs on the dev set
   (report agreement %).

**Done when:** each exported model passes the output-match check and file
sizes are recorded.

---

## 7. Phase 5 — Hybrid lookup + neural

- Build a lookup table of the top-N most frequent word forms → lemma
  (N = 1k, 5k, 10k, 50k).
- At inference: check table first; call the model only on a miss.
- Measure on `test_news` (frequency-weighted, realistic text):
  - hit rate
  - end-to-end accuracy
  - average latency per word
  - memory (table + model)

**Done when:** a table of N vs hit rate vs latency vs memory exists.

---

## 8. Phase 6 — Edge benchmarking

### 8.1 Devices

| Device | Role |
|---|---|
| Raspberry Pi CM5 | ARM CPU-only edge baseline |
| Mid-range Android phone | Consumer mobile target |
| Qualcomm automotive board (if feasible) | Embedded Android / automotive |
| Desktop x86 CPU | Reference |

Record CPU model, clock, RAM, OS, runtime version, and thread count in
`bench/devices.md`.

### 8.2 Protocol

- Fixed input: 10,000 words from `test_news`, same order on every device.
- Warm-up: 200 words, not measured.
- Run each configuration 5 times; report median.
- Threads: 1 thread and all cores, reported separately.
- Same power mode / governor on each run (record it).
- Batch sizes: 1 (interactive) and 64 (bulk).

### 8.3 Metrics

| Metric | How |
|---|---|
| Accuracy | Exact match, overall + NOUN/VERB + seen/unseen |
| Latency per word | Median and p95, batch 1 |
| Throughput | Words per second, batch 64 |
| Peak RAM | RSS during run |
| Model size | File size on disk |
| Cold start | Process start → first lemma |
| Energy (optional) | USB power meter on Pi: joules per 1,000 words |

All raw results go to `bench/results/<device>.csv` with one row per
(model, quantization, runtime, threads, batch).

**Done when:** every selected model has been run on every device and the
Pareto figure (accuracy vs median latency, one curve per device) is generated
by a script.

---

## 9. Phase 7 — Paper

### 9.1 Structure (short paper, 4 pages)

1. **Introduction** — Tamil morphology, need for on-device NLP, contributions.
2. **Related work** — Tamil morphological analysis and lemmatization
   (rule-based and neural), SIGMORPHON lemmatization tasks, efficient NLP
   (distillation, quantization).
3. **Dataset** — sources, pipeline, splits, agreement, license.
4. **Method** — model family, distillation, quantization, hybrid lookup.
5. **Experiments** — setup, devices, protocol.
6. **Results** — accuracy table, Pareto figure, hybrid table, error analysis.
7. **Conclusion and limitations.**

### 9.2 Required figures and tables

- Table: baselines vs teacher vs best compact models (accuracy).
- Figure: accuracy vs parameters (with/without distillation).
- Figure: accuracy vs latency Pareto curve per device.
- Table: quantization effect (size, latency, accuracy).
- Table: hybrid lookup (N, hit rate, latency, memory).
- Error analysis: 10–15 real examples with categories.

### 9.3 Release

- Code (open license, e.g. MIT or Apache-2.0)
- Dataset (license compatible with sources)
- Trained models with pinned versions
- Exact commands to reproduce every table

---

## 10. Timeline (suggested, 12 weeks)

| Weeks | Work |
|---|---|
| 1–3 | Data collection, annotation, splits |
| 4 | Baselines + teacher |
| 5–6 | Compact models + distillation |
| 7 | Quantization + export |
| 8 | Hybrid lookup |
| 9–10 | Device benchmarks |
| 11–12 | Writing, figures, release |

Start a minimal benchmark run on the Raspberry Pi in **week 4** with the
teacher model, so export and operator-support problems show up early.

---

## Appendix A — Annotation guidelines (draft)

1. **Verbs:** lemma is the verb root (imperative singular form).
   Example: சென்றார்கள் → செல்.
2. **Nouns:** lemma is the singular nominative form, without case markers,
   plural marker (-கள்), or clitics (-உம், -ஏ, -ஆ, -ஓ).
3. **Pronouns:** map to the base pronoun form.
4. **Compounds and sandhi:** keep compounds intact; undo sandhi doubling
   only at the case/suffix boundary.
5. **Loanwords:** lemmatize by the same rules; keep spelling as written.
6. **Proper nouns:** remove case markers; keep the name unchanged.
7. **Uncertain cases:** mark `uncertain=1` and add a note; resolve in a
   weekly review.
8. All lemmas in NFC Unicode; no punctuation; no non-Tamil characters.

Expand with 50+ worked examples before annotation starts.

---

## Appendix B — Starter CLAUDE.md

```
# Project: Tamil edge lemmatizer (research)

- Python 3.11, PyTorch. Use uv or venv.
- All code original. Never copy code from other lemmatizer repos.
- Baselines are external dependencies only.
- Every experiment is a YAML config in configs/ + a fixed seed.
- Results are written as CSV to bench/results/, never edited by hand.
- Figures are generated by scripts in paper/figures/.
- Normalize all Tamil text to NFC before any processing.
- Run tests in tests/ before committing.
```

---

## Appendix C — Pre-submission checklist

- [ ] All prior work cited, including the existing `tamil-lemmatizer` package
- [ ] Dataset licenses stated
- [ ] Every number reproducible from a script
- [ ] Seeds and versions recorded
- [ ] Limitations section written honestly
- [ ] Code, data, and models released or ready to release
