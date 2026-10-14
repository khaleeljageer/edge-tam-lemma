# Reproducing the results

Every number in `bench/results/` and every paper table and figure comes from
the commands below, in this order. Seeds are fixed in the configs (`train.seed`)
or in the scripts. Run from the repository root.

`tel.sweep` skips runs whose `runs/<name>/metrics.json` exists, and
`bench_runner.py` appends to its `--out` CSV. To reproduce from scratch, use an
empty `runs/` and write benchmark CSVs to new files.

## 0. Environments

```
uv sync                                    # main env: torch (CPU), onnxruntime, ncnn, matplotlib
uv sync --project envs/litert              # LiteRT export, quantization and runtime
uv sync --project envs/executorch          # ExecuTorch export and runtime (M2 only)
uv sync --project envs/baselines           # Stanza, tamil-lemmatizer (PyPI)
sudo apt install foma                      # ThamizhiMorph baseline (flookup)
```

## 1. Data (M1)

```
uv run python scripts/data/fetch_ud.py     # UD Tamil TTB + MWTT at a pinned ref -> data/raw/ud/
uv run python scripts/data/build_ud.py     # data/processed/{train,dev,test}.tsv, splits/ (seed 13)
uv run python scripts/data/wiki_freq.py    # pinned Tamil Wikipedia dump -> data/interim/wiki_freq.tsv
```

## 2. Baselines and teachers (M2) → `bench/results/accuracy_baselines.csv`

```
uv run python scripts/baselines/run_simple.py                  # identity, suffix rules
uv run python scripts/baselines/run_thamizhimorph.py
uv run --project envs/baselines python scripts/baselines/run_stanza.py
uv run --project envs/baselines python scripts/baselines/run_tamil_lemmatizer_pkg.py
uv run python -m tel.train configs/teacher_lstm.yaml
uv run python -m tel.train configs/teacher_transformer.yaml
uv run python scripts/baselines/collect_runs.py teacher_lstm teacher_transformer
```

## 3. Compact models (M3)

Distillation data (teacher-labelled Wikipedia words), then the size sweep →
`bench/results/m3_sweep/accuracy.csv`:

```
uv run python -m tel.distill --teacher runs/teacher_lstm --extra data/interim/wiki_freq.tsv \
    --top 20000 --out data/distill/teacher_lstm_wiki20k
uv run python -m tel.sweep configs/sweep/{lstm,edit,transformer,lstm_kd,edit_kd}.yaml \
    --sizes 500k 1M 2M 5M --seeds 13 14 15 --jobs 3
uv run python -m tel.sweep configs/sweep/{lstm,edit,transformer,lstm_kd,edit_kd}.yaml \
    --sizes 500k 1M 2M 5M --seeds 13 14 15 --csv bench/results/m3_sweep/accuracy.csv
```

Encoder ablation (1M) → `bench/results/m3_ablation/accuracy.csv`:
`uv run python -m tel.sweep configs/sweep/{lstm,char_cnn,edit}.yaml --sizes 1M --seeds 13 14 15`,
then the same command with `--csv bench/results/m3_ablation/accuracy.csv`.

Data expansion (M3 step 5) → `bench/results/m3_expansion/`:

```
uv run python scripts/data/expand_candidates.py calibrate      # calibration_dev.csv
uv run python scripts/data/expand_candidates.py propose --top 3000   # review TSV (then human review)
uv run python scripts/data/expand_candidates.py import data/annotation/wiki_batch01_reviewed.tsv \
    --out data/expanded/train.tsv
uv run python -m tel.distill --teacher runs/teacher_lstm --gold data/expanded/train.tsv \
    --extra data/interim/wiki_freq.tsv --top 20000 --out data/distill/teacher_lstm_wiki20k_exp
uv run python -m tel.sweep configs/sweep/{lstm_exp,edit_exp,lstm_kd_exp,edit_kd_exp}.yaml \
    --sizes 500k --seeds 13 14 15 --jobs 3
uv run python -m tel.sweep configs/sweep/{lstm,lstm_exp,lstm_kd,lstm_kd_exp,edit,edit_exp,edit_kd,edit_kd_exp}.yaml \
    --sizes 500k --seeds 13 14 15 --csv bench/results/m3_expansion/accuracy.csv
```

## 4. Export and quantization (M2, M4)

For each deployed run (`edit_kd_500k_s13`, `lstm_kd_500k_s13`; the M3 ablation
used the `*_1M_s13` runs):

```
R=runs/edit_kd_500k_s13
uv run python -m tel.export $R                                  # export/onnx
uv run python -m tel.export $R --runtime ncnn                   # export/ncnn
uv run --project envs/litert python -m tel.export $R --runtime litert   # export/tflite
uv run python -m tel.quantize $R/export/onnx --method dynamic   # export/onnx-int8dyn
uv run python -m tel.quantize $R/export/onnx --method static    # export/onnx-int8
uv run --project envs/litert python -m tel.quantize $R/export/tflite --method dynamic   # export/tflite-int8dyn
```

## 5. Hybrid lookup tables (M4) → `bench/results/m4_hybrid/coverage*.csv`

```
uv run python scripts/hybrid/build_tables.py                    # table_{1k,5k,10k,50k}.tsv
uv run python scripts/hybrid/build_tables.py --reviewed data/annotation/wiki_batch01_reviewed.tsv \
    --tag reviewed                                              # table_<N>_reviewed.tsv
```

## 6. Benchmarks

Desktop (one CSV per milestone; LiteRT needs its own env):

```
PYTHONPATH=src uv run python bench/bench_runner.py --device desktop --out bench/results/m4_quant/desktop.csv \
    --models runs/{edit,lstm}_kd_500k_s13/export/{onnx,onnx-int8dyn,onnx-int8}
PYTHONPATH=src uv run --project envs/litert python bench/bench_runner.py --device desktop \
    --out bench/results/m4_quant/desktop.csv --models runs/{edit,lstm}_kd_500k_s13/export/{tflite,tflite-int8dyn}
PYTHONPATH=src uv run python bench/bench_runner.py --device desktop --out bench/results/m4_hybrid/desktop.csv \
    --weighted --tables none data/interim/hybrid/table_{1k,5k,10k,50k}.tsv \
    --models runs/{edit,lstm}_kd_500k_s13/export/{onnx,ncnn}
```

Raspberry Pi 5 (no torch needed on the device): build a bundle on the desktop,
copy it over and run it there; copy `bench/results/<device>*.csv` and
`<device>.info.txt` back into the milestone folder.

```
bash bench/make_device_bundle.sh edit_kd_500k_s13 lstm_kd_500k_s13   # bench/tmp/device_bundle.tar.gz
# on the Pi, in the unpacked bundle:
bash bench/device_run.sh rpi5                                   # plain + hybrid passes
HYBRID_ONLY=1 bash bench/device_run.sh rpi5                     # hybrid pass only
FORMATS="tflite tflite-int8dyn" bash bench/device_run.sh rpi5   # only these exports
```

Record the device in `bench/devices.md` (`bench/device_info.sh` prints most
fields) and check the governor and throttling before each session.

## 7. Error analysis and paper outputs (M5)

```
uv run python scripts/analysis/error_analysis.py   # bench/results/m5_errors/
uv run python paper/figures/make_all.py            # paper/figures/out/: figures (pdf, png, tiff), tables (tex)
uv run python paper/order_refs.py                  # references in order of first citation
cd paper && latexmk -xelatex main.tex              # manuscript (Sadhana class, XeLaTeX)
```

| Paper item | Script | Inputs |
|---|---|---|
| Accuracy table | `tables.py` (`tab_accuracy`) | `accuracy_baselines.csv`, `m3_sweep/accuracy.csv` |
| Accuracy vs parameters | `fig_params.py` | `m3_sweep/accuracy.csv` |
| Pareto per device | `fig_pareto.py` | M2–M4 device CSVs |
| Quantization table | `tables.py` (`tab_quant`) | `m4_quant/` |
| Hybrid table | `tables.py` (`tab_hybrid`) | `m4_quant_hybrid/rpi5-hybrid.csv`, `m4_hybrid/coverage.csv` |
| Error tables | `tables.py` (`tab_errors`, `tab_error_examples`) | `m5_errors/` |
| Size sweep table | `tables.py` (`tab_sweep`) | `m3_sweep/accuracy.csv` |
| Encoder ablation table | `tables.py` (`tab_ablation`) | `m3_ablation/accuracy.csv`, `m3_ablation/rpi5.csv` |
| Data expansion table | `tables.py` (`tab_expansion`) | `m3_expansion/accuracy.csv` |

## 8. Release

Trained models and lookup tables for the GitHub release (`docs/models.md`):
`bash scripts/release/make_release.sh v1.0` → `bench/tmp/release/v1.0/`.
