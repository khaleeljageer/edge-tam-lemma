# Trained models and lookup tables (release v1.0)

Models are not stored in git (`runs/` is ignored); they are attached to the GitHub
release `v1.0` of https://github.com/khaleeljageer/edge-tam-lemma, built with
`bash scripts/release/make_release.sh v1.0` (deterministic archives, checksums in
`SHA256SUMS`). Licence: CC BY-NC-SA 4.0, like the rest of the repository.

| File | Contents |
|---|---|
| `tel-models-v1.0.tar.gz` | `runs/edit_kd_500k_s13/`, `runs/lstm_kd_500k_s13/` (the deployed 0.5M distilled students) and `runs/teacher_lstm/` (15.8M teacher): PyTorch checkpoint, config, tokenizer, metrics; the students also carry every export (`export/onnx`, `onnx-int8dyn`, `onnx-int8`, `tflite`, `tflite-int8dyn`, `ncnn`) |
| `tel-tables-v1.0.tar.gz` | hybrid lookup tables `data/interim/hybrid/table_{1k,5k,10k,50k}.tsv` and their `_reviewed` versions |
| `wiki_freq-v1.0.tsv.gz` | Tamil Wikipedia word-frequency list (`tawiki-20261001`, CC BY-SA), `data/interim/wiki_freq.tsv` |

Unpack in the repository root; the paths match the code:

```
tar xzf tel-models-v1.0.tar.gz
tar xzf tel-tables-v1.0.tar.gz
gunzip -c wiki_freq-v1.0.tsv.gz > data/interim/wiki_freq.tsv
```

Lemmatize with an exported model (needs only numpy and the runtime, no torch;
LiteRT exports need `uv run --project envs/litert`):

```python
from tel.runtime.backends import load_lemmatizer
lem = load_lemmatizer("runs/edit_kd_500k_s13/export/onnx-int8dyn", threads=1)
lem.lemmatize(["மரங்கள்", "வீட்டில்", "வாழ்ந்தான்"])   # ['மரம்', 'வீடு', 'வாழ்']
```

With a lookup table in front (`tel.hybrid.HybridLemmatizer`), see `bench/bench_runner.py
--tables`. Every other run in the paper is re-created from its config in `configs/`
(`docs/reproduce.md`).

## Publishing a release (maintainer)

```
bash scripts/release/make_release.sh v1.0
gh release create v1.0 bench/tmp/release/v1.0/* --title "v1.0" \
    --notes "Trained models and lookup tables for the paper; see docs/models.md."
```
