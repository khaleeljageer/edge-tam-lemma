# tel — Tamil edge lemmatizer

Compact neural lemmatization for Tamil on edge devices (research code).

```
uv sync
uv run pytest
uv run python scripts/data/fetch_ud.py      # download UD Tamil treebanks
uv run python scripts/data/build_ud.py      # build data/processed/*.tsv
```

See `tamil-edge-lemmatizer-plan.md` and `docs/code_plan.md`.
