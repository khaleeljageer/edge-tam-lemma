# tel — Tamil edge lemmatizer

Compact neural lemmatization for Tamil on edge devices (research code).

```
uv sync
uv run pytest
uv run python scripts/data/fetch_ud.py      # download UD Tamil treebanks
uv run python scripts/data/build_ud.py      # build data/processed/*.tsv
```

See `tamil-edge-lemmatizer-plan.md` and `docs/code_plan.md`.

Reproducing every result, table and figure: `docs/reproduce.md`. Trained models
and lookup tables are attached to the GitHub release (`docs/models.md`).

## License

- Code, models and data: Creative Commons Attribution-NonCommercial-ShareAlike
  4.0 International (CC BY-NC-SA 4.0, `LICENSE`).
- Sources: pairs derived from UD Tamil-TTB are CC BY-NC-SA 3.0, UD Tamil-MWTT
  pairs CC BY-SA 4.0, and the Wikipedia word lists and reviewed annotations
  (`data/annotation/`) CC BY-SA; see `docs/data_licenses.md`. The
  non-commercial term comes from TTB.
- Baseline tools (Stanza, ThamizhiMorph, tamil-lemmatizer) are external
  dependencies under their own licenses and are not redistributed.
