# tel — Tamil edge lemmatizer

Compact neural lemmatization for Tamil on edge devices (research code).

```
uv sync
uv run pytest
uv run python scripts/data/fetch_ud.py      # download UD Tamil treebanks
uv run python scripts/data/build_ud.py      # build data/processed/*.tsv
```

See `tamil-edge-lemmatizer-plan.md` and `docs/code_plan.md`.

Reproducing every result, table and figure: `docs/reproduce.md`.

## License

- Code: GNU General Public License v3.0 (`LICENSE`).
- Data: pairs derived from UD Tamil-TTB are CC BY-NC-SA 3.0 (non-commercial),
  UD Tamil-MWTT pairs CC BY-SA 4.0, and the Wikipedia word lists and reviewed
  annotations (`data/annotation/`) CC BY-SA; see `docs/data_licenses.md`.
  A combined release of the dataset is therefore CC BY-NC-SA.
- Baseline tools (Stanza, ThamizhiMorph, tamil-lemmatizer) are external
  dependencies under their own licenses and are not redistributed.
