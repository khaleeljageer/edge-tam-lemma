# Data licenses

| Dataset | Version / ref | License | Commercial use | Used for |
|---|---|---|---|---|
| UD Tamil-TTB | `master` (pin a release tag) | CC BY-NC-SA 3.0 | **No** | Gold lemmas |
| UD Tamil-MWTT | `master` (pin a release tag) | CC BY-SA 4.0 | Yes (share-alike) | Gold lemmas |
| Tamil Wikipedia dump | `tawiki-20261001-pages-articles` | CC BY-SA | Yes (share-alike) | Word-frequency list (`data/interim/wiki_freq.tsv`), unlabeled words for distillation; review batches in `data/annotation/` (words + reviewed lemmas) |
| ThamizhiMorph output | — | check before use | — | Candidate lemmas (not yet used) |
| Own news text | — | confirm usage rights | — | `test_news` (not yet used) |

Consequence: any released dataset that includes TTB-derived pairs inherits
NC-SA and must be released under CC BY-NC-SA. State this in the paper.

License texts are downloaded with the data to `data/raw/ud/*/LICENSE.txt`.

## Baseline tools (external dependencies, not redistributed)

| Tool | Version | License | Trained on | Notes |
|---|---|---|---|---|
| Stanza (Tamil) | 1.15.0 | Apache-2.0 (code); models follow their treebank | UD Tamil-TTB train (CC BY-NC-SA) | Scores ~96% on test words seen in TTB train vs ~65% otherwise: report `test_clean` |
| tamil-lemmatizer (PyPI) | 0.0.2 | MIT | undocumented ("curated Tamil lemma dataset") | Model downloaded from Hugging Face at runtime |
| ThamizhiMorph | commit `adbacceda5e8` | Apache-2.0 | rule-based FSTs (lexicons in repo) | Needs foma (`flookup`) |
