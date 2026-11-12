# `cornerstone.dataset`: HDF5 datasets from the Parquet store

Replaces `Cornerstone.create_hdf5_from_db` (no database): labels from `cornerstone.labeling`,
SHARP records from the store, the same time windows.

```python
from cornerstone.data import DataStore
from cornerstone.labeling import LabelConfig, label_from_store
from cornerstone.dataset import build_dataset

store = DataStore("datastore")
labels, meta = label_from_store(store, "DEF", LabelConfig(threshold="M"))
build_dataset(store, "DEF", labels, "out/DEF_M.h5", features="clean", label_meta=meta)
```

| parameter | default | meaning |
|---|---|---|
| `features` | `"clean"` | `"clean"` (17 normalized features), `"all"` (22 raw keywords) or a list (`KEY`, `KEY_N1`, `KEY_N2`) |
| `quality` | `None` | records used in the windows; `None` = `QualityFilter.legacy_clean(series)` for `"clean"` (rows of `SHARPC_clean`), `QualityFilter()` otherwise |
| `window_hours` | `24` | window `[t_X - window_hours, t_X]` |
| `sampling_minutes` | `12` | grid step; `T_OBS` rounded to it, first record kept per step, NaN where none |
| `max_samples` / `seed` | `None` / `42` | random subset of the label rows (quick tests) |

Label rows whose window holds no record are left out and listed in `<file>.missing.csv` (as
the former missing-data log).

## Layout

| dataset | shape | content |
|---|---|---|
| `X` | (N, T, F) float32 | features, NaN = missing, T = `window_hours*60/sampling_minutes + 1` |
| `Y`, `label_<name>` | (N,) bytes | prediction label YES / NO1..NO4 |
| `label_u_<name>` | (N,) bytes | stratification class: C/M/X... or NO1..NO4 |
| `HARPNUM` | (N,) int32 | |
| `timestamps` | (N, T) bytes | `YYYY-mm-dd HH:MM:SS` (UTC) |
| `QUALITY` | (N,) bytes | most frequent QUALITY of the window |

`<name>` is `<threshold>plus_delta<W>`, e.g. `Mplus_delta24`. Attributes: `feature_names`,
`database_version` (feature set), `table_name` (`store:sharp/<series>`) and `metadata` (JSON: every
parameter of the dataset and of the labeling, data provenance, counts).

`SharpDataset` reads `label_<name>` as the target (YES vs NO*) and `label_u_<name>` as the
stratification class: the train/val/test splits stratify on it and `multi_class=True` predicts it
(NO* merged into NO). Files without `label_u_<name>` fall back to the prediction label.

## Checks

On the store of November 2026, labels with the static parameters (M, W = L = 24 h, daily):

| | DEF | NRT |
|---|---|---|
| label rows | 48,393 | 43,275 |
| samples (clean) | 21,325 | 17,457 |
| build time | 10 s | 10 s |
| file | 127 MB | 92 MB |

300 random windows per series are identical (NaN pattern and values) to the windows the former
`create_hdf5_from_db` extracted from `SHARPC_clean_*`.
