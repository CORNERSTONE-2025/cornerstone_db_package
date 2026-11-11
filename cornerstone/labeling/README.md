# `cornerstone.labeling` — parametric labeling

Replaces the static tables `DF_<type>_<class>_delta<h>_three_classes` and
`Cornerstone.splitting_labeling`. Definitions in `core.py`; in short, for every issue time `t_X`
and HARP observed at `t_X`, flare window `[t_X, t_X + W]`:

- **prediction label** (threshold `S_p`): `YES` if a flare ≥ `S_p` peaks in the window, otherwise
  NO1 (no flare ≥ `S_p` before nor after), NO2 (only after), NO3 (before and after), NO4 (only
  before) — original naming;
- **stratification class** (threshold `S_s`, default `S_p`): class letter of the strongest flare of
  the window when ≥ `S_s`, otherwise NO1–NO4 computed with `S_s`;
- "before"/"after": the whole past/future of the HARP by default, or `L` hours
  (`strat_window_hours`) before `t_X` and after `t_X + W`.

| `LabelConfig` | default | |
|---|---|---|
| `threshold` | `"M"` | `S_p`: class (`"C"`, `"M5"`) or flux in W/m² |
| `flare_window_hours` | 24 | `W` |
| `step_hours` | 24 | grid of `t_X`, aligned to `grid_origin` (1996-01-01 00:00 UTC) |
| `strat_threshold` | `None` (= `S_p`) | `S_s` |
| `strat_window_hours` | `None` (unlimited) | `L` |
| `start`, `end` | data range | first/last `t_X` |

```python
from cornerstone.data import default_store
from cornerstone.labeling import LabelConfig, label_from_store
labels, meta = label_from_store(default_store(), "DEF", LabelConfig(threshold="M", strat_threshold="C"))
```

`meta` holds every parameter (labeling, quality filter, flare calibration/association), the data
provenance and the counts: it is saved with the label artifacts.

Speed (laptop): DEF daily grid 48k rows in 5 s; NRT hourly grid 1.04 M rows in 7 s.

## Reproduction of the static tables

`validation.compare_with_legacy` feeds the inputs of the former labeling (flare list and
observation spans of the dump) to `label` with the static parameters (`W = L = 24 h`, daily grid):

| NRT | C | M | X |
|---|---|---|---|
| same (t_X, HARP) rows | 36,076 / 36,076 | 36,076 / 36,076 | 36,076 / 36,076 |
| same label, as is | 98.35% | 99.36% | 99.91% |
| same label, with the two quirks below, t_X ≤ 2024-11-28 | **100%** | **100%** | **100%** |

The differences have two causes, both in the former inputs, not in the definitions:
1. flares exactly at the threshold (C1.0, M1.0, X1.0) were not counted (MySQL FLOAT: 1e-4 stored
   as 0.99999997e-4 < 0.0001); here "≥ threshold" includes them (likewise `label_u` gave C for M1.0);
2. the static tables of late 2024–2025 were computed with a newer flare list than the one in the dump
   (which ends on 2024-12-31): YES there have no flare in the dump list.

DEF static tables cannot be checked the same way: their flare table (`NOAA_SHARPC_DEF_AR`) is not in
the dump.
