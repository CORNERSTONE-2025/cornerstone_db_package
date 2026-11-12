# `cornerstone.processing` — from the raw store to the inputs of the labeling

Independent of the ML stack. Every choice is a parameter (dataclasses with `to_dict()`, saved with
the artifacts that use them).

## Good data: `QualityFilter`

| parameter | default | meaning |
|---|---|---|
| `require_position` | `True` | drop records with an empty patch (`LAT_MIN`/`LON_MIN` null) — what the former database kept |
| `quality_mask` | `None` | drop records with any of these `QUALITY` bits |
| `quality_values` | `None` | keep only records whose `QUALITY` is exactly one of these values |
| `cmask_gt` | `None` | keep `CMASK` > value |
| `meangbt_gt` | `None` | keep `MEANGBT` > value |
| `max_abs_lon` | `None` | patch within ±lon of the central meridian |
| `exclude_years` | `()` | drop these years (year of the JSOC `T_OBS`, TAI) |

`QualityFilter.legacy_clean(series)` reproduces the rows of the former `SHARPC_clean_*` tables
(filters of `000_README_SHARPC_clean_NRT.pdf`; checked on the dump of 2025-06-06: identical row sets,
DEF 1,820,785 and NRT 1,260,158 records):

| | DEF | NRT |
|---|---|---|
| `QUALITY` | exactly `0x00000000` | exactly `0x00000400` (cosmic-ray bit, set on almost every NRT record) |
| `CMASK` | > 5000 | > 5000 |
| `MEANGBT` | > 0 | > 0 |
| years | all | 2013 excluded |

## Features: `FEATURE_SETS`, `derive`

`"all"`: the 22 SHARP keywords of the former `SHARPC_all`; `"clean"`: the 17 features of
`SHARPC_clean` with the normalizations `KEY_N1 = KEY / CMASK` (USFLUX, TOTUSJZ, TOTUSJH) and
`KEY_N2 = KEY / MEANGBT` (MEANGBZ, MEANGBH), computed on the fly from the raw records (equal to the
dump within its 6-digit rounding). Any list of keywords and `_N1`/`_N2` names is accepted.

`harp_spans` gives the first/last observation of every HARP after the filter ("observed at t" =
`start <= t <= end`, as in the former labeling); `harp_regions` the (HARP, NOAA region) pairs:
JSOC map for DEF, `NOAA_ARS` keyword for NRT.

## Flares: `FlareConfig` + `associate(store, series, quality, config)`

| parameter | default | meaning |
|---|---|---|
| `calibration` | `"science"` | GOES-8…15 fluxes divided by 0.7 (operational scaling removed); `"operational"`: as reported |
| `peak_fallback` | `"start"` | events without observed peak: use the start time (`"drop"`: discard) |
| `ambiguous` | `"position_then_flux"` | several HARPs observed at the peak contain the region: the patch containing the flare position (± `position_margin_deg`), otherwise the largest `USFLUX`; also `"position_then_all"`, `"flux"`, `"all"` |
| `position_margin_deg` | `2.0` | tolerance on the patch box (SWPC positions are whole degrees) |
| `max_gap_hours` | `24` | no candidate observed at the peak: the HARP with the nearest span, within this gap |

One satellite per flare: when SWPC lists the same flare from two GOES under one event number,
the line of the day's primary satellite is kept. The output has one row per (event, HARP) with
`rule` (`unique`, `nearest_span`, `position`, `position+flux`, `flux`, `all`), `n_candidates`
and `gap_hours`.

**Measured** on the store of November 2026 (33,525 GOES events 2010–2026, 28,401 with a NOAA region),
default parameters, ~2 s per series:

| | DEF | NRT |
|---|---|---|
| associated | 27,287 | 23,693 (NRT starts 2012-09) |
| unique / nearest span | 25,926 / 923 | 21,844 / 974 |
| ambiguous → position / position+flux / flux | 140 / 34 / 264 | 292 / 29 / 554 |

Against the flare list of the former database (12,888 events associated to NRT HARPs): 99.65% are
associated again, 98.1% to the same HARP; the rest are ambiguous cases resolved by a different
rule, or HARPs that did not contain the region at the peak time.
