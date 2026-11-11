# `cornerstone.data` — local Parquet store

Data layer of CORNERSTONE, independent of the ML stack (it does not import torch or ClearML).
It replaces the MySQL dump / SQLite database as the storage of the raw data: the labeling and
the datasets are built from these files.

## Layout

```
<data_root>/                         # cornerstone/configs/data_config.json, or $CORNERSTONE_DATA_ROOT
├── manifest.json                    # datasets, provenance, rows, bytes and sha256 of every file
├── raw/sharp/DEF/harp_000000-000999.parquet   # hmi.sharp_720s keywords, 1000 HARPs per file
├── raw/sharp/NRT/...                          # hmi.sharp_720s_nrt
└── reference/legacy/*.parquet       # tables of the former dump, unchanged, for comparisons only
```

SHARP files hold the **raw** records (no quality filter: that is a parametric processing step),
sorted by `(HARPNUM, T_OBS)`:

| column | type | notes |
|---|---|---|
| `T_OBS` | timestamp UTC | converted once from TAI (`TAI − UTC` = 34–37 s since 2010) |
| `HARPNUM` | int32 | |
| `LAT_MIN` … `LON_MAX`, 17 SHARP parameters, `CMASK` | float32 | JSOC prints ≤ 7 significant digits: float32 keeps them |
| `QUALITY` | uint32 | bit mask (hex in JSOC) |
| `NOAA_AR`, `NOAA_NUM`, `NOAA_ARS` | int32, int16, string | `NOAA_ARS` null when the HARP has no NOAA region |

Files are written atomically and compressed with zstd; float columns use the BYTE_STREAM_SPLIT
encoding. Readers (pyarrow, pandas, DuckDB) read only the columns and row groups they need.

## Usage

```python
from cornerstone.data import default_store
store = default_store()
df = store.read_sharp("DEF", columns=["T_OBS", "HARPNUM", "USFLUX"], harpnums=[377],
                      start="2011-02-12", end="2011-02-16")
store.catalog()          # files, rows, MB
```

```bash
python -m cornerstone.data status
python -m cornerstone.data bootstrap-sqlite db/cornerstone.sqlite   # once, from the former database
```

## Bootstrap from the former database

`bootstrap-sqlite` converts `SHARPC_all_DEF` / `SHARPC_all_NRT` of a database built by
`Cornerstone.db_update()` (85 s on a laptop, 1.3 GB of RAM at peak) and copies the flare list and
the static label tables to `reference/legacy`. Measured on the 2025-11-01 dump:

| | rows | size |
|---|---|---|
| SQLite database (all tables) | | 4.3 GB |
| dump `cornerstone.sql.gz` | | 1.46 GB |
| `raw/sharp/DEF` | 4,622,872 | 299 MB |
| `raw/sharp/NRT` | 3,997,442 | 253 MB |
| `reference/legacy` | 246,447 | 6 MB |

Checked row by row against the database: identical keys, times, QUALITY and NOAA fields;
float32 relative error ≤ 6·10⁻⁸.

The store folder is also the snapshot to publish (Google Drive) so that new installations
download ~550 MB of Parquet instead of building the database from the dump.
