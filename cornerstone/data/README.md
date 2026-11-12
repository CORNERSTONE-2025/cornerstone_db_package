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
├── raw/harp_noaa/harp_noaa.parquet  # JSOC HARP <-> NOAA map, one row per pair
├── raw/goes/swpc_events_<year>.parquet        # GOES X-ray events (NOAA SWPC edited events)
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

## Sources and updates

| dataset | source | update |
|---|---|---|
| `raw/sharp/DEF`, `raw/sharp/NRT` | JSOC `show_info`, series `hmi.sharp_720s` / `hmi.sharp_720s_nrt`, 100 HARPs per request, 3 requests in parallel, retry with backoff | incremental by block (below); `--full` downloads everything |
| `raw/harp_noaa` | JSOC `all_harps_with_noaa_ars.txt` (definitive HARPs) | always downloaded again (35 kB) |
| `raw/goes` | NOAA SWPC edited event reports, `ftp://ftp.swpc.noaa.gov/pub/warehouse/<year>/` | past years once; the last stored year and the current one again |

`show_info` instead of `drms`: `drms` uses the JSON `jsoc_info`, which on 100 HARPs with all the
keywords answers `status 6`; `show_info` returns the same records as tab-separated text.

**Which SHARP blocks are downloaded.** HARP numbers are assigned in time order and a HARP is observed
from its creation. A block (1000 HARPs) is complete when, at the time its data were taken
(`data_until`: download time, or the newest record of the dump for imported blocks), its newest
record was already 60 days old. Missing blocks, incomplete blocks and the block of the last HARP
number are downloaded; requests answered are cached, so an interrupted download resumes.

**Records.** JSOC also returns the observations where the HARP patch is empty (`LAT_MIN`/`LON_MIN`
null, `USFLUX` = 0). The former database had removed them (checked: membership coincides 100% with
"position not null" on 100 HARPs), so blocks imported from the dump lack them while downloaded
blocks have them. The processing step selects the records with a position by default.

**GOES events.** `XRA` lines of the reports (satellite, begin/max/end, class, NOAA region) with the
position of the `FLA` line of the same event. Checked against the former flare list (12,888 events):
every event found, same class, region and position; same peak time except where the report has no
peak (`////`, the former list had put a time) and a few events listed twice. Notes for processing:
- the flux is stored as reported (`flux_reported`, `satellite`): GOES-13/14/15 fluxes carry the
  operational 0.7 scaling, the calibration choice is made when the events are used;
- SWPC groups several `XRA` lines under one event number (`event`): distinct flares, or the same
  flare from two satellites (e.g. G14 and G15); `event_id` adds a sequence number and is unique;
- reports that appear in two yearly archives are kept once, in their own year;
- 7 events in the former list period have no peak time (`tmax` null).

**Measured** (November 2026, from the store bootstrapped from the 2025-11-01 dump): first update
17 min — SHARP DEF 4 blocks / 33 requests in 10 min (store 4.62 → 5.63 M records, last HARP 15222,
data to 2026-10), NRT 4 blocks / 32 requests in 7 min (4.00 → 4.75 M, last HARP 14198, data to
today); GOES 2010–2026: 33,536 events, ~5 min (the current year is fetched day by day). Records
present in the dump and downloaded again agree to the rounding of the dump (6 significant digits,
JSOC now prints 7: relative differences ≤ 5·10⁻⁶).

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
python -m cornerstone.data bootstrap-snapshot          # first install: published Parquet snapshot
python -m cornerstone.data update                      # then: incremental update from the sources
python -m cornerstone.data bootstrap-sqlite db/cornerstone.sqlite   # or: from the former database
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

## Published snapshot

`bootstrap_url` in `cornerstone/configs/data_config.json` points to the public Drive folder of the
snapshot of 2026-11-12 (`manifest.json`, `raw/`, `reference/`; ~630 MB, raw data as downloaded from
the sources, no `.cache`). `bootstrap-snapshot` lists the folder, downloads every file of its
manifest into `<root>/.snapshot-partial`, checks the sha256 of each one and moves them into the
store only when all of them are verified. If Drive interrupts the download (network, quota),
running the command again resumes it, skipping the files already verified.

To publish a new snapshot: upload `manifest.json`, `raw/` and `reference/` of an up-to-date store
(without `.cache`) to a new Drive folder, share it as "anyone with the link" (reader) and set its
URL as `bootstrap_url`.
