"""Bootstrap of the Parquet store from a SQLite database built from the former MySQL dump.

The dump (``Cornerstone.db_update``) is no longer needed once the store exists: this import is
run once, to create the store from a database somebody already has, or to prepare the Parquet
snapshot published for everybody.

Only the raw SHARP tables (``SHARPC_all_DEF`` / ``SHARPC_all_NRT``) become store data. The other
tables of the dump (flare list associated by the former pipeline, static label tables) are kept,
unchanged, under ``reference/legacy`` to compare the new labeling with the old one.
"""
from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
import pyarrow as pa

from . import sharp
from .store import HARP_BLOCK, DataStore, block_name, block_start, say

LEGACY_SHARP_TABLE = "SHARPC_all_{series}"
LEGACY_REFERENCE_TABLES = ("NOAA_SHARP_AR",)          # plus every DF_*_three_classes(_unified) table


def _connect_ro(path: str | Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)


def import_sharp_from_sqlite(sqlite_path: str | Path, store: DataStore, series: str,
                             table: str | None = None, chunk_rows: int = 250_000,
                             log: Callable[[str], None] = say) -> dict:
    """Copy one raw SHARP table of a legacy SQLite database into the store (one file per HARP block).

    Rows are streamed in (HARPNUM, T_OBS) order, so memory stays bounded by ``chunk_rows`` plus
    one block of HARPs.
    """
    table = table or LEGACY_SHARP_TABLE.format(series=series)
    cols = ", ".join(f'"{k}"' for k in sharp.KEYWORDS)
    t0 = time.time()
    files: dict[str, dict] = {}
    pending: list[pd.DataFrame] = []
    current: int | None = None

    def flush():
        if current is None or not pending:
            return
        df = sharp.normalize(pd.concat(pending, ignore_index=True))
        files[block_name(current)] = {**store.write_sharp_block(series, current, df), "source": "legacy-sqlite",
                                      "max_t_obs": df["T_OBS"].max().isoformat() if len(df) else None}
        log(f"   {series} HARP {current}-{current + HARP_BLOCK - 1}: {len(df):,} rows "
            f"({time.time() - t0:.0f}s)")
        pending.clear()

    with closing(_connect_ro(sqlite_path)) as con:
        n = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        log(f"Importing {table} ({n:,} rows) -> {store.sharp_dir(series)}")
        query = f'SELECT {cols} FROM "{table}" ORDER BY HARPNUM, T_OBS'
        for chunk in pd.read_sql_query(query, con, chunksize=chunk_rows, dtype={"QUALITY": "string"}):
            blocks = chunk["HARPNUM"].map(block_start)
            for b, part in chunk.groupby(blocks, sort=True):
                if b != current:
                    flush()
                    current = b
                pending.append(part)
        flush()

        # the dump is a snapshot: its newest record tells until when the data are complete
        until = con.execute(f'SELECT MAX(T_OBS) FROM "{table}"').fetchone()[0]
    until_utc = sharp.parse_t_obs(pd.Series([until])).iloc[0].isoformat() if until else None
    for f in files.values():
        f["data_until"] = until_utc

    # blocks of a previous import that no longer have data would mix old and new records
    for old in store.sharp_files(series):
        if old.name not in files:
            old.unlink()
    rows = sum(f["rows"] for f in files.values())
    if rows != n:
        log(f"   note: {n - rows:,} rows without T_OBS/HARPNUM or duplicated were dropped")
    return store.record_dataset(
        f"sharp/{series}", files, series=sharp.SERIES[series], time_scale="UTC (converted from TAI)",
        source={"kind": "legacy-sqlite", "file": Path(sqlite_path).name, "table": table,
                "source_rows": n,
                # the former pipeline kept only the records with a patch position: empty patches
                # (LAT_MIN/LON_MIN null, USFLUX = 0) are absent from these blocks, while blocks
                # downloaded from JSOC hold every record
                "records": "with patch position only"},
        duration_s=round(time.time() - t0, 1))


def legacy_reference_tables(con: sqlite3.Connection) -> list[str]:
    names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    return [t for t in names if t in LEGACY_REFERENCE_TABLES
            or (t.startswith("DF_") and t.endswith(("_three_classes", "_three_classes_unified")))]


def import_legacy_references(sqlite_path: str | Path, store: DataStore,
                             log: Callable[[str], None] = say) -> dict:
    """Copy the flare list and the static label tables of the legacy database, values unchanged."""
    out = store.reference_dir("legacy")
    files: dict[str, dict] = {}
    with closing(_connect_ro(sqlite_path)) as con:
        for t in legacy_reference_tables(con):
            df = pd.read_sql_query(f'SELECT * FROM "{t}"', con)
            name = t.removesuffix("_unified") + ".parquet"
            files[name] = store.write_table(out / name, pa.Table.from_pandas(df, preserve_index=False))
            log(f"   reference {t}: {len(df):,} rows")
    return store.record_dataset(
        "reference/legacy", files,
        note="Tables of the former MySQL dump, values unchanged (times as text, as in the dump): "
             "flare list associated to NRT HARPs and static label tables (*_unified). For comparison only.",
        source={"kind": "legacy-sqlite", "file": Path(sqlite_path).name})


def bootstrap_from_sqlite(sqlite_path: str | Path, store: DataStore,
                          series: Iterable[str] = ("DEF", "NRT"), references: bool = True,
                          log: Callable[[str], None] = say) -> dict:
    """Create the store from a legacy SQLite database: raw SHARP tables and reference tables."""
    out = {}
    for s in series:
        out[f"sharp/{s}"] = import_sharp_from_sqlite(sqlite_path, store, s, log=log)
    if references:
        out["reference/legacy"] = import_legacy_references(sqlite_path, store, log=log)
    return out


# ---------------------------------------------------------------------------------------- snapshot
def bootstrap_from_snapshot(store: DataStore, url: str | None = None,
                            log: Callable[[str], None] = say) -> dict:
    """Create the store from the published Parquet snapshot (a Google Drive folder with the store
    layout: ``manifest.json``, ``raw/...``). Every file is checked against the sha256 of the
    snapshot manifest; files that do not match are removed and the call raises.

    After the snapshot, ``update_all`` downloads only what changed since it was made.
    """
    import inspect
    import json
    import tempfile

    import gdown

    from .config import load_config
    from .store import MANIFEST, sha256

    url = url or load_config().get("bootstrap_url")
    if not url:
        raise ValueError("No snapshot URL: set bootstrap_url in cornerstone/configs/data_config.json")
    store.root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=store.root, prefix=".snapshot-") as tmp:
        log(f"Downloading the Parquet snapshot from {url}")
        kwargs = {"url": url, "output": tmp, "quiet": False}
        # gdown < 6 stops at 50 files per folder unless remaining_ok; gdown 6 dropped the argument
        if "remaining_ok" in inspect.signature(gdown.download_folder).parameters:
            kwargs["remaining_ok"] = True
        paths = gdown.download_folder(**kwargs)
        if not paths:
            raise RuntimeError("The snapshot folder is empty or not reachable")
        tmp = Path(tmp)
        snap = json.loads((tmp / MANIFEST).read_text())
        bad = []
        for name, d in snap["datasets"].items():
            for f, info in d["files"].items():
                if not info.get("sha256"):
                    continue
                p = tmp / _dataset_dir(name) / f
                if not p.exists() or sha256(p) != info["sha256"]:
                    bad.append(f"{name}/{f}")
        if bad:
            raise RuntimeError(f"Snapshot files missing or corrupted: {bad[:5]}{' ...' if len(bad) > 5 else ''}")
        for p in sorted(tmp.rglob("*.parquet")):
            dst = store.root / p.relative_to(tmp)
            dst.parent.mkdir(parents=True, exist_ok=True)
            p.replace(dst)
        store.save_manifest(snap)
    log(f"Snapshot installed in {store.root}")
    return snap


def _dataset_dir(name: str) -> Path:
    """Folder of a manifest dataset inside the store."""
    if name.startswith("sharp/"):
        return Path("raw/sharp") / name.split("/", 1)[1]
    return {"goes/swpc_events": Path("raw/goes"), "harp_noaa": Path("raw/harp_noaa"),
            "reference/legacy": Path("reference/legacy")}.get(name, Path(name))
