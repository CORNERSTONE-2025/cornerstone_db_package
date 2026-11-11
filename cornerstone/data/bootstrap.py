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
from .store import HARP_BLOCK, DataStore, block_name, block_start

LEGACY_SHARP_TABLE = "SHARPC_all_{series}"
LEGACY_REFERENCE_TABLES = ("NOAA_SHARP_AR",)          # plus every DF_*_three_classes(_unified) table


def _connect_ro(path: str | Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)


def import_sharp_from_sqlite(sqlite_path: str | Path, store: DataStore, series: str,
                             table: str | None = None, chunk_rows: int = 250_000,
                             log: Callable[[str], None] = print) -> dict:
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
        files[block_name(current)] = store.write_sharp_block(series, current, df)
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
                "source_rows": n},
        duration_s=round(time.time() - t0, 1))


def legacy_reference_tables(con: sqlite3.Connection) -> list[str]:
    names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    return [t for t in names if t in LEGACY_REFERENCE_TABLES
            or (t.startswith("DF_") and t.endswith(("_three_classes", "_three_classes_unified")))]


def import_legacy_references(sqlite_path: str | Path, store: DataStore,
                             log: Callable[[str], None] = print) -> dict:
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
                          log: Callable[[str], None] = print) -> dict:
    """Create the store from a legacy SQLite database: raw SHARP tables and reference tables."""
    out = {}
    for s in series:
        out[f"sharp/{s}"] = import_sharp_from_sqlite(sqlite_path, store, s, log=log)
    if references:
        out["reference/legacy"] = import_legacy_references(sqlite_path, store, log=log)
    return out
