"""Bootstrap of the Parquet store from a SQLite database built from the former MySQL dump.

The dump (``Cornerstone.db_update``) is no longer needed once the store exists: this import is
run once, to create the store from a database somebody already has, or to prepare the Parquet
snapshot published for everybody.

Only the raw SHARP tables (``SHARPC_all_DEF`` / ``SHARPC_all_NRT``) become store data. The other
tables of the dump (flare list associated by the former pipeline, static label tables) are not
imported: GOES flares and labels come from the sources and from ``cornerstone.labeling``
(``cornerstone.labeling.validation`` compares with the static tables reading the database itself).
"""
from __future__ import annotations

import inspect
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


def bootstrap_from_sqlite(sqlite_path: str | Path, store: DataStore,
                          series: Iterable[str] = ("DEF", "NRT"),
                          log: Callable[[str], None] = say) -> dict:
    """Create the store from a legacy SQLite database (raw SHARP tables only)."""
    store.drop_obsolete(log=log)
    return {f"sharp/{s}": import_sharp_from_sqlite(sqlite_path, store, s, log=log) for s in series}


# ---------------------------------------------------------------------------------------- snapshot
SNAPSHOT_PARTIAL = ".snapshot-partial"      # kept between attempts: a new call resumes from it
_DRIVE_DOWNLOAD = "https://drive.usercontent.google.com/download"


def _drive_get(file_id: str, dst: Path, retries: int = 5, wait: float = 10.0) -> None:
    """Download one public Drive file to ``dst`` (direct endpoint, retries with backoff).

    gdown's per-file download parses the HTML of drive.google.com/uc and, after a few large files,
    stops finding the link ("Cannot retrieve the public link"), while the direct endpoint keeps
    working: gdown is only used to list the folder.
    """
    import requests

    tmp = dst.with_name(dst.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            with requests.get(_DRIVE_DOWNLOAD, params={"id": file_id, "export": "download", "confirm": "t"},
                              stream=True, timeout=(30, 300)) as r:
                r.raise_for_status()
                if "text/html" in r.headers.get("content-type", ""):
                    raise RuntimeError("Drive answered with a web page instead of the file (quota or permissions)")
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            tmp.replace(dst)
            return
        except Exception as err:                                   # noqa: BLE001 - network: retry
            tmp.unlink(missing_ok=True)
            if attempt == retries:
                raise RuntimeError(f"{dst.name}: download failed after {retries} attempts ({err})") from err
            time.sleep(wait * attempt)


def bootstrap_from_snapshot(store: DataStore, url: str | None = None,
                            log: Callable[[str], None] = say) -> dict:
    """Create the store from the published Parquet snapshot (a public Google Drive folder with the
    store layout: ``manifest.json``, ``raw/...``). Datasets of earlier versions listed by the
    snapshot (``reference/...``, tables of the former dump) are skipped.

    Every file is checked against the sha256 of the snapshot manifest. Files are downloaded into
    ``<root>/.snapshot-partial`` and moved into the store only when all of them are verified; if
    the download stops (network, Drive quota), calling again resumes it, skipping verified files.
    After the snapshot, ``update_all`` downloads only what changed since it was made.
    """
    import json
    import shutil

    import gdown

    from .config import load_config
    from .store import MANIFEST, OBSOLETE_PREFIXES, sha256

    url = url or load_config().get("bootstrap_url")
    if not url:
        raise ValueError("No snapshot URL: set bootstrap_url in cornerstone/configs/data_config.json")
    store.root.mkdir(parents=True, exist_ok=True)
    tmp = store.root / SNAPSHOT_PARTIAL
    tmp.mkdir(exist_ok=True)

    log(f"Listing the Parquet snapshot {url}")
    kwargs = {"url": url, "output": str(tmp), "quiet": True, "skip_download": True}
    # gdown < 6 stops at 50 files per folder unless remaining_ok; gdown 6 dropped the argument
    if "remaining_ok" in inspect.signature(gdown.download_folder).parameters:
        kwargs["remaining_ok"] = True
    listing = gdown.download_folder(**kwargs) or []
    files = {Path(f.local_path).resolve().relative_to(tmp.resolve()).as_posix(): f.id for f in listing}
    if MANIFEST not in files:
        raise RuntimeError("The snapshot folder is empty, not public, or has no manifest.json")

    _drive_get(files[MANIFEST], tmp / MANIFEST)
    snap = json.loads((tmp / MANIFEST).read_text())
    snap["datasets"] = {k: v for k, v in snap["datasets"].items() if not k.startswith(OBSOLETE_PREFIXES)}
    expected = {(_dataset_dir(name) / f).as_posix(): info.get("sha256")
                for name, d in snap["datasets"].items() for f, info in d["files"].items()}
    missing = sorted(set(expected) - set(files))
    if missing:
        raise RuntimeError(f"Files of the manifest not in the snapshot folder: {missing[:5]}"
                           f"{' ...' if len(missing) > 5 else ''}")

    bad = []
    for n, (rel, digest) in enumerate(sorted(expected.items()), 1):
        p = tmp / rel
        if p.exists() and (not digest or sha256(p) == digest):
            continue                                               # verified by a previous attempt
        log(f"[{n}/{len(expected)}] {rel}")
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            _drive_get(files[rel], p)
        except RuntimeError as err:
            log(str(err))
            bad.append(rel)
            continue
        if digest and sha256(p) != digest:
            p.unlink()
            bad.append(rel)
    if bad:
        raise RuntimeError(f"Snapshot files not downloaded or corrupted: {bad[:5]}{' ...' if len(bad) > 5 else ''}"
                           " - run the command again to resume (verified files are kept)")
    for rel in expected:
        dst = store.root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        (tmp / rel).replace(dst)
    store.drop_obsolete(log=log)
    store.save_manifest(snap)
    shutil.rmtree(tmp, ignore_errors=True)
    log(f"Snapshot installed in {store.root}")
    return snap


def _dataset_dir(name: str) -> Path:
    """Folder of a manifest dataset inside the store."""
    if name.startswith("sharp/"):
        return Path("raw/sharp") / name.split("/", 1)[1]
    return {"goes/swpc_events": Path("raw/goes"), "harp_noaa": Path("raw/harp_noaa")}.get(name, Path(name))
