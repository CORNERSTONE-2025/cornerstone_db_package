"""Local Parquet store of the CORNERSTONE data.

Layout under ``root`` (one folder, configurable)::

    manifest.json                      # datasets present, provenance, rows, sizes, checksums
    raw/sharp/DEF/harp_000000-000999.parquet   # SHARP keywords, one file per block of HARPNUMs,
    raw/sharp/NRT/...                          # sorted by (HARPNUM, T_OBS)
    reference/legacy/<table>.parquet   # tables of the former MySQL dump kept for comparisons

Files are written atomically (temporary file + rename), so an interrupted write never leaves a
truncated file, and a block can be rewritten on its own (incremental updates).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from . import sharp

HARP_BLOCK = 1000           # HARPNUMs per file: ~300k rows, a few tens of MB
ROW_GROUP_ROWS = 65_536     # row-group statistics let readers skip HARPs and times not requested
COMPRESSION = "zstd"
MANIFEST = "manifest.json"


def say(msg: str) -> None:
    """Default progress log: printed and flushed at once (logs redirected to files stay live)."""
    print(msg, flush=True)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def block_start(harpnum: int, block: int = HARP_BLOCK) -> int:
    return (int(harpnum) // block) * block


def block_name(start: int, block: int = HARP_BLOCK) -> str:
    return f"harp_{start:06d}-{start + block - 1:06d}.parquet"


class DataStore:
    """Parquet files of the CORNERSTONE data under one root folder."""

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root).expanduser().resolve()

    # ------------------------------------------------------------------ paths
    def sharp_dir(self, series: str) -> Path:
        if series not in sharp.SERIES:
            raise ValueError(f"Unknown SHARP series {series!r}: expected one of {list(sharp.SERIES)}")
        return self.root / "raw" / "sharp" / series

    def sharp_files(self, series: str) -> list[Path]:
        d = self.sharp_dir(series)
        return sorted(d.glob("harp_*.parquet")) if d.exists() else []

    def reference_dir(self, name: str = "legacy") -> Path:
        return self.root / "reference" / name

    # ------------------------------------------------------------------ writing
    def write_table(self, path: Path, table: pa.Table) -> dict:
        """Write ``table`` to ``path`` atomically; returns rows, bytes and sha256 of the file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".partial")
        # float columns: BYTE_STREAM_SPLIT encoding (~35% smaller than dictionary/plain on SHARP data)
        floats = [f.name for f in table.schema if pa.types.is_floating(f.type)]
        pq.write_table(table, tmp, compression=COMPRESSION, row_group_size=ROW_GROUP_ROWS,
                       use_byte_stream_split=floats or False,
                       use_dictionary=[c for c in table.column_names if c not in floats])
        os.replace(tmp, path)
        return {"rows": table.num_rows, "bytes": path.stat().st_size, "sha256": sha256(path)}

    def write_sharp_block(self, series: str, start: int, df: pd.DataFrame) -> dict:
        """Write the normalized SHARP records of the HARPs ``[start, start + HARP_BLOCK)``."""
        if len(df) and not df["HARPNUM"].between(start, start + HARP_BLOCK - 1).all():
            raise ValueError(f"Records outside the HARP block starting at {start}")
        return self.write_table(self.sharp_dir(series) / block_name(start), sharp.to_table(df))

    # ------------------------------------------------------------------ reading
    def sharp_dataset(self, series: str) -> ds.Dataset:
        files = self.sharp_files(series)
        if not files:
            raise FileNotFoundError(f"No SHARP {series} data in {self.sharp_dir(series)}")
        return ds.dataset([str(f) for f in files], format="parquet", schema=sharp.SCHEMA)

    def read_sharp(self, series: str, columns: Iterable[str] | None = None,
                   harpnums: Iterable[int] | None = None,
                   start: str | pd.Timestamp | None = None, end: str | pd.Timestamp | None = None
                   ) -> pd.DataFrame:
        """SHARP records of ``series``, optionally restricted to some HARPs and to [start, end] (UTC).

        Only the requested columns and the row groups that can match are read from disk.
        """
        flt = None
        if harpnums is not None:
            flt = ds.field("HARPNUM").isin([int(h) for h in harpnums])
        for bound, op in ((start, "__ge__"), (end, "__le__")):
            if bound is not None:
                ts = pd.Timestamp(bound)
                ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
                cond = getattr(ds.field("T_OBS"), op)(pa.scalar(ts.to_pydatetime(), type=pa.timestamp("s", tz="UTC")))
                flt = cond if flt is None else flt & cond
        cols = list(columns) if columns is not None else None
        return self.sharp_dataset(series).to_table(columns=cols, filter=flt).to_pandas()

    # ------------------------------------------------------------------ catalogue
    def catalog(self) -> pd.DataFrame:
        """One row per Parquet file (from the file footers, no data is read)."""
        rows = []
        for p in sorted(self.root.glob("**/*.parquet")):
            md = pq.ParquetFile(p).metadata
            rows.append({"dataset": str(p.parent.relative_to(self.root)), "file": p.name,
                         "rows": md.num_rows, "MB": round(p.stat().st_size / 1e6, 2)})
        return pd.DataFrame(rows, columns=["dataset", "file", "rows", "MB"])

    def load_manifest(self) -> dict:
        p = self.root / MANIFEST
        if not p.exists():
            return {"format": 1, "datasets": {}}
        return json.loads(p.read_text())

    def save_manifest(self, manifest: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        p = self.root / MANIFEST
        tmp = p.with_name(p.name + ".partial")
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        os.replace(tmp, p)

    def record_dataset(self, name: str, files: dict[str, dict], merge: bool = False, **info) -> dict:
        """Register dataset ``name`` in the manifest with its files and provenance.

        ``merge=False`` replaces the entry; ``merge=True`` adds/replaces only the given files
        (incremental updates), keeping the others and the previous provenance fields.
        """
        manifest = self.load_manifest()
        old = manifest["datasets"].get(name, {}) if merge else {}
        all_files = {**old.get("files", {}), **files}
        entry = {**{k: v for k, v in old.items() if k != "files"}, **info,
                 "updated_at": now_iso(), "files": all_files,
                 "rows": sum(f["rows"] for f in all_files.values()),
                 "bytes": sum(f["bytes"] for f in all_files.values())}
        manifest["datasets"][name] = entry
        self.save_manifest(manifest)
        return entry

    def forget_files(self, name: str, file_names: Iterable[str]) -> None:
        """Remove files from a dataset entry of the manifest (the files must already be deleted)."""
        manifest = self.load_manifest()
        entry = manifest["datasets"].get(name)
        if not entry:
            return
        for f in file_names:
            entry["files"].pop(f, None)
        entry["rows"] = sum(f["rows"] for f in entry["files"].values())
        entry["bytes"] = sum(f["bytes"] for f in entry["files"].values())
        self.save_manifest(manifest)
