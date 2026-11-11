"""HARP <-> NOAA active region map published by JSOC for the definitive HARPs.

``all_harps_with_noaa_ars.txt`` lists, for every definitive HARP that contains at least one NOAA
region, the comma-separated NOAA numbers. The relation is many-to-many: a HARP can hold several
NOAA regions and a NOAA region can appear in several HARPs (HARPs that follow each other, or
overlap in time). It is stored normalized, one row per (HARPNUM, NOAA_AR) pair.

NRT HARPs are numbered independently; for them the association comes from the ``NOAA_ARS``
keyword of ``hmi.sharp_720s_nrt`` (see ``cornerstone.data.sharp``).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable

import pandas as pd
import pyarrow as pa
import requests

from .store import DataStore, now_iso, say

URL = "http://jsoc.stanford.edu/doc/data/hmi/harpnum_to_noaa/all_harps_with_noaa_ars.txt"
SCHEMA = pa.schema([pa.field("HARPNUM", pa.int32(), nullable=False),
                    pa.field("NOAA_AR", pa.int32(), nullable=False)])


def parse(text: str) -> pd.DataFrame:
    """``HARPNUM NOAA_ARS`` text -> pairs (HARPNUM, NOAA_AR), sorted, without duplicates."""
    pairs = []
    for i, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line or line.startswith("#") or line.upper().startswith("HARPNUM"):
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"Line {i + 1} not understood: {line!r}")
        h = int(parts[0])
        pairs += [(h, int(a)) for a in parts[1].split(",") if a.strip()]
    df = pd.DataFrame(pairs, columns=["HARPNUM", "NOAA_AR"]).astype("int32")
    return df.drop_duplicates().sort_values(["HARPNUM", "NOAA_AR"]).reset_index(drop=True)


def harp_to_ars(pairs: pd.DataFrame) -> dict[int, list[int]]:
    out: dict[int, list[int]] = defaultdict(list)
    for h, a in zip(pairs["HARPNUM"], pairs["NOAA_AR"]):
        out[int(h)].append(int(a))
    return dict(out)


def ar_to_harps(pairs: pd.DataFrame) -> dict[int, list[int]]:
    out: dict[int, list[int]] = defaultdict(list)
    for h, a in zip(pairs["HARPNUM"], pairs["NOAA_AR"]):
        out[int(a)].append(int(h))
    return dict(out)


def path(store: DataStore):
    return store.root / "raw" / "harp_noaa" / "harp_noaa.parquet"


def download(store: DataStore, url: str = URL, log: Callable[[str], None] = say) -> dict:
    """Download the map (small, always the current version) and store it as Parquet."""
    r = requests.get(url, timeout=120, headers={"User-Agent": "cornerstone-data (python-requests)"})
    r.raise_for_status()
    df = parse(r.text)
    info = store.write_table(path(store), pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False))
    log(f"HARP-NOAA map: {df.HARPNUM.nunique():,} HARPs, {df.NOAA_AR.nunique():,} NOAA regions, {len(df):,} pairs")
    return store.record_dataset("harp_noaa", {path(store).name: {**info, "downloaded_at": now_iso()}},
                                source={"kind": "jsoc", "url": url})


def load(store: DataStore) -> pd.DataFrame:
    return pd.read_parquet(path(store))
