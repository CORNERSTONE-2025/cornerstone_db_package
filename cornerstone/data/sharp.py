"""SHARP keywords: schema and normalization of the raw records.

One record per HARP and observation (12-minute cadence), as served by JSOC for the series
``hmi.sharp_720s`` (definitive, ``DEF``) and ``hmi.sharp_720s_nrt`` (near real time, ``NRT``).
Raw data are stored unfiltered: the quality selection is a parametric processing step.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa

from .timescale import parse_t_obs

SERIES = {"DEF": "hmi.sharp_720s", "NRT": "hmi.sharp_720s_nrt"}

# keywords requested from JSOC, in this order
POSITION_KEYS = ("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX")
PHYSICAL_KEYS = ("USFLUX", "MEANGAM", "MEANGBT", "MEANGBZ", "MEANGBH", "MEANJZD", "TOTUSJZ", "MEANALP",
                 "MEANJZH", "TOTUSJH", "ABSNJZH", "SAVNCPP", "MEANPOT", "TOTPOT", "MEANSHR", "SHRGT45",
                 "R_VALUE")
KEYWORDS = ("T_OBS", "HARPNUM", *POSITION_KEYS, *PHYSICAL_KEYS, "CMASK", "QUALITY",
            "NOAA_AR", "NOAA_NUM", "NOAA_ARS")
FLOAT_KEYS = (*POSITION_KEYS, *PHYSICAL_KEYS, "CMASK")

# Stored schema. JSOC prints the keywords with at most 7 significant digits, so float32 keeps
# them; T_OBS is converted from TAI to UTC once, here.
SCHEMA = pa.schema(
    [pa.field("T_OBS", pa.timestamp("s", tz="UTC"), nullable=False),
     pa.field("HARPNUM", pa.int32(), nullable=False)]
    + [pa.field(k, pa.float32()) for k in FLOAT_KEYS]
    + [pa.field("QUALITY", pa.uint32()),
       pa.field("NOAA_AR", pa.int32()),
       pa.field("NOAA_NUM", pa.int16()),
       pa.field("NOAA_ARS", pa.string())]
)
COLUMNS = tuple(SCHEMA.names)


def _quality_to_int(q: pd.Series) -> pd.Series:
    """QUALITY is a 32-bit mask written in hex (``0x00010400``); integers pass through."""
    def conv(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        if isinstance(v, str):
            v = v.strip()
            if not v or v.upper() == "MISSING":
                return None
            return int(v, 16) if v.lower().startswith("0x") else int(float(v))
        return int(v)
    return pd.Series([conv(v) for v in q], index=q.index, dtype="UInt32")


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Raw SHARP records (strings or numbers, ``T_OBS`` in TAI) -> stored schema.

    Rows whose ``T_OBS`` or ``HARPNUM`` is missing cannot be placed in time or attributed to a
    region and are dropped. The result is sorted by (HARPNUM, T_OBS) without duplicates.
    """
    missing = [k for k in KEYWORDS if k not in df.columns]
    if missing:
        raise ValueError(f"SHARP records without the keywords {missing}")
    out = pd.DataFrame(index=df.index)
    out["T_OBS"] = parse_t_obs(df["T_OBS"])
    out["HARPNUM"] = pd.to_numeric(df["HARPNUM"], errors="coerce")
    for k in FLOAT_KEYS:
        out[k] = pd.to_numeric(df[k], errors="coerce").astype("float32")
    out["QUALITY"] = _quality_to_int(df["QUALITY"])
    out["NOAA_AR"] = pd.to_numeric(df["NOAA_AR"], errors="coerce").astype("Int32")
    out["NOAA_NUM"] = pd.to_numeric(df["NOAA_NUM"], errors="coerce").astype("Int16")
    ars = df["NOAA_ARS"].astype("string").str.strip()
    out["NOAA_ARS"] = ars.mask(ars.isin(["", "MISSING"]))
    out = out.dropna(subset=["T_OBS", "HARPNUM"])
    out["HARPNUM"] = out["HARPNUM"].astype("int32")
    out = out.sort_values(["HARPNUM", "T_OBS"], kind="stable")
    out = out.drop_duplicates(["HARPNUM", "T_OBS"], keep="last")
    return out.reset_index(drop=True)


def to_table(df: pd.DataFrame) -> pa.Table:
    """Normalized DataFrame -> Arrow table with the stored schema."""
    return pa.Table.from_pandas(df[list(COLUMNS)], schema=SCHEMA, preserve_index=False)
