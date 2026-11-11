"""Per-HARP summaries of the SHARP records: observation spans and associated NOAA regions."""
from __future__ import annotations

import pandas as pd

from ..data import DataStore, harp_noaa
from .quality import QualityFilter


def harp_spans(store: DataStore, series: str, quality: QualityFilter = QualityFilter()) -> pd.DataFrame:
    """First and last observation (UTC) and number of records of every HARP, after ``quality``.

    A HARP is "observed at t" when ``start <= t <= end``: the same rule as the former labeling
    (``t_X between min(T_OBS) and max(T_OBS)``).
    """
    cols = ["HARPNUM", "T_OBS", *quality.columns()]
    df = quality.apply(store.read_sharp(series, columns=list(dict.fromkeys(cols))))
    g = df.groupby("HARPNUM")["T_OBS"]
    out = pd.DataFrame({"start": g.min(), "end": g.max(), "n_obs": g.size()}).reset_index()
    return out.astype({"HARPNUM": "int32"})


def harp_regions(store: DataStore, series: str, quality: QualityFilter = QualityFilter()) -> pd.DataFrame:
    """(HARPNUM, NOAA_AR) pairs used to associate flares to the HARPs of ``series``.

    DEF: the JSOC map ``all_harps_with_noaa_ars.txt`` (``cornerstone.data.harp_noaa``).
    NRT: the regions listed in the ``NOAA_ARS`` keyword of the NRT records (NRT HARPs have their
    own numbers and are not in the map), over the records kept by ``quality``.
    """
    if series == "DEF":
        return harp_noaa.load(store)
    cols = ["HARPNUM", "NOAA_ARS", *quality.columns()]
    df = quality.apply(store.read_sharp(series, columns=list(dict.fromkeys(cols))))
    pairs = df.dropna(subset=["NOAA_ARS"]).drop_duplicates(["HARPNUM", "NOAA_ARS"])
    pairs = pairs.assign(NOAA_AR=pairs["NOAA_ARS"].str.split(",")).explode("NOAA_AR")
    pairs["NOAA_AR"] = pd.to_numeric(pairs["NOAA_AR"].str.strip(), errors="coerce")
    pairs = pairs[pairs["NOAA_AR"] > 0][["HARPNUM", "NOAA_AR"]].drop_duplicates()
    return pairs.astype("int32").sort_values(["HARPNUM", "NOAA_AR"]).reset_index(drop=True)
