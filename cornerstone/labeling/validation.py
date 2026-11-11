"""Reproduction of the former static label tables (``DF_<type>_<class>_delta24_three_classes``).

The former labels were computed from the flare list of the dump (``NOAA_SHARP_AR``, associated
to the NRT HARPs) and the observation spans of ``SHARPC_all_<type>`` (TAI times compared with UTC
flare times as plain strings). ``compare_with_legacy`` feeds exactly those inputs to
``cornerstone.labeling.label`` with the static parameters and compares row by row.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

from .core import LabelConfig, label

STATIC = dict(flare_window_hours=24, step_hours=24, strat_window_hours=24, grid_origin="1996-01-01T00:00:00Z")


def legacy_inputs(sqlite_path: str | Path, series: str = "NRT") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Spans and flares exactly as the former labeling saw them (naive times read as UTC)."""
    with closing(sqlite3.connect(f"file:{Path(sqlite_path).resolve()}?mode=ro", uri=True)) as con:
        spans = pd.read_sql_query(f'SELECT HARPNUM, MIN(T_OBS1) AS start, MAX(T_OBS1) AS "end" '
                                  f'FROM "SHARPC_all_{series}" GROUP BY HARPNUM', con)
        flares = pd.read_sql_query('SELECT HARPNUM, tmax AS t_peak, fflux AS flux FROM NOAA_SHARP_AR', con)
    for df, cols in ((spans, ("start", "end")), (flares, ("t_peak",))):
        for c in cols:
            df[c] = pd.to_datetime(df[c]).dt.tz_localize("UTC")
    return spans, flares


def legacy_table(sqlite_path: str | Path, series: str, cls: str) -> pd.DataFrame:
    with closing(sqlite3.connect(f"file:{Path(sqlite_path).resolve()}?mode=ro", uri=True)) as con:
        df = pd.read_sql_query(f'SELECT t_X, HARPn AS HARPNUM, label, label_u FROM '
                               f'"DF_{series}_{cls}plus_delta24_three_classes_unified"', con)
    df["t_X"] = pd.to_datetime(df["t_X"]).dt.tz_localize("UTC")
    return df


def compare_with_legacy(sqlite_path: str | Path, series: str = "NRT", cls: str = "M",
                        end: str = "2025-04-30") -> dict:
    """Label with the static parameters on the legacy inputs and compare with the static table."""
    spans, flares = legacy_inputs(sqlite_path, series)
    old = legacy_table(sqlite_path, series, cls)
    new = label(spans, flares, LabelConfig(threshold=cls, end=end, **STATIC))
    m = old.merge(new, on=["t_X", "HARPNUM"], how="outer", suffixes=("_old", "_new"), indicator=True)
    both = m[m["_merge"] == "both"]
    diff = both[both["label_old"] != both["label_new"]]
    # strat_class with S_s = S_p is label_u of the static tables
    strat_old = both["label_u"].where(both["label_old"] != "YES", both["label_u"])
    return {"series": series, "class": cls, "rows_old": len(old), "rows_new": len(new),
            "only_old": int((m["_merge"] == "left_only").sum()), "only_new": int((m["_merge"] == "right_only").sum()),
            "same_label": float((both["label_old"] == both["label_new"]).mean()),
            "label_differences": diff.groupby(["label_old", "label_new"]).size().to_dict(),
            "same_strat_class": float((strat_old == both["strat_class"]).mean()),
            "differences": diff[["t_X", "HARPNUM", "label_old", "label_new"]]}
