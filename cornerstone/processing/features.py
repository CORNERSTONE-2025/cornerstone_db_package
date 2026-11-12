"""SHARP features used by the models: raw keywords and normalized ("clean") versions.

Normalizations of the former ``SHARPC_clean_*`` tables (reproduced to the precision of the dump):

- ``<KEY>_N1 = KEY / CMASK`` (extensive quantities per patch pixel): ``USFLUX``, ``TOTUSJZ``, ``TOTUSJH``;
- ``<KEY>_N2 = KEY / MEANGBT``: ``MEANGBZ``, ``MEANGBH``.

The ``clean`` feature set also drops the keywords strongly correlated (r > 0.9) with a kept one:
MEANGAM, MEANSHR (-> SHRGT45), TOTPOT (-> MEANPOT), MEANALP (-> MEANJZH), SAVNCPP (-> ABSNJZH).
"""
from __future__ import annotations

import pandas as pd

N1 = ("USFLUX", "TOTUSJZ", "TOTUSJH")      # divided by CMASK
N2 = ("MEANGBZ", "MEANGBH")                # divided by MEANGBT

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "all": ("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX", "USFLUX", "MEANGAM", "MEANGBT", "MEANGBZ", "MEANGBH",
            "MEANJZD", "TOTUSJZ", "MEANALP", "MEANJZH", "TOTUSJH", "ABSNJZH", "SAVNCPP", "MEANPOT", "TOTPOT",
            "MEANSHR", "SHRGT45", "R_VALUE", "CMASK"),
    "clean": ("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX", "USFLUX_N1", "MEANGBT", "MEANGBZ_N2", "MEANGBH_N2",
              "MEANJZD", "TOTUSJZ_N1", "MEANJZH", "TOTUSJH_N1", "ABSNJZH", "MEANPOT", "SHRGT45", "R_VALUE", "CMASK"),
}


def resolve_features(features: str | list[str] | tuple[str, ...]) -> list[str]:
    """A named set (``"all"``, ``"clean"``) or an explicit list of raw/derived feature names."""
    if isinstance(features, str):
        if features not in FEATURE_SETS:
            raise ValueError(f"unknown feature set {features!r}; use one of {list(FEATURE_SETS)} or a list")
        return list(FEATURE_SETS[features])
    return list(features)


def source_columns(features: list[str]) -> list[str]:
    """Raw SHARP keywords needed to compute ``features``."""
    cols = []
    for f in features:
        if f.endswith("_N1"):
            cols += [f[:-3], "CMASK"]
        elif f.endswith("_N2"):
            cols += [f[:-3], "MEANGBT"]
        else:
            cols.append(f)
    return list(dict.fromkeys(cols))


def derive(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """``df`` with the derived columns of ``features`` added (division by 0 gives NaN)."""
    out = {}
    for f in features:
        if f in df.columns:
            continue
        if f.endswith("_N1"):
            den = df["CMASK"].astype("float64")
        elif f.endswith("_N2"):
            den = df["MEANGBT"].astype("float64")
        else:
            raise KeyError(f"feature {f!r} is neither a SHARP keyword nor a derived feature")
        num = df[f[:-3]].astype("float64")
        out[f] = (num / den.where(den != 0)).astype("float32")
    return df.assign(**out) if out else df


__all__ = ["FEATURE_SETS", "N1", "N2", "resolve_features", "source_columns", "derive"]
