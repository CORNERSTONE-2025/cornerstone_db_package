"""Selection of the SHARP records used downstream ("good data"), as an explicit parameter set.

The store keeps every record; a ``QualityFilter`` decides which ones enter the HARP observation
spans, the labels and the datasets, and is saved with the artifacts that used it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..data.timescale import LEAP_SECONDS

QUALITY_NOCOSMICRAY = 0x400     # HMI: cosmic-ray lists unavailable (set on almost every NRT record)


@dataclass(frozen=True)
class QualityFilter:
    """Criteria a SHARP record must meet. The default reproduces the former database
    (``SHARPC_all_*``), which kept exactly the records with a patch position (``LAT_MIN``/``LON_MIN``
    not null); ``QualityFilter.legacy_clean(series)`` reproduces ``SHARPC_clean_*``.

    Attributes
    ----------
    require_position:
        drop records whose patch has no position (empty patch, ``USFLUX`` = 0).
    quality_mask:
        drop records with any of these ``QUALITY`` bits set (``None``: no check). HMI sets bit 16
        (0x10000) for poor/missing data, among others; e.g. ``0xFFFFFFFF`` keeps QUALITY == 0 only.
    quality_values:
        keep only records whose ``QUALITY`` is exactly one of these values (``None``: no check).
    cmask_gt:
        keep records with ``CMASK`` (pixels of the patch used for the parameters) strictly above.
    meangbt_gt:
        keep records with ``MEANGBT`` strictly above (the ``_N2`` features divide by it).
    max_abs_lon:
        maximum ``|longitude|`` (Stonyhurst, deg) of the whole patch; e.g. 70 removes records
        near the limb, where the SHARP parameters are less reliable.
    exclude_years:
        drop records of these years (year of ``T_OBS`` on the TAI scale of the JSOC strings, as the
        former database compared them).
    """

    require_position: bool = True
    quality_mask: int | None = None
    quality_values: tuple[int, ...] | None = None
    cmask_gt: float | None = None
    meangbt_gt: float | None = None
    max_abs_lon: float | None = None
    exclude_years: tuple[int, ...] = ()

    def __post_init__(self):
        if self.quality_values is not None:
            object.__setattr__(self, "quality_values", tuple(int(v) for v in self.quality_values))
        object.__setattr__(self, "exclude_years", tuple(int(y) for y in self.exclude_years))

    @classmethod
    def legacy_clean(cls, series: str) -> "QualityFilter":
        """Rows of the former ``SHARPC_clean_<series>`` tables (reproduced exactly on the dump of
        2025-06-06): ``CMASK > 5000``, ``MEANGBT > 0`` and QUALITY 0x00000000 (DEF) or exactly
        0x00000400 (NRT, where the cosmic-ray bit is always set); NRT also without 2013."""
        if series == "DEF":
            return cls(quality_values=(0,), cmask_gt=5000, meangbt_gt=0)
        if series == "NRT":
            return cls(quality_values=(QUALITY_NOCOSMICRAY,), cmask_gt=5000, meangbt_gt=0, exclude_years=(2013,))
        raise ValueError(f"series must be 'DEF' or 'NRT', not {series!r}")

    def columns(self) -> list[str]:
        """Columns needed to apply the filter."""
        cols = []
        if self.require_position:
            cols += ["LAT_MIN", "LON_MIN"]
        if self.quality_mask is not None or self.quality_values is not None:
            cols.append("QUALITY")
        if self.cmask_gt is not None:
            cols.append("CMASK")
        if self.meangbt_gt is not None:
            cols.append("MEANGBT")
        if self.max_abs_lon is not None:
            cols += ["LON_MIN", "LON_MAX"]
        if self.exclude_years:
            cols.append("T_OBS")
        return list(dict.fromkeys(cols))

    def mask(self, df: pd.DataFrame) -> pd.Series:
        keep = pd.Series(True, index=df.index)
        if self.require_position:
            keep &= df["LAT_MIN"].notna() & df["LON_MIN"].notna()
        if self.quality_mask is not None:
            keep &= (df["QUALITY"].fillna(0).astype("int64") & int(self.quality_mask)) == 0
        if self.quality_values is not None:
            keep &= df["QUALITY"].isin(self.quality_values).fillna(False).astype(bool)
        if self.cmask_gt is not None:
            keep &= df["CMASK"].fillna(-np.inf) > self.cmask_gt
        if self.meangbt_gt is not None:
            keep &= df["MEANGBT"].fillna(-np.inf) > self.meangbt_gt
        if self.max_abs_lon is not None:
            keep &= (df["LON_MIN"] >= -self.max_abs_lon) & (df["LON_MAX"] <= self.max_abs_lon)
        if self.exclude_years:
            keep &= ~_tai_year(df["T_OBS"]).isin(self.exclude_years)
        return keep

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.mask(df)]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["quality_values"] = None if self.quality_values is None else list(self.quality_values)
        d["exclude_years"] = list(self.exclude_years)
        return d


def _tai_year(t_utc: pd.Series) -> pd.Series:
    """Year of UTC timestamps on the TAI scale (TAI = UTC + leap offset)."""
    t = pd.to_datetime(t_utc, utc=True)
    offset = pd.Series(0, index=t.index, dtype="int64")
    for start_utc, sec in LEAP_SECONDS:
        offset[t >= pd.Timestamp(start_utc, tz="UTC")] = sec
    return (t + pd.to_timedelta(offset, unit="s")).dt.year
