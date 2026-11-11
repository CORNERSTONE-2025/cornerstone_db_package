"""Selection of the SHARP records used downstream ("good data"), as an explicit parameter set.

The store keeps every record; a ``QualityFilter`` decides which ones enter the HARP observation
spans, the labels and the datasets, and is saved with the artifacts that used it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class QualityFilter:
    """Criteria a SHARP record must meet. The default reproduces the former database, which kept
    exactly the records with a patch position (``LAT_MIN``/``LON_MIN`` not null).

    Attributes
    ----------
    require_position:
        drop records whose patch has no position (empty patch, ``USFLUX`` = 0).
    quality_mask:
        drop records with any of these ``QUALITY`` bits set (``None``: no check). HMI sets bit 16
        (0x10000) for poor/missing data, among others; e.g. ``0xFFFFFFFF`` keeps QUALITY == 0 only.
    min_cmask:
        minimum ``CMASK`` (number of pixels in the patch used for the parameters).
    max_abs_lon:
        maximum ``|longitude|`` (Stonyhurst, deg) of the whole patch; e.g. 70 removes records
        near the limb, where the SHARP parameters are less reliable.
    """

    require_position: bool = True
    quality_mask: int | None = None
    min_cmask: float | None = None
    max_abs_lon: float | None = None

    def columns(self) -> list[str]:
        """Columns needed to apply the filter."""
        cols = []
        if self.require_position:
            cols += ["LAT_MIN", "LON_MIN"]
        if self.quality_mask is not None:
            cols.append("QUALITY")
        if self.min_cmask is not None:
            cols.append("CMASK")
        if self.max_abs_lon is not None:
            cols += ["LON_MIN", "LON_MAX"]
        return list(dict.fromkeys(cols))

    def mask(self, df: pd.DataFrame) -> pd.Series:
        keep = pd.Series(True, index=df.index)
        if self.require_position:
            keep &= df["LAT_MIN"].notna() & df["LON_MIN"].notna()
        if self.quality_mask is not None:
            keep &= (df["QUALITY"].fillna(0).astype("int64") & int(self.quality_mask)) == 0
        if self.min_cmask is not None:
            keep &= df["CMASK"].fillna(-1) >= self.min_cmask
        if self.max_abs_lon is not None:
            keep &= (df["LON_MIN"] >= -self.max_abs_lon) & (df["LON_MAX"] <= self.max_abs_lon)
        return keep

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.mask(df)]

    def to_dict(self) -> dict:
        return asdict(self)
