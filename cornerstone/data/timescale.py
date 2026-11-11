"""Time scales: HMI/SHARP ``T_OBS`` is in TAI, GOES flare times are in UTC.

TAI - UTC changes only at leap seconds; from the start of SDO (2010) it went from 34 s to 37 s.
Converting ``T_OBS`` to UTC once, when the data are stored, puts SHARP observations and flare
events on the same time scale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# (UTC instant from which the offset applies, TAI - UTC in seconds). Source: IERS Bulletin C.
# No leap second has been announced after 2017-01-01.
LEAP_SECONDS: tuple[tuple[str, int], ...] = (
    ("1999-01-01", 32),
    ("2006-01-01", 33),
    ("2009-01-01", 34),
    ("2012-07-01", 35),
    ("2015-07-01", 36),
    ("2017-01-01", 37),
)

TAI_FORMAT = "%Y.%m.%d_%H:%M:%S"     # JSOC keyword format, followed by "_TAI"


def tai_minus_utc(t_tai: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """TAI - UTC (seconds) for naive TAI datetimes."""
    t = pd.DatetimeIndex(t_tai)
    out = np.zeros(len(t), dtype="int64")
    for start_utc, offset in LEAP_SECONDS:
        # the change happens at start_utc (UTC), i.e. at start_utc + offset in TAI
        start_tai = pd.Timestamp(start_utc) + pd.Timedelta(seconds=offset)
        out[np.asarray(t >= start_tai)] = offset
    return out


def tai_to_utc(t_tai: pd.Series) -> pd.Series:
    """Naive TAI datetimes -> timezone-aware UTC datetimes (NaT preserved)."""
    t = pd.to_datetime(t_tai)
    utc = t - pd.to_timedelta(tai_minus_utc(t), unit="s")
    return utc.dt.tz_localize("UTC")


def parse_t_obs(t_obs: pd.Series) -> pd.Series:
    """Parse JSOC ``T_OBS`` strings (``2010.05.01_00:00:04_TAI``) to UTC timestamps.

    Strings that do not parse (``MISSING`` in JSOC answers) become NaT.
    """
    s = t_obs.astype("string").str.slice(0, 19)
    t = pd.to_datetime(s, format=TAI_FORMAT, errors="coerce")
    return tai_to_utc(t)
