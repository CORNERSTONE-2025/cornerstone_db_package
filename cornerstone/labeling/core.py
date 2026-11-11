"""Parametric flare labeling of (t_X, HARP) pairs.

For every issue time ``t_X`` of a regular grid and every HARP observed at ``t_X``
(``start <= t_X <= end``), with the **flare window** ``[t_X, t_X + W]``:

**Prediction label** (threshold ``S_p``, ``LabelConfig.threshold``):

- ``YES`` if at least one flare with flux >= ``S_p`` peaks in the flare window;
- otherwise ``NO1``–``NO4`` from the flares >= ``S_p`` *before* and *after*:

  ======  ========  =======
  label   before    after
  ======  ========  =======
  NO1     no        no
  NO2     no        yes
  NO3     yes       yes
  NO4     yes       no
  ======  ========  =======

  "before" is ``[t_X - L, t_X]`` and "after" ``[t_X + W, t_X + W + L]``; without ``L``
  (``strat_window_hours=None``, default) "before" is the whole past of the HARP and "after" its
  whole future.

**Stratification class** (threshold ``S_s``, ``strat_threshold``, default ``S_p``; same flare
window and before/after windows): the class letter (C, M, X, ...) of the strongest flare of the flare
window when it is >= ``S_s``, otherwise ``NO1``–``NO4`` computed with ``S_s``. It is used to
stratify the dataset splits; the target of the models is ``YES`` vs ``NO*``.

Window bounds are inclusive, as in the former implementation (``BETWEEN``). The former static
labels are reproduced with ``W = L = 24 h``, ``step = 24 h``, grid from 1996-01-01, ``S_s = S_p``
(see ``cornerstone.labeling.validation``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

CLASS_FLUX = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}
LABELS = ("YES", "NO1", "NO2", "NO3", "NO4")
_NO = np.array(["NO1", "NO2", "NO4", "NO3"])          # index = 2*before + after
_REL = 1e-9                                           # float tolerance on the thresholds


def threshold_flux(threshold: str | float) -> float:
    """``"M"`` -> 1e-5, ``"M5"``/``"M5.0"`` -> 5e-5, a number is a flux in W/m²."""
    if isinstance(threshold, (int, float, np.floating)):
        return float(threshold)
    s = str(threshold).strip().upper()
    if s and s[0] in CLASS_FLUX:
        return CLASS_FLUX[s[0]] * (float(s[1:]) if len(s) > 1 else 1.0)
    return float(s)


def class_letter(flux: np.ndarray) -> np.ndarray:
    out = np.full(len(flux), "", dtype=object)
    for letter, lo in sorted(CLASS_FLUX.items(), key=lambda kv: kv[1]):
        out[flux >= lo * (1 - _REL)] = letter
    return out


@dataclass(frozen=True)
class LabelConfig:
    """Parameters of the labeling (saved with the label artifacts)."""

    threshold: str | float = "M"                 # S_p: prediction threshold (class or W/m²)
    flare_window_hours: float = 24.0             # W
    step_hours: float = 24.0                     # spacing of the t_X grid
    strat_threshold: str | float | None = None   # S_s: stratification threshold (None: = S_p)
    strat_window_hours: float | None = None      # L: before/after windows (None: unlimited)
    start: str | None = None                     # first t_X (UTC); None: first observation
    end: str | None = None                       # last t_X (UTC); None: last observation
    grid_origin: str = "1996-01-01T00:00:00Z"    # the grid is aligned to this instant

    def __post_init__(self):
        if self.flare_window_hours <= 0 or self.step_hours <= 0:
            raise ValueError("flare_window_hours and step_hours must be positive")
        if self.strat_window_hours is not None and self.strat_window_hours <= 0:
            raise ValueError("strat_window_hours must be positive (or None for unlimited)")
        threshold_flux(self.threshold)
        if self.strat_threshold is not None:
            threshold_flux(self.strat_threshold)

    @property
    def s_p(self) -> float:
        return threshold_flux(self.threshold)

    @property
    def s_s(self) -> float:
        return threshold_flux(self.threshold if self.strat_threshold is None else self.strat_threshold)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(threshold_flux_Wm2=self.s_p, strat_threshold_flux_Wm2=self.s_s)
        return d


def _utc(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def issue_grid(start: pd.Timestamp, end: pd.Timestamp, config: LabelConfig) -> pd.DatetimeIndex:
    """t_X from ``start`` to ``end`` (inclusive), every ``step_hours``, aligned to ``grid_origin``."""
    origin, step = _utc(config.grid_origin), pd.Timedelta(hours=config.step_hours)
    first = origin + np.ceil((start - origin) / step) * step
    if first > end:
        return pd.DatetimeIndex([], tz="UTC")
    return pd.date_range(first, end, freq=step)


def label(spans: pd.DataFrame, flares: pd.DataFrame, config: LabelConfig = LabelConfig()) -> pd.DataFrame:
    """Label every (t_X, HARP) pair.

    Parameters
    ----------
    spans:
        one row per HARP: ``HARPNUM``, ``start``, ``end`` (UTC timestamps; observed at t when
        ``start <= t <= end``), e.g. ``cornerstone.processing.harp_spans``.
    flares:
        one row per (flare, HARP): ``HARPNUM``, ``t_peak`` (UTC), ``flux`` (W/m²), e.g.
        ``cornerstone.processing.associate``.

    Returns
    -------
    DataFrame sorted by (t_X, HARPNUM) with ``t_X``, ``HARPNUM``, ``label`` (YES/NO1..NO4),
    ``target`` (1 for YES), ``strat_class``, ``max_flux`` and ``max_class`` (strongest flare of the
    flare window, any class), ``n_window``/``n_before``/``n_after`` (flares >= S_p), ``obs_start``,
    ``obs_end`` (observation span of the HARP).
    """
    W = pd.Timedelta(hours=config.flare_window_hours)
    L = None if config.strat_window_hours is None else pd.Timedelta(hours=config.strat_window_hours)
    s_p, s_s = config.s_p * (1 - _REL), config.s_s * (1 - _REL)
    spans = spans.dropna(subset=["start", "end"])
    lo = _utc(config.start) if config.start else spans["start"].min()
    hi = _utc(config.end) if config.end else spans["end"].max()
    grid = issue_grid(lo, hi, config)
    g_ns = grid.asi8
    by_harp = {}
    if len(flares):
        fl = flares.dropna(subset=["t_peak", "flux"])
        t_ns = pd.to_datetime(fl["t_peak"], utc=True).astype("datetime64[ns, UTC]")
        fl = fl.assign(_t=t_ns.to_numpy("datetime64[ns]").astype("int64"))
        by_harp = {h: (g["_t"].to_numpy(), g["flux"].to_numpy(float)) for h, g in fl.groupby("HARPNUM")}
    w_ns = W.value
    l_ns = None if L is None else L.value
    parts = []
    for h, s, e in zip(spans["HARPNUM"], spans["start"], spans["end"]):
        i0 = np.searchsorted(g_ns, _utc(s).value, "left")
        i1 = np.searchsorted(g_ns, _utc(e).value, "right")
        if i1 <= i0:
            continue
        tx = g_ns[i0:i1]
        n = len(tx)
        if h in by_harp:
            tf, ff = by_harp[h]
            T, X = tf[None, :], tx[:, None]
            in_win = (T >= X) & (T <= X + w_ns)
            before = T <= X
            after = T >= X + w_ns
            if l_ns is not None:
                before &= T >= X - l_ns
                after &= T <= X + w_ns + l_ns
            fw = np.where(in_win, ff[None, :], 0.0)
            max_flux = fw.max(axis=1)
            big_p, big_s = ff >= s_p, ff >= s_s
            n_window = (in_win & big_p).sum(1)
            n_before = (before & big_p).sum(1)
            n_after = (after & big_p).sum(1)
            b_s = (before & big_s).any(1)
            a_s = (after & big_s).any(1)
        else:
            max_flux = np.zeros(n)
            n_window = n_before = n_after = np.zeros(n, dtype=int)
            b_s = a_s = np.zeros(n, dtype=bool)
        yes = n_window > 0
        lab = np.where(yes, "YES", _NO[2 * (n_before > 0) + (n_after > 0)])
        strat = np.where(max_flux >= s_s, class_letter(max_flux), _NO[2 * b_s + a_s])
        parts.append(pd.DataFrame({
            "t_X": tx, "HARPNUM": h, "label": lab, "target": yes.astype("int8"), "strat_class": strat,
            "max_flux": max_flux, "n_window": n_window, "n_before": n_before, "n_after": n_after,
            "obs_start": s, "obs_end": e}))
    cols = ["t_X", "HARPNUM", "label", "target", "strat_class", "max_flux", "max_class", "n_window", "n_before",
            "n_after", "obs_start", "obs_end"]
    if not parts:
        return pd.DataFrame(columns=cols)
    out = pd.concat(parts, ignore_index=True)
    out["t_X"] = pd.to_datetime(out["t_X"], utc=True)
    out["max_flux"] = out["max_flux"].where(out["max_flux"] > 0)
    out["max_class"] = pd.Series(class_letter(out["max_flux"].fillna(0).to_numpy()), index=out.index).replace("", None)
    out = out.astype({"HARPNUM": "int32", "n_window": "int16", "n_before": "int32", "n_after": "int32"})
    return out[cols].sort_values(["t_X", "HARPNUM"], kind="stable").reset_index(drop=True)
