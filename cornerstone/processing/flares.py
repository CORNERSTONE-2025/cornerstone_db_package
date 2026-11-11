"""GOES flare events ready for labeling: calibration, one satellite per flare, association to HARPs.

Steps (all parameters in ``FlareConfig``, saved with the artifacts):

1. **Calibration.** SWPC fluxes of GOES-8…15 carry the operational 0.7 scaling (classes ~30% low);
   GOES-16 and later are reported unscaled. ``calibration="science"`` divides the GOES-8…15
   fluxes by 0.7 (science-quality scale, consistent across satellites); ``"operational"`` keeps
   the reported values. Classes are recomputed from the flux.
2. **One satellite per flare.** SWPC sometimes lists the same flare from two GOES under one event
   number; the line of the day's primary satellite (the one with most reports that day) is kept.
   Several lines of the same satellite under one number are distinct flares and are all kept.
   Non-GOES lines (e.g. SDO) are dropped.
3. **Peak time.** Events whose peak was not observed get ``tstart`` (``peak_fallback="start"``)
   or are dropped (``"drop"``).
4. **Association.** Candidate HARPs of a flare: HARPs that contain its NOAA region (DEF: JSOC map;
   NRT: ``NOAA_ARS`` keyword) and are observed at the peak time. A NOAA number denotes a single
   passage of a region, so when none of them is observed at the peak (flare just before the HARP
   starts or after it ends, e.g. beyond the limb) the HARP with the nearest observation span counts,
   within ``max_gap_hours`` (the former list associated by day). The relation is many-to-many;
   when several HARPs are candidates (``ambiguous``):
   - ``"position_then_flux"`` (default): the HARP whose patch contains the flare position (±
     ``position_margin_deg``); if the position is missing or matches none/several, the candidate
     with the largest ``USFLUX`` at the peak time;
   - ``"position_then_all"``: as above, but without a position match the flare goes to every candidate;
   - ``"flux"``: largest ``USFLUX``; ``"all"``: every candidate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..data import DataStore, goes
from .harps import harp_regions, harp_spans
from .quality import QualityFilter

OPERATIONAL_SCALING = 0.7
CLASS_LETTERS = (("X", 1e-4), ("M", 1e-5), ("C", 1e-6), ("B", 1e-7), ("A", 1e-8))
AMBIGUOUS_RULES = ("position_then_flux", "position_then_all", "flux", "all")


@dataclass(frozen=True)
class FlareConfig:
    calibration: str = "science"            # "science" | "operational"
    peak_fallback: str = "start"            # "start" | "drop"
    ambiguous: str = "position_then_flux"   # see AMBIGUOUS_RULES
    position_margin_deg: float = 2.0
    max_gap_hours: float = 24.0             # flare outside every candidate span: nearest within this

    def __post_init__(self):
        if self.calibration not in ("science", "operational"):
            raise ValueError(f"calibration must be 'science' or 'operational', not {self.calibration!r}")
        if self.peak_fallback not in ("start", "drop"):
            raise ValueError(f"peak_fallback must be 'start' or 'drop', not {self.peak_fallback!r}")
        if self.ambiguous not in AMBIGUOUS_RULES:
            raise ValueError(f"ambiguous must be one of {AMBIGUOUS_RULES}, not {self.ambiguous!r}")

    def to_dict(self) -> dict:
        return asdict(self)


def satellite_number(sat: pd.Series) -> pd.Series:
    """``G15`` -> 15; non-GOES (``SDO``, ``////``) -> NaN."""
    return pd.to_numeric(sat.astype("string").str.extract(r"^G(\d+)$")[0], errors="coerce")


def flux_to_class(flux: float) -> str | None:
    """1.23e-5 -> ``M1.2`` (GOES class from a 1-8 Å peak flux in W/m²)."""
    if flux is None or not np.isfinite(flux) or flux <= 0:
        return None
    for letter, lo in CLASS_LETTERS:
        if flux >= lo * (1 - 1e-9):
            return f"{letter}{flux / lo:.1f}"
    return f"A{flux / 1e-8:.1f}"


def calibrate(ev: pd.DataFrame, calibration: str) -> pd.Series:
    """Peak flux on the chosen scale (see module docstring)."""
    flux = ev["flux_reported"].astype(float)
    if calibration == "operational":
        return flux
    scaled = satellite_number(ev["satellite"]) <= 15
    return flux.where(~scaled, flux / OPERATIONAL_SCALING)


def one_satellite_per_flare(ev: pd.DataFrame) -> pd.DataFrame:
    """Drop the lines of non-primary satellites inside an event number reported by several GOES."""
    ev = ev[satellite_number(ev["satellite"]).notna()].copy()
    day = ev["event"].str.slice(0, 8)
    counts = ev.groupby([day, ev["satellite"]]).size()
    ev["_rank"] = [counts[(d, s)] for d, s in zip(day, ev["satellite"])]
    n_sat = ev.groupby("event")["satellite"].transform("nunique")
    best = ev.groupby("event")["_rank"].transform("max")
    keep = (n_sat == 1) | (ev["_rank"] == best)
    return ev[keep].drop(columns="_rank")


def prepare_events(store: DataStore, config: FlareConfig = FlareConfig()) -> pd.DataFrame:
    """GOES events of the store on the chosen scale, one satellite per flare, with ``t_peak``."""
    ev = one_satellite_per_flare(goes.load(store))
    ev["flux"] = calibrate(ev, config.calibration)
    ev["goes_class"] = [flux_to_class(f) for f in ev["flux"]]
    ev["t_peak"] = ev["tmax"]
    if config.peak_fallback == "start":
        ev["t_peak"] = ev["t_peak"].fillna(ev["tstart"])
    ev = ev.dropna(subset=["t_peak", "flux"])
    cols = ["event_id", "event", "t_peak", "tstart", "tmax", "tend", "satellite", "goes_class", "flux",
            "flux_reported", "NOAA_AR", "location", "lat", "lon"]
    return ev[cols].sort_values(["t_peak", "event_id"]).reset_index(drop=True)


def _nearest_records(store: DataStore, series: str, quality: QualityFilter, cand: pd.DataFrame) -> pd.DataFrame:
    """For every (event, HARP) candidate: the SHARP record nearest to the peak (±1 h)."""
    cols = list(dict.fromkeys(["HARPNUM", "T_OBS", "LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX", "USFLUX",
                               *quality.columns()]))
    rec = quality.apply(store.read_sharp(series, columns=cols, harpnums=cand["HARPNUM"].unique()))
    ns = "datetime64[ns, UTC]"                      # merge_asof needs the same time resolution
    rec = rec.astype({"T_OBS": ns}).sort_values("T_OBS")
    left = cand[["event_id", "HARPNUM", "t_peak"]].astype({"t_peak": ns}).sort_values("t_peak")
    near = pd.merge_asof(left, rec, left_on="t_peak", right_on="T_OBS", by="HARPNUM",
                         direction="nearest", tolerance=pd.Timedelta(hours=1))
    return near


def associate(store: DataStore, series: str, quality: QualityFilter = QualityFilter(),
              config: FlareConfig = FlareConfig(), events: pd.DataFrame | None = None
              ) -> tuple[pd.DataFrame, dict]:
    """Flare events of ``series`` (DEF/NRT) with their HARP: one row per (event, HARP).

    Returns the table and counts of how the events were associated (for the artifact metadata).
    """
    ev = prepare_events(store, config) if events is None else events
    spans = harp_spans(store, series, quality)
    pairs = harp_regions(store, series, quality)
    cand = (ev.dropna(subset=["NOAA_AR"]).astype({"NOAA_AR": "int32"})
            .merge(pairs, on="NOAA_AR").merge(spans[["HARPNUM", "start", "end"]], on="HARPNUM"))
    # hours between the peak and the observation span of the HARP (0 when observed at the peak)
    gap = np.maximum((cand["start"] - cand["t_peak"]).dt.total_seconds(),
                     (cand["t_peak"] - cand["end"]).dt.total_seconds()).clip(lower=0) / 3600
    cand["gap_hours"] = gap.astype("float32")
    observed = cand["gap_hours"] == 0
    n_obs = observed.groupby(cand["event_id"]).transform("sum")
    # a NOAA number denotes one passage of a region: when no candidate HARP is observed at the peak
    # (flare just before the HARP starts or after it ends, e.g. at the limb) the nearest one counts
    nearest = (n_obs == 0) & (cand["gap_hours"] <= config.max_gap_hours) \
        & (cand["gap_hours"] == cand.groupby("event_id")["gap_hours"].transform("min"))
    cand = cand[observed | nearest].drop(columns=["start", "end"])
    cand["n_candidates"] = cand.groupby("event_id")["HARPNUM"].transform("size").astype("int16")

    single = cand[cand["n_candidates"] == 1]
    single = single.assign(rule=np.where(single["gap_hours"] == 0, "unique", "nearest_span"))
    multi = cand[cand["n_candidates"] > 1]
    chosen = [single]
    if len(multi):
        near = _nearest_records(store, series, quality, multi)
        m = multi.merge(near[["event_id", "HARPNUM", "LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX", "USFLUX"]],
                        on=["event_id", "HARPNUM"], how="left")
        d = config.position_margin_deg
        m["inside"] = (m["lat"].between(m["LAT_MIN"] - d, m["LAT_MAX"] + d)
                       & m["lon"].between(m["LON_MIN"] - d, m["LON_MAX"] + d))
        m["USFLUX"] = m["USFLUX"].fillna(-np.inf)
        by_position = config.ambiguous.startswith("position")
        to_all = config.ambiguous in ("all", "position_then_all")
        for _, g in m.groupby("event_id", sort=False):
            inside = g[g["inside"]] if by_position else g.iloc[:0]
            if len(inside) == 1:
                chosen.append(inside.assign(rule="position"))
            elif len(inside) > 1:               # several patches contain the position
                chosen.append(inside.assign(rule="position") if to_all
                              else inside.loc[[inside["USFLUX"].idxmax()]].assign(rule="position+flux"))
            elif to_all:
                chosen.append(g.assign(rule="all"))
            else:
                chosen.append(g.loc[[g["USFLUX"].idxmax()]].assign(rule="flux"))
    out = pd.concat(chosen, ignore_index=True)
    out = out[["event_id", "HARPNUM", *[c for c in ev.columns if c != "event_id"], "n_candidates", "gap_hours",
               "rule"]]
    out = out.sort_values(["t_peak", "event_id", "HARPNUM"]).reset_index(drop=True)
    with_region = int(ev["NOAA_AR"].notna().sum())
    stats = {"events": len(ev), "with_noaa_region": with_region,
             "associated": int(out["event_id"].nunique()),
             "not_associated": with_region - int(out["event_id"].nunique()),
             "ambiguous": int(multi["event_id"].nunique()),
             "rules": {k: int(v) for k, v in out.drop_duplicates("event_id")["rule"].value_counts().items()}}
    return out, stats
