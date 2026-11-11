"""GOES X-ray flare events from the NOAA SWPC edited event reports.

Source: the daily "Edited Events" reports of the NOAA Space Weather Prediction Center
(``ftp://ftp.swpc.noaa.gov/pub/warehouse/<year>/``: one ``<year>_events.tar.gz`` per past
year, daily files in ``<year>_events/`` for the current one). It is the source of the flare list
of the former database: ``XRA`` lines (GOES 1-8 Å events: satellite, begin/max/end, class,
integrated flux, NOAA region) and ``FLA`` lines (optical flares; the line with the same event
number gives the heliographic position of the flare).

Fluxes are stored **as reported** (``flux_reported``) together with the satellite; the calibration
choice (science-quality vs operational) is applied when the events are processed, not here.
"""
from __future__ import annotations

import ftplib
import io
import re
import tarfile
import time
import urllib.request
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa

import pyarrow.parquet as pq

from .store import DataStore, now_iso, sha256, say

WAREHOUSE = "ftp://ftp.swpc.noaa.gov/pub/warehouse"
FIRST_YEAR = 2010                     # SDO/HMI starts in May 2010
CLASS_FLUX = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}

SCHEMA = pa.schema([
    pa.field("event_id", pa.string(), nullable=False),     # <yyyymmdd>-<event number>-<k>, unique
    pa.field("event", pa.string(), nullable=False),        # <yyyymmdd>-<event number> of the report: SWPC
                                                           # groups several XRA lines under one number
                                                           # (distinct flares, or one flare seen by two GOES)
    pa.field("tstart", pa.timestamp("s", tz="UTC")),
    pa.field("tmax", pa.timestamp("s", tz="UTC")),
    pa.field("tend", pa.timestamp("s", tz="UTC")),
    pa.field("satellite", pa.string()),                    # G13, G15, G16, ...
    pa.field("quality", pa.string()),
    pa.field("goes_class", pa.string()),                   # as reported, e.g. M1.0
    pa.field("flux_reported", pa.float64()),               # W/m^2 from the reported class
    pa.field("integrated_flux", pa.float64()),             # J/m^2 (1-8 A, as reported)
    pa.field("region_reported", pa.int32()),               # 4-digit NOAA number in the report
    pa.field("NOAA_AR", pa.int32()),                       # full NOAA number
    pa.field("location", pa.string()),                     # e.g. S14E18, from the FLA line
    pa.field("lat", pa.float32()),                         # heliographic, degrees (N > 0)
    pa.field("lon", pa.float32()),                         # Stonyhurst, degrees (W > 0)
])

_TIME = re.compile(r"(\d{4})")
_CLASS = re.compile(r"^([ABCMX])(\d+(?:\.\d+)?)$")
_LOC = re.compile(r"^([NS])(\d{2})([EW])(\d{2})$")


def class_to_flux(goes_class: str | None) -> float:
    """``M1.0`` -> 1e-5 W/m^2 (NaN when not a GOES class)."""
    m = _CLASS.match(goes_class or "")
    return CLASS_FLUX[m.group(1)] * float(m.group(2)) if m else np.nan


def full_noaa_number(region: int | None, day: pd.Timestamp) -> int | None:
    """SWPC reports the NOAA number modulo 10000 (``1944`` is AR 11944): regions passed 10000 on
    2002-06-14, so later reports are offset by 10000 (valid until region 20000)."""
    if region is None:
        return None
    return region + 10000 if day >= pd.Timestamp("2002-06-14") else region


def _hhmm(token: str, day: pd.Timestamp) -> pd.Timestamp | None:
    m = _TIME.search(token)
    if not m:
        return None                         # "////": not observed
    h, mi = int(m.group(1)[:2]), int(m.group(1)[2:])
    return day + pd.Timedelta(hours=h, minutes=mi)


def parse_events(text: str, day: pd.Timestamp | None = None) -> pd.DataFrame:
    """XRA events of one daily report, with the position of the matching FLA line."""
    if day is None:
        m = re.search(r":Date:\s+(\d{4})\s+(\d{2})\s+(\d{2})", text)
        if not m:
            raise ValueError("Report without ':Date:' line")
        day = pd.Timestamp(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
    xra, fla = [], {}
    for line in text.splitlines():
        if not line.strip() or line[0] in "#:":
            continue
        t = line.split()
        if len(t) > 1 and t[1] == "+":
            t.pop(1)
        if len(t) < 8 or not t[0].isdigit():
            continue
        ev, typ = t[0], t[6]
        if typ == "FLA":
            if ev not in fla and _LOC.match(t[7]):
                fla[ev] = t[7]
        elif typ == "XRA":
            rest = t[8:]
            reg = int(rest[-1]) if rest and re.fullmatch(r"\d{4}", rest[-1]) and len(rest) >= 2 else None
            parts = rest[:-1] if reg is not None else rest
            xra.append({"ev": ev, "begin": t[1], "max": t[2], "end": t[3], "satellite": t[4], "quality": t[5],
                        "goes_class": parts[0] if parts and _CLASS.match(parts[0]) else None,
                        "integrated_flux": pd.to_numeric(parts[1], errors="coerce") if len(parts) > 1 else np.nan,
                        "region_reported": reg})
    rows = []
    seen: dict[str, int] = {}
    for e in xra:
        k = seen[e["ev"]] = seen.get(e["ev"], 0) + 1
        t0, tm, t1 = (_hhmm(e[k], day) for k in ("begin", "max", "end"))
        # times after midnight of the next day are written HHMM < begin
        if t0 is not None and tm is not None and tm < t0:
            tm += pd.Timedelta(days=1)
        ref = tm or t0
        if ref is not None and t1 is not None and t1 < ref:
            t1 += pd.Timedelta(days=1)
        loc = fla.get(e["ev"])
        lat = lon = np.nan
        if loc:
            m = _LOC.match(loc)
            lat = float(m.group(2)) * (1 if m.group(1) == "N" else -1)
            lon = float(m.group(4)) * (1 if m.group(3) == "W" else -1)
        rows.append({"event_id": f"{day:%Y%m%d}-{e['ev']}-{k}", "event": f"{day:%Y%m%d}-{e['ev']}", "tstart": t0, "tmax": tm, "tend": t1,
                     "satellite": e["satellite"], "quality": e["quality"], "goes_class": e["goes_class"],
                     "flux_reported": class_to_flux(e["goes_class"]), "integrated_flux": e["integrated_flux"],
                     "region_reported": e["region_reported"],
                     "NOAA_AR": full_noaa_number(e["region_reported"], day),
                     "location": loc, "lat": lat, "lon": lon})
    return _typed(pd.DataFrame(rows, columns=SCHEMA.names))


def _typed(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("tstart", "tmax", "tend"):
        t = pd.to_datetime(df[c])
        df[c] = t.dt.tz_localize("UTC") if t.dt.tz is None else t.dt.tz_convert("UTC")
    for c in ("region_reported", "NOAA_AR"):
        df[c] = df[c].astype("Int32")
    for c in ("lat", "lon"):
        df[c] = df[c].astype("float32")
    return df


# ---------------------------------------------------------------------------------------- download
def _get(url: str, timeout: float = 300, retries: int = 4) -> bytes:
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except OSError:
            if attempt == retries:
                raise
            time.sleep(10 * 3 ** attempt)
    raise AssertionError("unreachable")


def _year_reports(year: int) -> Iterable[str]:
    """Texts of the daily reports of ``year`` (archive for past years, daily files otherwise)."""
    base = f"{WAREHOUSE}/{year}"
    listing = _get(f"{base}/").decode(errors="replace")
    if f"{year}_events.tar.gz" in listing:
        with tarfile.open(fileobj=io.BytesIO(_get(f"{base}/{year}_events.tar.gz")), mode="r:gz") as tar:
            for m in tar.getmembers():
                if m.isfile() and m.name.endswith("events.txt"):
                    yield tar.extractfile(m).read().decode(errors="replace")
    elif f"{year}_events" in listing:
        # current year: daily files, fetched over a single FTP connection
        host, path = WAREHOUSE.removeprefix("ftp://").split("/", 1)
        with ftplib.FTP(host, timeout=300) as ftp:
            ftp.login()
            ftp.cwd(f"/{path}/{year}/{year}_events")
            for n in sorted(x for x in ftp.nlst() if re.fullmatch(r"\d{8}events\.txt", x)):
                buf = io.BytesIO()
                ftp.retrbinary(f"RETR {n}", buf.write)
                yield buf.getvalue().decode(errors="replace")
    else:
        raise FileNotFoundError(f"No event reports for {year} in {base}")


def year_path(store: DataStore, year: int):
    return store.root / "raw" / "goes" / f"swpc_events_{year}.parquet"


def download(store: DataStore, years: Iterable[int] | None = None, full: bool = False,
             log: Callable[[str], None] = say) -> dict:
    """Download the SWPC events into one Parquet file per year.

    Without ``years``: every year from ``FIRST_YEAR``; years already stored are skipped except the
    last stored one and the current year (still growing), unless ``full``.
    """
    this_year = pd.Timestamp.now(tz="UTC").year
    years = list(years) if years is not None else list(range(FIRST_YEAR, this_year + 1))
    stored = sorted(int(p.stem.rsplit("_", 1)[1]) for p in (store.root / "raw" / "goes").glob("swpc_events_*.parquet"))
    keep = set() if full else set(stored[:-1]) - {this_year}
    registered = store.load_manifest()["datasets"].get("goes/swpc_events", {}).get("files", {})
    name = "goes/swpc_events"
    info_fields = dict(source={"kind": "noaa-swpc-edited-events", "url": WAREHOUSE},
                       flux="as reported by SWPC (calibration applied in processing)")
    entry = {}
    for y in years:
        p = year_path(store, y)
        if y in keep:
            if p.name not in registered:          # written by an interrupted run: register it
                entry = store.record_dataset(name, {p.name: {"rows": pq.ParquetFile(p).metadata.num_rows,
                                                             "bytes": p.stat().st_size, "sha256": sha256(p)}},
                                             merge=True, **info_fields)
            continue
        frames = [parse_events(t) for t in _year_reports(y)]
        # a report can appear in two yearly archives, or twice in one: keep each day once, in its year
        frames = [f for f in frames if len(f) and f["event"].iloc[0].startswith(str(y))]
        frames = list({f["event"].iloc[0][:8]: f for f in frames}.values())
        df = pd.concat(frames, ignore_index=True) if frames else _typed(pd.DataFrame(columns=SCHEMA.names))
        df = df.sort_values(["tmax", "event_id"], kind="stable").reset_index(drop=True)
        info = store.write_table(p, pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False))
        entry = store.record_dataset(name, {p.name: {**info, "downloaded_at": now_iso()}}, merge=True, **info_fields)
        log(f"GOES {y}: {len(df):,} X-ray events ({int(df['NOAA_AR'].notna().sum()):,} with NOAA region, "
            f"{int(df['location'].notna().sum()):,} with position)")
    return entry or store.load_manifest()["datasets"].get(name, {})


def load(store: DataStore) -> pd.DataFrame:
    files = sorted((store.root / "raw" / "goes").glob("swpc_events_*.parquet"))
    if not files:
        raise FileNotFoundError("No GOES events in the store: run the download first")
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
