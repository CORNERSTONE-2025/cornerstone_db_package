"""Data sources (offline: the network calls are replaced): HARP-NOAA map, SWPC events, JSOC SHARP."""
import numpy as np
import pandas as pd
import pytest

from cornerstone.data import DataStore, goes, harp_noaa, jsoc, sharp
from cornerstone.data.store import block_name


# ------------------------------------------------------------------ HARP <-> NOAA
def test_harp_noaa_many_to_many():
    pairs = harp_noaa.parse("HARPNUM NOAA_ARS\n38 11073\n45 11073,11077\n700 11253\n705 11253\n45 11077\n")
    assert pairs.values.tolist() == [[38, 11073], [45, 11073], [45, 11077], [700, 11253], [705, 11253]]
    assert harp_noaa.harp_to_ars(pairs)[45] == [11073, 11077]          # one HARP, several regions
    assert harp_noaa.ar_to_harps(pairs)[11073] == [38, 45]             # one region, several HARPs
    with pytest.raises(ValueError):
        harp_noaa.parse("HARPNUM NOAA_ARS\n38\n")


# ------------------------------------------------------------------ GOES (SWPC edited events)
REPORT = """:Product: 20140107events.txt
:Date: 2014 01 07
#Event    Begin    Max       End  Obs  Q  Type  Loc/Frq   Particulars       Reg#
#-------------------------------------------------------------------------------
8560       0229   0234      0239  G15  5   XRA  1-8A      C2.6    1.1E-03   1944
8560       0232   0232      0244  LEA  3   FLA  S14E18    SF      ERU       1944
8760 +     1804   1832      1858  G15  5   XRA  1-8A      X1.2    2.5E-01   1944
8760 +     1810   1820      1900  LEA  3   FLA  S12W08    2B                1944
9000       2349   ////      0004  G15  5   XRA  1-8A      C3.8    1.8E-03
9010       2350   0010      0030  G15  5   XRA  1-8A      M1.0    2.0E-03   1946
8590       0000   0040      0323  LEA  G   RNS  245       150
"""


def test_swpc_report_parsing():
    df = goes.parse_events(REPORT)
    assert df["event_id"].tolist() == ["20140107-8560-1", "20140107-8760-1", "20140107-9000-1", "20140107-9010-1"]
    a = df.iloc[0]
    assert a.tmax == pd.Timestamp("2014-01-07 02:34", tz="UTC") and a.goes_class == "C2.6"
    assert a.flux_reported == pytest.approx(2.6e-6) and a.NOAA_AR == 11944 and a.region_reported == 1944
    assert a.location == "S14E18" and a.lat == -14 and a.lon == -18          # east is negative
    b = df.iloc[1]                                                           # "+" (grouped event)
    assert b.goes_class == "X1.2" and b.lon == 8 and b.flux_reported == pytest.approx(1.2e-4)
    c = df.iloc[2]                                                           # max not observed, no region
    assert pd.isna(c.tmax) and pd.isna(c.NOAA_AR) and c.tend == pd.Timestamp("2014-01-08 00:04", tz="UTC")
    d = df.iloc[3]                                                           # crosses midnight
    assert d.tmax == pd.Timestamp("2014-01-08 00:10", tz="UTC") and pd.isna(d.location)
    assert goes.full_noaa_number(9393, pd.Timestamp("2001-03-29")) == 9393
    two = goes.parse_events(REPORT + "8560       0300   0305      0310  G14  5   XRA  1-8A      C2.5    1.0E-03   1944\n")
    assert two["event_id"].is_unique and (two["event"] == "20140107-8560").sum() == 2


def test_goes_download_by_year(tmp_path, monkeypatch):
    def reports(y):
        r = REPORT.replace("2014 01 07", f"{y} 01 07")
        return [r, r, REPORT.replace("2014 01 07", f"{y - 1} 12 31")]   # duplicate + other year
    monkeypatch.setattr(goes, "_year_reports", reports)
    store = DataStore(tmp_path)
    goes.download(store, years=[2013, 2014], log=lambda m: None)
    ev = goes.load(store)
    assert len(ev) == 8 and ev["tmax"].dt.year.dropna().unique().tolist() == [2013, 2014]
    assert store.load_manifest()["datasets"]["goes/swpc_events"]["rows"] == 8


# ------------------------------------------------------------------ JSOC SHARP
def _line(h, t, **kw):
    vals = {k: "1.000000e+00" for k in sharp.KEYWORDS}
    vals.update(T_OBS=t, HARPNUM=str(h), QUALITY="0x00000000", NOAA_AR="0", NOAA_NUM="0", NOAA_ARS="MISSING", **kw)
    return "\t".join(vals[k] for k in sharp.KEYWORDS)


def test_show_info_parsing():
    text = "series hmi.Marmask_720s not found\n" + _line(5, "2014.01.01_00:00:00_TAI") + "\n"
    df = jsoc._parse_show_info(text)
    assert len(df) == 1 and df.HARPNUM.tolist() == ["5"]
    assert len(jsoc._parse_show_info("series hmi.Marmask_720s not found\n")) == 0      # empty range
    with pytest.raises(jsoc.JsocError, match="Malformed"):
        jsoc._parse_show_info(_line(5, "x")[:40] + "\n")
    with pytest.raises(jsoc.JsocError, match="Truncated"):
        jsoc._parse_show_info(_line(5, "2014.01.01_00:00:00_TAI")[:-3])


def test_blocks_to_update(tmp_path):
    store = DataStore(tmp_path)
    for b in (0, 1000, 2000):
        (store.sharp_dir("DEF")).mkdir(parents=True, exist_ok=True)
        (store.sharp_dir("DEF") / block_name(b)).write_bytes(b"x")
    store.record_dataset("sharp/DEF", {
        block_name(0): {"rows": 5, "bytes": 1, "max_t_obs": "2011-01-01T00:00:00+00:00",
                        "data_until": "2025-04-23T00:00:00+00:00"},                    # quiet: complete
        block_name(1000): {"rows": 5, "bytes": 1, "max_t_obs": "2025-04-20T00:00:00+00:00",
                           "data_until": "2025-04-23T00:00:00+00:00"},                 # active at the dump
        block_name(2000): {"rows": 5, "bytes": 1, "max_t_obs": "2025-01-01T00:00:00+00:00"}})  # no data_until
    assert jsoc.blocks_to_update(store, "DEF", last_harp=4500) == [1000, 2000, 3000, 4000]
    assert jsoc.blocks_to_update(store, "DEF", last_harp=4500, full=True) == [0, 1000, 2000, 3000, 4000]


def test_download_sharp_resumes_from_cache(tmp_path, monkeypatch):
    calls = []

    def fake_fetch(series, lo, hi, session=None, timeout=None):
        calls.append((lo, hi))
        if lo == 1300 and len([c for c in calls if c[0] == 1300]) == 1:
            raise jsoc.JsocError("HTTP 503")
        lines = [_line(h, f"2014.01.0{d}_00:00:00_TAI", USFLUX=f"{h}.0") for h in range(lo, min(hi, 1450) + 1, 50)
                 for d in (1, 2)]
        return jsoc._parse_show_info("\n".join(lines) + "\n" if lines else "")

    monkeypatch.setattr(jsoc, "fetch_range", fake_fetch)
    monkeypatch.setattr(jsoc, "latest_harpnum", lambda s, session=None: 1450)
    monkeypatch.setattr(jsoc, "BACKOFF_S", (0,))
    monkeypatch.setattr(jsoc, "RETRIES", 0)
    store = DataStore(tmp_path)
    with pytest.raises(jsoc.JsocError):
        jsoc.download_sharp(store, "NRT", full=True, workers=1, log=lambda m: None)
    assert (store.sharp_dir("NRT") / block_name(0)).exists()        # block 0 complete, written
    assert not (store.sharp_dir("NRT") / block_name(1000)).exists()  # block 1000 incomplete
    n_before = len(calls)
    entry = jsoc.download_sharp(store, "NRT", full=True, workers=1, log=lambda m: None)
    resent = calls[n_before:]
    # the cached ranges of the interrupted block are not requested again
    assert all(lo >= 1300 for lo, _ in resent if lo >= 1000) and (1000, 1099) not in resent
    df = store.read_sharp("NRT")
    assert df.HARPNUM.nunique() == len(range(0, 1001, 50)) + len(range(1050, 1451, 50))
    assert df.groupby("HARPNUM").size().eq(2).all() and df["USFLUX"].dtype == np.float32
    assert entry["files"][block_name(1000)]["source"] == "jsoc"
    assert not list((tmp_path / ".cache").glob("**/*.parquet"))      # cache emptied
