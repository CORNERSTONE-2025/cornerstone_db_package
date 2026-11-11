"""Processing: quality filter, HARP spans/regions, GOES calibration, one satellite per flare,
flare -> HARP association rules. Synthetic store, no network."""
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from cornerstone.data import DataStore, goes, harp_noaa, sharp
from cornerstone.processing import FlareConfig, QualityFilter, associate, calibrate, flux_to_class, harp_regions, harp_spans
from cornerstone.processing.flares import one_satellite_per_flare

T0 = pd.Timestamp("2014-01-01", tz="UTC")


def _records(harp, hours, box, usflux=1e22, ars="11944", empty_at=()):
    rows = []
    for h in hours:
        r = {k: np.float32(1.0) for k in sharp.FLOAT_KEYS}
        lat0, lat1, lon0, lon1 = box
        r.update(T_OBS=T0 + pd.Timedelta(hours=h), HARPNUM=harp, QUALITY=0, NOAA_AR=int(ars.split(",")[0]),
                 NOAA_NUM=len(ars.split(",")), NOAA_ARS=ars, USFLUX=usflux,
                 LAT_MIN=lat0, LAT_MAX=lat1, LON_MIN=lon0, LON_MAX=lon1)
        if h in empty_at:                                   # empty patch: no position
            r.update(LAT_MIN=np.nan, LAT_MAX=np.nan, LON_MIN=np.nan, LON_MAX=np.nan, USFLUX=0.0)
        rows.append(r)
    return rows


def _event(eid, hour, cls, ar, sat="G15", loc=None, lat=np.nan, lon=np.nan, event=None):
    t = T0 + pd.Timedelta(hours=hour)
    return {"event_id": eid, "event": event or eid.rsplit("-", 1)[0], "tstart": t - pd.Timedelta(minutes=10),
            "tmax": t, "tend": t + pd.Timedelta(minutes=10), "satellite": sat, "quality": "5", "goes_class": cls,
            "flux_reported": goes.class_to_flux(cls), "integrated_flux": np.nan, "region_reported": ar % 10000,
            "NOAA_AR": ar, "location": loc, "lat": lat, "lon": lon}


@pytest.fixture
def store(tmp_path):
    s = DataStore(tmp_path)
    recs = (_records(10, range(0, 48), (-25, -15, 0, 20), usflux=5e22, ars="11944", empty_at=(47,))   # big
            + _records(11, range(0, 48), (5, 15, -40, -30), usflux=1e21, ars="11944")                 # small
            + _records(12, range(60, 90), (10, 20, 30, 40), ars="11950"))
    df = pd.DataFrame(recs)
    df["QUALITY"] = df["QUALITY"].astype("UInt32")
    for c in ("NOAA_AR",):
        df[c] = df[c].astype("Int32")
    df["NOAA_NUM"] = df["NOAA_NUM"].astype("Int16")
    df["NOAA_ARS"] = df["NOAA_ARS"].astype("string")
    for c in sharp.FLOAT_KEYS:
        df[c] = df[c].astype("float32")
    for series in ("DEF", "NRT"):
        s.write_sharp_block(series, 0, df.sort_values(["HARPNUM", "T_OBS"]))
    s.write_table(harp_noaa.path(s), pa.Table.from_pandas(
        pd.DataFrame({"HARPNUM": [10, 11, 12], "NOAA_AR": [11944, 11944, 11950]}).astype("int32"),
        schema=harp_noaa.SCHEMA, preserve_index=False))
    ev = pd.DataFrame([
        _event("20140101-1-1", 5, "M1.0", 11944, loc="S20W10", lat=-20, lon=10),     # ambiguous, position -> 10
        _event("20140101-2-1", 6, "C5.0", 11944, loc="N10E35", lat=10, lon=-35),     # ambiguous, position -> 11
        _event("20140101-3-1", 7, "X1.0", 11944),                                     # ambiguous, no position -> flux -> 10
        _event("20140103-4-1", 58, "M2.0", 11950),                                    # 2 h before HARP 12 -> nearest
        _event("20140105-5-1", 140, "M2.0", 11950),                                   # 50 h after: not associated
        _event("20140101-6-1", 8, "M3.0", 11944, sat="G15", event="20140101-6"),      # same flare, two GOES:
        _event("20140101-6-2", 8, "M2.9", 11944, sat="G14", event="20140101-6"),      # G15 is primary that day
    ])
    ev = goes._typed(ev[goes.SCHEMA.names])
    s.write_table(goes.year_path(s, 2014), pa.Table.from_pandas(ev, schema=goes.SCHEMA, preserve_index=False))
    return s


def test_quality_filter_and_spans(store):
    spans = harp_spans(store, "DEF").set_index("HARPNUM")
    assert spans.loc[10, "n_obs"] == 47 and spans.loc[10, "end"] == T0 + pd.Timedelta(hours=46)   # empty patch out
    all_rec = harp_spans(store, "DEF", QualityFilter(require_position=False)).set_index("HARPNUM")
    assert all_rec.loc[10, "n_obs"] == 48
    q = QualityFilter(max_abs_lon=30)
    assert set(harp_spans(store, "DEF", q)["HARPNUM"]) == {10}                 # 11, 12 reach |lon| > 30
    assert q.to_dict()["max_abs_lon"] == 30
    nrt = harp_regions(store, "NRT")
    assert nrt.values.tolist() == [[10, 11944], [11, 11944], [12, 11950]]


def test_calibration_and_classes():
    ev = pd.DataFrame({"flux_reported": [7e-5, 7e-5], "satellite": ["G15", "G16"]})
    assert calibrate(ev, "science").tolist() == pytest.approx([1e-4, 7e-5])    # G15: 0.7 scaling removed
    assert calibrate(ev, "operational").tolist() == pytest.approx([7e-5, 7e-5])
    assert [flux_to_class(f) for f in (1e-4, 9.99e-5, 1.23e-5, 2e-6, np.nan)] == ["X1.0", "M10.0", "M1.2", "C2.0", None]


def test_one_satellite_per_flare(store):
    ev = one_satellite_per_flare(goes.load(store))
    assert "20140101-6-2" not in set(ev["event_id"]) and "20140101-6-1" in set(ev["event_id"])


def test_association_rules(store):
    a, stats = associate(store, "DEF")
    got = dict(zip(a["event_id"], zip(a["HARPNUM"], a["rule"])))
    assert got["20140101-1-1"] == (10, "position")
    assert got["20140101-2-1"] == (11, "position")
    assert got["20140101-3-1"] == (10, "flux")
    assert got["20140103-4-1"] == (12, "nearest_span")
    assert "20140105-5-1" not in got
    assert stats["not_associated"] == 1 and stats["ambiguous"] == 4
    # science calibration: G15 M3.0 -> M4.3
    assert a.set_index("event_id").loc["20140101-6-1", "goes_class"] == "M4.3"

    allr, _ = associate(store, "DEF", config=FlareConfig(ambiguous="all"))
    assert sorted(allr[allr["event_id"] == "20140101-1-1"]["HARPNUM"]) == [10, 11]
    flux, _ = associate(store, "DEF", config=FlareConfig(ambiguous="flux"))
    assert flux.set_index("event_id").loc["20140101-2-1", "HARPNUM"] == 10      # position ignored
    strict, _ = associate(store, "DEF", config=FlareConfig(max_gap_hours=1))
    assert "20140103-4-1" not in set(strict["event_id"])
    op, _ = associate(store, "DEF", config=FlareConfig(calibration="operational"))
    assert op.set_index("event_id").loc["20140101-6-1", "goes_class"] == "M3.0"
    with pytest.raises(ValueError):
        FlareConfig(ambiguous="nope")
