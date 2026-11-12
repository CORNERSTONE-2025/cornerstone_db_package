"""Dataset construction from the Parquet store: clean filter, derived features, time windows,
HDF5 layout and stratification class read by the splitting code. Synthetic store, no network."""
import json

import h5py
import numpy as np
import pandas as pd
import pytest

from cornerstone.data import DataStore, sharp
from cornerstone.dataset import build_dataset, label_name
from cornerstone.processing import FEATURE_SETS, QualityFilter, derive, source_columns

T0 = pd.Timestamp("2014-03-01", tz="UTC")
STEP = pd.Timedelta(minutes=12)


def _block(harp, n, quality=0, cmask=6000.0, t0=T0, skip=()):
    rows = []
    for i in range(n):
        if i in skip:
            continue
        r = {k: np.float32(1.0) for k in sharp.FLOAT_KEYS}
        r.update(T_OBS=t0 + i * STEP - pd.Timedelta(seconds=35),        # UTC = TAI - 35 s, as in the store
                 HARPNUM=harp, QUALITY=quality, NOAA_AR=11900, NOAA_NUM=1, NOAA_ARS="11900",
                 USFLUX=float(i + 1) * 1e20, CMASK=cmask, TOTUSJZ=2.0, MEANGBZ=3.0, MEANGBT=2.0)
        rows.append(r)
    return rows


@pytest.fixture
def store(tmp_path):
    s = DataStore(tmp_path / "store")
    df = pd.DataFrame(_block(1, 241) + _block(2, 241, quality=0x10000) + _block(3, 241, cmask=100.0)
                      + _block(4, 241, skip=range(0, 121)))
    df = df.astype({"QUALITY": "UInt32", "NOAA_AR": "Int32", "NOAA_NUM": "Int16", "NOAA_ARS": "string",
                    **{c: "float32" for c in sharp.FLOAT_KEYS}})
    s.write_sharp_block("DEF", 0, df.sort_values(["HARPNUM", "T_OBS"]))
    return s


def _labels(harps, days=(1, 2)):
    rows = [{"t_X": T0 + pd.Timedelta(days=d), "HARPNUM": h, "label": "YES" if d == 2 else "NO2",
             "strat_class": "M" if d == 2 else "NO2"} for h in harps for d in days]
    return pd.DataFrame(rows)


def test_legacy_clean_filter():
    df = pd.DataFrame({"LAT_MIN": [1.0] * 6, "LON_MIN": 1.0, "CMASK": [6000, 6000, 5000, 6000, 6000, 6000],
                       "MEANGBT": [10, 10, 10, 0, 10, 10], "QUALITY": [0, 0x400, 0, 0, 0x10400, 0x400],
                       "T_OBS": pd.to_datetime(["2014-01-01T00:00:00"] * 5 + [
                                                "2013-12-31T23:59:40"], utc=True)})
    assert QualityFilter.legacy_clean("DEF").mask(df).tolist() == [True, False, False, False, False, False]
    # NRT: QUALITY exactly 0x400; the last record is 2014-01-01 00:00:15 TAI -> kept (TAI year)
    assert QualityFilter.legacy_clean("NRT").mask(df).tolist() == [False, True, False, False, False, True]
    df.loc[5, "T_OBS"] = pd.Timestamp("2013-12-31T23:59:00", tz="UTC")
    assert not QualityFilter.legacy_clean("NRT").mask(df)[5]                     # 2013 excluded
    assert QualityFilter.legacy_clean("NRT").to_dict()["exclude_years"] == [2013]


def test_derived_features():
    assert source_columns(["USFLUX_N1", "MEANGBZ_N2", "R_VALUE"]) == ["USFLUX", "CMASK", "MEANGBZ", "MEANGBT",
                                                                      "R_VALUE"]
    df = derive(pd.DataFrame({"USFLUX": [10.0, 5.0], "CMASK": [2.0, 0.0], "MEANGBZ": [3.0, 3.0],
                              "MEANGBT": [1.5, 3.0]}), ["USFLUX_N1", "MEANGBZ_N2"])
    assert df["USFLUX_N1"].tolist()[0] == 5.0 and np.isnan(df["USFLUX_N1"][1])
    assert df["MEANGBZ_N2"].tolist() == [2.0, 1.0]
    assert len(FEATURE_SETS["clean"]) == 17 and len(FEATURE_SETS["all"]) == 22


def test_build_dataset_windows(store, tmp_path):
    meta_in = {"label_config": {"threshold": "M", "flare_window_hours": 24}}
    out = tmp_path / "ds.h5"
    meta = build_dataset(store, "DEF", _labels([1, 2, 3, 4, 99]), out, label_meta=meta_in, log=lambda *_: None)
    assert label_name(meta_in["label_config"]) == "Mplus_delta24"
    with h5py.File(out, "r") as f:
        X, harp = f["X"][:], f["HARPNUM"][:]
        names = f.attrs["feature_names"].split(",")
        assert names == list(FEATURE_SETS["clean"])
        # HARP 2 (QUALITY != 0), 3 (CMASK <= 5000), 99 (absent) are out; HARP 4 has no record on day 1
        assert harp.tolist() == [1, 1, 4]
        assert X.shape == (3, 121, 17)
        assert f["Y"][:].tolist() == [b"NO2", b"YES", b"YES"]
        assert f["label_Mplus_delta24"][:].tolist() == f["Y"][:].tolist()
        assert f["label_u_Mplus_delta24"][:].tolist() == [b"NO2", b"M", b"M"]
        ts = f["timestamps"][:]
        assert ts[0, -1] == b"2014-03-02 00:00:00" and ts[0, 0] == b"2014-03-01 00:00:00"
        u = names.index("USFLUX_N1")
        # window of day 1 for HARP 1: records 0..120, USFLUX = (i+1)e20, CMASK 6000
        assert X[0, :, u] == pytest.approx(np.arange(1, 122) * 1e20 / 6000, rel=1e-6)
        assert X[1, 0, u] == pytest.approx(121e20 / 6000, rel=1e-6)
        # HARP 4: records from index 121 on -> nothing on day 1, day 2 misses only the first step
        assert np.isnan(X[2, 0]).all() and not np.isnan(X[2, 1:]).any()
        assert f["QUALITY"][:].tolist() == [b"0x00000000"] * 3
        m = json.loads(f.attrs["metadata"])
    assert m["counts"]["samples"] == 3 and m["counts"]["missing"] == 7
    assert m["quality_filter"]["cmask_gt"] == 5000 and meta["labeling"] == meta_in
    miss = pd.read_csv(out.with_suffix(".missing.csv"))
    assert set(miss["HARPNUM"]) == {2, 3, 4, 99}


def test_build_dataset_raw_features_and_grid_check(store, tmp_path):
    meta = build_dataset(store, "DEF", _labels([2, 3]), tmp_path / "raw.h5", features="all", log=lambda *_: None)
    assert meta["counts"]["samples"] == 4 and meta["quality_filter"]["quality_values"] is None
    bad = _labels([1]).assign(t_X=lambda d: d["t_X"] + pd.Timedelta(minutes=5))
    with pytest.raises(ValueError):
        build_dataset(store, "DEF", bad, tmp_path / "bad.h5", log=lambda *_: None)


def test_sharp_dataset_reads_stratification_class(store, tmp_path):
    torch = pytest.importorskip("torch")
    from cornerstone.sharp_data import SharpDataset

    out = tmp_path / "ds.h5"
    build_dataset(store, "DEF", _labels([1, 4]), out, label_meta={"label_config": {"threshold": "M"}},
                  log=lambda *_: None)
    ds = SharpDataset(str(out), label="Mplus_delta24")
    assert ds._raw_labels.tolist() == ["NO2", "YES", "YES"]
    assert ds._strat_labels.tolist() == ["NO2", "M", "M"]
    assert ds.Y.squeeze(1).tolist() == [0.0, 1.0, 1.0]
    mc = SharpDataset(str(out), label="Mplus_delta24", multi_class=True)
    assert mc.label_map == {"NO": 0, "M": 1} and mc.Y.tolist() == [0, 1, 1]
    assert isinstance(ds.X, torch.Tensor)
