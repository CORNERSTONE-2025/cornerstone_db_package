"""Data layer: time scale, SHARP normalization, Parquet store and bootstrap from a legacy SQLite DB."""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cornerstone.data import DataStore, bootstrap_from_sqlite, sharp
from cornerstone.data.timescale import parse_t_obs, tai_minus_utc


def test_tai_to_utc_leap_seconds():
    t = parse_t_obs(pd.Series(["2010.05.01_00:00:04_TAI", "2012.07.01_00:00:34_TAI",
                               "2012.07.01_00:00:35_TAI", "2017.01.01_00:00:37_TAI", "MISSING"]))
    assert t.iloc[0] == pd.Timestamp("2010-04-30 23:59:30", tz="UTC")          # TAI - UTC = 34 s
    assert t.iloc[1] == pd.Timestamp("2012-06-30 23:59:60".replace(":60", ":59"), tz="UTC") + pd.Timedelta(seconds=1)
    assert t.iloc[2] == pd.Timestamp("2012-07-01 00:00:00", tz="UTC")           # 35 s from 2012-07-01
    assert t.iloc[3] == pd.Timestamp("2017-01-01 00:00:00", tz="UTC")           # 37 s from 2017-01-01
    assert pd.isna(t.iloc[4])
    assert tai_minus_utc(pd.to_datetime(["2025-01-01"])).tolist() == [37]


def _raw(harps=(1, 2, 1001), n=3):
    rows = []
    for h in harps:
        for i in range(n):
            r = {k: 1.5 for k in sharp.FLOAT_KEYS}
            r.update(T_OBS=f"2014.01.0{i + 1}_00:00:00_TAI", HARPNUM=h, QUALITY="0x00010400",
                     NOAA_AR=11000 + h, NOAA_NUM=1, NOAA_ARS=f"{11000 + h}", USFLUX=1e22 * (i + 1))
            rows.append(r)
    return pd.DataFrame(rows)


def test_normalize_types_order_duplicates():
    raw = _raw()
    raw = pd.concat([raw.iloc[::-1], raw.iloc[[0]]], ignore_index=True)          # unsorted + duplicate
    raw.loc[len(raw)] = {**raw.iloc[0].to_dict(), "T_OBS": "MISSING"}           # unusable row
    df = sharp.normalize(raw)
    assert len(df) == 9
    keys = list(zip(df.HARPNUM, df.T_OBS))
    assert keys == sorted(keys) and len(set(keys)) == len(keys)
    assert df["USFLUX"].dtype == np.float32 and df["QUALITY"].iloc[0] == 0x00010400
    assert str(df["T_OBS"].dt.tz) == "UTC"
    t = sharp.to_table(df)
    assert t.schema == sharp.SCHEMA


def test_store_roundtrip_and_filters(tmp_path):
    store = DataStore(tmp_path)
    df = sharp.normalize(_raw(harps=(1, 2)))
    info = store.write_sharp_block("DEF", 0, df)
    assert info["rows"] == 6 and len(info["sha256"]) == 64
    assert not list(tmp_path.glob("**/*.partial"))
    got = store.read_sharp("DEF", columns=["HARPNUM", "T_OBS", "USFLUX"], harpnums=[2],
                           start="2014-01-01T12:00", end="2014-01-02T12:00")   # 2nd record only
    assert got["HARPNUM"].tolist() == [2] and got["USFLUX"].tolist() == [np.float32(2e22)]
    with pytest.raises(ValueError):
        store.write_sharp_block("DEF", 1000, df)                                 # wrong block
    cat = store.catalog()
    assert cat["rows"].sum() == 6 and cat["dataset"].tolist() == ["raw/sharp/DEF"]


def test_bootstrap_from_legacy_sqlite(tmp_path):
    db = tmp_path / "legacy.sqlite"
    con = sqlite3.connect(db)
    raw = _raw(harps=(5, 7, 1003, 2500))
    for s in ("DEF", "NRT"):
        raw.assign(T_OBS1="x").to_sql(f"SHARPC_all_{s}", con, index=False)
    pd.DataFrame({"tmax": ["2014-01-02T00:00:00"], "fflux": [1e-5], "HARPNUM": [5]}).to_sql("NOAA_SHARP_AR", con, index=False)
    pd.DataFrame({"t_X": ["2014-01-01T00:00:00"], "HARPn": [5], "label": ["YES"]}).to_sql(
        "DF_NRT_Mplus_delta24_three_classes_unified", con, index=False)
    con.close()
    store = DataStore(tmp_path / "store")
    out = bootstrap_from_sqlite(db, store, log=lambda m: None)
    assert out["sharp/DEF"]["rows"] == 12
    assert sorted(out["sharp/DEF"]["files"]) == ["harp_000000-000999.parquet", "harp_001000-001999.parquet",
                                                  "harp_002000-002999.parquet"]
    back = store.read_sharp("NRT")
    pd.testing.assert_frame_equal(back, sharp.normalize(raw), check_dtype=False)
    m = store.load_manifest()["datasets"]
    assert set(m) == {"sharp/DEF", "sharp/NRT", "reference/legacy"}
    assert set(m["reference/legacy"]["files"]) == {"NOAA_SHARP_AR.parquet", "DF_NRT_Mplus_delta24_three_classes.parquet"}
    # a second import replaces the blocks and removes those without data any more
    con = sqlite3.connect(db)
    con.execute("DELETE FROM SHARPC_all_DEF WHERE HARPNUM = 2500")
    con.commit()
    con.close()
    out = bootstrap_from_sqlite(db, store, series=["DEF"], references=False, log=lambda m: None)
    assert out["sharp/DEF"]["rows"] == 9 and len(store.sharp_files("DEF")) == 2


def test_bootstrap_from_snapshot(tmp_path, monkeypatch):
    import shutil
    from types import SimpleNamespace

    import gdown

    from cornerstone.data import bootstrap as bs
    src = DataStore(tmp_path / "published")
    src.record_dataset("sharp/DEF", {"harp_000000-000999.parquet":
                                     src.write_sharp_block("DEF", 0, sharp.normalize(_raw(harps=(1, 2))))})

    def fake_listing(url, output, quiet, skip_download, **kw):         # file id = path in the folder
        assert skip_download
        return [SimpleNamespace(id=p.relative_to(src.root).as_posix(), local_path=str(Path(output) / p.relative_to(src.root)))
                for p in src.root.rglob("*") if p.is_file()]

    calls = []

    def fake_get(file_id, dst, **kw):
        calls.append(file_id)
        shutil.copyfile(src.root / file_id, dst)

    monkeypatch.setattr(gdown, "download_folder", fake_listing)
    monkeypatch.setattr(bs, "_drive_get", fake_get)
    store = DataStore(tmp_path / "local")
    bs.bootstrap_from_snapshot(store, url="https://drive.example/folder", log=lambda m: None)
    assert len(store.read_sharp("DEF")) == 6 and "sharp/DEF" in store.load_manifest()["datasets"]
    assert not list(store.root.glob(".snapshot-*"))
    # a corrupted file is refused; the partial download is kept and a new call resumes from it
    good = (src.sharp_dir("DEF") / "harp_000000-000999.parquet").read_bytes()
    (src.sharp_dir("DEF") / "harp_000000-000999.parquet").write_bytes(b"corrupted")
    other = DataStore(tmp_path / "other")
    with pytest.raises(RuntimeError, match="corrupted"):
        bs.bootstrap_from_snapshot(other, url="x", log=lambda m: None)
    assert (other.root / bs.SNAPSHOT_PARTIAL).is_dir()
    (src.sharp_dir("DEF") / "harp_000000-000999.parquet").write_bytes(good)
    calls.clear()
    bs.bootstrap_from_snapshot(other, url="x", log=lambda m: None)
    assert calls == ["manifest.json", "raw/sharp/DEF/harp_000000-000999.parquet"]
    assert len(other.read_sharp("DEF")) == 6
