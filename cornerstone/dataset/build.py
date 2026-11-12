"""Build the HDF5 dataset of SHARP time windows for labeled (t_X, HARP) pairs.

For every label row the window holds the records of the HARP with ``T_OBS`` (rounded to the
sampling interval) in ``[t_X - window_hours, t_X]``, on the regular grid of the sampling interval
(``window_hours * 60 / sampling_minutes + 1`` steps, NaN where no record): the same windows as the
former ``create_hdf5_from_db``. Rows whose window holds no record are left out and counted.

Layout (compatible with ``SHARPDataset`` and the GUI)::

    X              (N, T, F) float32, NaN = missing
    Y              (N,) bytes   prediction label YES / NO1..NO4
    label_<name>   (N,) bytes   same as Y
    label_u_<name> (N,) bytes   stratification class: C/M/X... or NO1..NO4 (used for the splits)
    HARPNUM        (N,) int32
    timestamps     (N, T) bytes ``YYYY-mm-dd HH:MM:SS`` (UTC)
    QUALITY        (N,) bytes   most frequent QUALITY of the window (``0x%08x``, ``NULL`` if none)
    attrs: feature_names, database_version, table_name, metadata (JSON: every parameter + provenance)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from ..data import DataStore
from ..data.store import HARP_BLOCK, now_iso
from ..processing import QualityFilter, derive, resolve_features, source_columns


def label_name(label_config: dict | None) -> str:
    """Name of the label datasets, e.g. ``Mplus_delta24`` (``<threshold>plus_delta<W>``)."""
    if not label_config:
        return "label"
    w = float(label_config.get("flare_window_hours", 24))
    return f"{label_config.get('threshold', 'M')}plus_delta{w:g}"


def _mode_hex(q: np.ndarray) -> np.ndarray:
    """Most frequent QUALITY per row of ``q`` (int64, -1 = missing) as ``0x%08x`` bytes."""
    out = np.full(len(q), b"NULL", dtype="S10")
    valid = q >= 0
    first = np.where(valid.any(1), q.max(1), -1)
    same = ~((q != first[:, None]) & valid).any(1) & (first >= 0)
    out[same] = [b"0x%08x" % v for v in first[same]]
    for i in np.flatnonzero(~same & valid.any(1)):
        vals, counts = np.unique(q[i][valid[i]], return_counts=True)
        out[i] = b"0x%08x" % vals[np.argmax(counts)]
    return out


class _Writer:
    """Appends samples to resizable HDF5 datasets."""

    def __init__(self, f: h5py.File, n_steps: int, n_feat: int, name: str):
        kw = dict(maxshape=(None,), chunks=True)
        self.d = {
            "X": f.create_dataset("X", (0, n_steps, n_feat), "float32", maxshape=(None, n_steps, n_feat),
                                  chunks=(64, n_steps, n_feat), compression="gzip", compression_opts=4),
            "Y": f.create_dataset("Y", (0,), "S3", **kw),
            f"label_{name}": f.create_dataset(f"label_{name}", (0,), "S3", **kw),
            f"label_u_{name}": f.create_dataset(f"label_u_{name}", (0,), "S3", **kw),
            "HARPNUM": f.create_dataset("HARPNUM", (0,), "int32", **kw),
            "timestamps": f.create_dataset("timestamps", (0, n_steps), "S19", maxshape=(None, n_steps),
                                           chunks=(256, n_steps), compression="gzip", compression_opts=4),
            "QUALITY": f.create_dataset("QUALITY", (0,), "S10", **kw),
        }
        self.name, self.n = name, 0

    def append(self, X, y, strat, harp, ts, qual):
        m = len(X)
        if not m:
            return
        values = {"X": X, "Y": y, f"label_{self.name}": y, f"label_u_{self.name}": strat, "HARPNUM": harp,
                  "timestamps": ts, "QUALITY": qual}
        for k, ds in self.d.items():
            ds.resize(self.n + m, axis=0)
            ds[self.n:self.n + m] = values[k]
        self.n += m


FLUSH_SAMPLES = 20_000          # samples kept in memory before writing


def _flush(w: _Writer, out: dict) -> None:
    if out["X"]:
        w.append(*(np.concatenate(out[k]) for k in ("X", "y", "s", "h", "ts", "q")))
    for v in out.values():
        v.clear()


def build_dataset(store: DataStore, series: str, labels: pd.DataFrame, path: str | Path,
                  features: str | list[str] = "clean", quality: QualityFilter | None = None,
                  window_hours: float = 24, sampling_minutes: int = 12, label_meta: dict | None = None,
                  max_samples: int | None = None, seed: int = 42, log=print, progress=None) -> dict:
    """Write the HDF5 dataset of ``labels`` (rows ``t_X``, ``HARPNUM``, ``label``, ``strat_class``,
    e.g. from ``cornerstone.labeling.label_from_store``) to ``path`` and return its metadata.

    Parameters
    ----------
    features:
        ``"clean"`` (normalized features of the former ``SHARPC_clean`` tables), ``"all"`` (raw
        keywords) or a list of names (``KEY``, ``KEY_N1`` = KEY/CMASK, ``KEY_N2`` = KEY/MEANGBT).
    quality:
        records used in the windows; default ``QualityFilter.legacy_clean(series)`` for
        ``"clean"`` (the rows of ``SHARPC_clean``), ``QualityFilter()`` otherwise.
    max_samples:
        random subset of the label rows (quick tests), drawn with ``seed``.
    progress:
        optional ``progress(done_harps, total_harps)``.
    """
    t0 = time.time()
    feats = resolve_features(features)
    if quality is None:
        quality = QualityFilter.legacy_clean(series) if features == "clean" else QualityFilter()
    if (window_hours * 60) % sampling_minutes:
        raise ValueError("window_hours * 60 must be a multiple of sampling_minutes")
    n_steps = int(window_hours * 60 // sampling_minutes) + 1
    samp_ns = pd.Timedelta(minutes=sampling_minutes).value
    name = label_name((label_meta or {}).get("label_config"))

    lab = labels[["t_X", "HARPNUM", "label", "strat_class"]].copy()
    if max_samples is not None and len(lab) > max_samples:
        lab = lab.sample(n=max_samples, random_state=seed)
    lab["_k"] = pd.to_datetime(lab["t_X"], utc=True).astype("datetime64[ns, UTC]").to_numpy("datetime64[ns]") \
        .astype("int64")
    if (lab["_k"] % samp_ns).any():
        raise ValueError(f"every t_X must lie on the {sampling_minutes}-minute grid")
    lab["_k"] //= samp_ns
    lab = lab.sort_values(["HARPNUM", "_k"], kind="stable")
    harps = lab["HARPNUM"].unique()
    raw_cols = source_columns(feats)
    cols = list(dict.fromkeys(["HARPNUM", "T_OBS", "QUALITY", *raw_cols, *quality.columns()]))
    offsets = np.arange(-n_steps + 1, 1)
    missing = []
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    log(f"[dataset] {series}: {len(lab):,} label rows, {len(harps):,} HARPs, {len(feats)} features, "
        f"{n_steps} steps of {sampling_minutes} min")
    with h5py.File(tmp, "w") as f:
        w = _Writer(f, n_steps, len(feats), name)
        lab_by_harp = dict(tuple(lab.groupby("HARPNUM", sort=False)))
        for b0 in range(0, len(harps), HARP_BLOCK):
            block = harps[b0:b0 + HARP_BLOCK]
            rec = quality.apply(store.read_sharp(series, columns=cols, harpnums=block))
            rec = derive(rec, feats)
            t = pd.to_datetime(rec["T_OBS"], utc=True).astype("datetime64[ns, UTC]").dt.round(f"{sampling_minutes}min")
            rec = rec.assign(_k=t.to_numpy("datetime64[ns]").astype("int64") // samp_ns)
            rec = rec.sort_values(["HARPNUM", "T_OBS"]).drop_duplicates(["HARPNUM", "_k"], keep="first")
            rec_by_harp = dict(tuple(rec.groupby("HARPNUM", sort=False)))
            out = {k: [] for k in ("X", "y", "s", "h", "ts", "q")}
            for h in block:
                lh = lab_by_harp[h]
                kx = lh["_k"].to_numpy()
                r = rec_by_harp.get(h)
                k0, k1 = kx.min() - n_steps + 1, kx.max()
                dense = np.full((k1 - k0 + 1, len(feats)), np.nan, dtype="float32")
                qd = np.full(k1 - k0 + 1, -1, dtype="int64")
                if r is not None:
                    rk = r["_k"].to_numpy()
                    sel = (rk >= k0) & (rk <= k1)
                    dense[rk[sel] - k0] = r.loc[sel, feats].to_numpy("float32")
                    qd[rk[sel] - k0] = r.loc[sel, "QUALITY"].fillna(-1).to_numpy("int64")
                idx = (kx - k0)[:, None] + offsets[None, :]        # rows of dense, shape (m, T), >= 0
                X = dense[idx]
                has = ~np.isnan(X).all(axis=(1, 2))
                if (~has).any():
                    for t_x, lbl in zip(lh["t_X"].to_numpy()[~has], lh["label"].to_numpy()[~has]):
                        missing.append({"HARPNUM": int(h), "t_X": str(t_x), "label": lbl,
                                        "reason": "no record in the window" if r is not None else "no record of the HARP"})
                if not has.any():
                    continue
                grid = pd.to_datetime(np.arange(k0, k1 + 1) * samp_ns).strftime("%Y-%m-%d %H:%M:%S")
                grid = np.asarray(grid, dtype="S19")
                out["X"].append(X[has])
                out["y"].append(lh["label"].to_numpy()[has].astype("S3"))
                out["s"].append(lh["strat_class"].to_numpy()[has].astype("S3"))
                out["h"].append(np.full(has.sum(), h, dtype="int32"))
                out["ts"].append(grid[idx[has]])
                out["q"].append(_mode_hex(qd[idx[has]]))
                if sum(map(len, out["X"])) >= FLUSH_SAMPLES:
                    _flush(w, out)
            _flush(w, out)
            if progress:
                progress(min(b0 + HARP_BLOCK, len(harps)), len(harps))
            log(f"[dataset] HARPs {min(b0 + HARP_BLOCK, len(harps)):,}/{len(harps):,}: {w.n:,} samples")
        y = w.d["Y"][:]
        s = w.d[f"label_u_{name}"][:]
        manifest = store.load_manifest()["datasets"]
        meta = {
            "created_at": now_iso(),
            "series": series,
            "features": feats,
            "feature_set": features if isinstance(features, str) else "custom",
            "quality_filter": quality.to_dict(),
            "window_hours": window_hours,
            "sampling_minutes": sampling_minutes,
            "n_steps": n_steps,
            "label_name": name,
            "labeling": label_meta or {},
            "max_samples": max_samples,
            "seed": seed if max_samples is not None else None,
            "data": {k: {kk: vv for kk, vv in d.items() if kk in ("updated_at", "source", "last_download")}
                     for k, d in manifest.items() if k == f"sharp/{series}"},
            "counts": {"label_rows": int(len(lab)), "samples": int(w.n), "missing": len(missing),
                       "label": {k.decode(): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
                       "strat_class": {k.decode(): int(v) for k, v in zip(*np.unique(s, return_counts=True))}},
            "duration_s": round(time.time() - t0, 1),
        }
        f.attrs["feature_names"] = ",".join(feats)
        f.attrs["database_version"] = meta["feature_set"]
        f.attrs["table_name"] = f"store:sharp/{series}"
        f.attrs["metadata"] = json.dumps(meta, default=str)
    tmp.replace(path)
    if missing:
        pd.DataFrame(missing).to_csv(path.with_suffix(".missing.csv"), index=False)
    log(f"[dataset] {path.name}: {w.n:,} samples, {len(missing):,} label rows without data "
        f"({meta['duration_s']} s)")
    return meta
