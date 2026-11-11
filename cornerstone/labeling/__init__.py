"""CORNERSTONE parametric labeling (replaces the static ``DF_*_three_classes`` tables and
``Cornerstone.splitting_labeling``).

- ``label(spans, flares, config)``: pure function, see ``core`` for the definitions;
- ``label_from_store(store, series, ...)``: spans and flares from the Parquet store, plus the
  metadata to save with the labels (all parameters, data provenance, counts);
- ``validation.compare_with_legacy``: reproduction of the former static tables.
"""
from __future__ import annotations

import time

import pandas as pd

from ..data import DataStore
from ..processing import FlareConfig, QualityFilter, associate, harp_spans
from .core import LABELS, LabelConfig, class_letter, issue_grid, label, threshold_flux

__all__ = ["LabelConfig", "label", "label_from_store", "threshold_flux", "issue_grid", "class_letter", "LABELS"]


def label_from_store(store: DataStore, series: str, config: LabelConfig = LabelConfig(),
                     quality: QualityFilter = QualityFilter(), flares: FlareConfig = FlareConfig()
                     ) -> tuple[pd.DataFrame, dict]:
    """Labels of the HARPs of ``series`` (DEF/NRT) from the store, and their metadata."""
    t0 = time.time()
    spans = harp_spans(store, series, quality)
    assoc, assoc_stats = associate(store, series, quality, flares)
    out = label(spans, assoc, config)
    manifest = store.load_manifest()["datasets"]
    sharp_entry = manifest.get(f"sharp/{series}", {})
    meta = {
        "series": series,
        "label_config": config.to_dict(),
        "quality_filter": quality.to_dict(),
        "flare_config": flares.to_dict(),
        "flare_association": assoc_stats,
        "data": {name: {k: v for k, v in d.items() if k in ("updated_at", "source", "last_download", "rows")}
                 for name, d in manifest.items() if name in (f"sharp/{series}", "goes/swpc_events", "harp_noaa")},
        "sharp_data_until": max((f.get("data_until") or "" for f in sharp_entry.get("files", {}).values()),
                                default=None),
        "counts": {"rows": len(out), "harps": int(out["HARPNUM"].nunique()) if len(out) else 0,
                   "t_X": [str(out["t_X"].min()), str(out["t_X"].max())] if len(out) else None,
                   "label": out["label"].value_counts().to_dict(),
                   "strat_class": out["strat_class"].value_counts().to_dict()},
        "duration_s": round(time.time() - t0, 1),
    }
    return out, meta
