"""Download/update of all the data sources into the store (used by the CLI and the GUI)."""
from __future__ import annotations

import time
from typing import Callable, Iterable

from . import goes, harp_noaa, jsoc
from .store import DataStore, say

SOURCES = ("harp_noaa", "goes", "sharp_DEF", "sharp_NRT")


def update_all(store: DataStore, sources: Iterable[str] = SOURCES, full: bool = False,
               log: Callable[[str], None] = say,
               progress: Callable[[str, int, int], None] | None = None) -> dict:
    """Update the chosen sources; small ones first. ``progress(source, done, total)``.

    Each source is independent: a failure is reported and the others still run; the function
    raises at the end if any source failed.
    """
    t0 = time.time()
    results, errors = {}, {}
    for src in sources:
        if src not in SOURCES:
            raise ValueError(f"Unknown source {src!r}: expected one of {SOURCES}")
        try:
            if src == "harp_noaa":
                results[src] = harp_noaa.download(store, log=log)
            elif src == "goes":
                results[src] = goes.download(store, full=full, log=log)
            else:
                series = src.split("_")[1]
                cb = (lambda d, t, s=src: progress(s, d, t)) if progress else None
                results[src] = jsoc.download_sharp(store, series, full=full, log=log, progress=cb)
        except Exception as e:  # noqa: BLE001 - reported per source
            errors[src] = f"{type(e).__name__}: {e}"
            log(f"!! {src} failed: {errors[src]}")
    log(f"Update finished in {time.time() - t0:.0f}s" + (f" with errors in {sorted(errors)}" if errors else ""))
    if errors:
        raise RuntimeError("Update failed for: " + "; ".join(f"{k}: {v}" for k, v in errors.items()))
    return results
