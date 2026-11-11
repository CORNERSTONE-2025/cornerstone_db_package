"""Download of the SHARP keywords from JSOC into the Parquet store.

Records are requested with the JSOC ``show_info`` CGI (tab-separated text), by ranges of
``HARPS_PER_REQUEST`` HARP numbers, a few requests in parallel. ``show_info`` was preferred to the
JSON ``jsoc_info`` (used by the ``drms`` client): on 100 HARPs with all the keywords ``jsoc_info``
answers ``status 6`` while ``show_info`` returns the ~50k records in ~40 s.

Every answered range is cached on disk, and a store block (1000 HARPs) is written only when all
its ranges are there: an interrupted download resumes from the cached ranges. Updates download
only the blocks that are missing or may still change (HARPs observed recently).
"""
from __future__ import annotations

import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests

from . import sharp
from .store import HARP_BLOCK, DataStore, block_name, now_iso, say

SHOW_INFO_URL = "http://jsoc.stanford.edu/cgi-bin/ajax/show_info"
HARPS_PER_REQUEST = 100
WORKERS = 3                      # parallel requests: moderate, JSOC is a shared service
TIMEOUT_S = 900
RETRIES = 5
BACKOFF_S = (10, 30, 90, 180, 300)
RECENT_DAYS = 60                 # blocks with records newer than this may still change
USER_AGENT = "cornerstone-data (python-requests)"


class JsocError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------- requests
def _parse_show_info(text: str, keys: tuple[str, ...] = sharp.KEYWORDS) -> pd.DataFrame:
    """Records of a ``show_info q=1`` answer; lines that are not records (warnings) are skipped.

    A truncated answer (last record with missing fields) raises, so that it is requested again.
    """
    rows = []
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if not line:
            continue
        f = line.split("\t")
        if len(f) == len(keys):
            rows.append(f)
        elif "\t" in line:
            raise JsocError(f"Malformed record at line {i + 1} ({len(f)} fields instead of {len(keys)})")
        # else: messages such as "series hmi.Marmask_720s not found" printed by show_info
    if text and not text.endswith("\n") and lines[-1].count("\t") == len(keys) - 1:
        raise JsocError("Truncated answer")
    return pd.DataFrame(rows, columns=list(keys))


def fetch_range(series: str, first: int, last: int, session: requests.Session | None = None,
                timeout: float = TIMEOUT_S) -> pd.DataFrame:
    """All records of HARPs ``first..last`` of a SHARP series (DEF/NRT), raw strings."""
    ds = f"{sharp.SERIES[series]}[{first}-{last}][]"
    s = session or requests.Session()
    r = s.get(SHOW_INFO_URL, params={"ds": ds, "q": "1", "key": ",".join(sharp.KEYWORDS)},
              timeout=timeout, headers={"User-Agent": USER_AGENT})
    if r.status_code != 200:
        raise JsocError(f"HTTP {r.status_code} for {ds}")
    return _parse_show_info(r.text)


def fetch_range_retry(series: str, first: int, last: int, session: requests.Session | None = None,
                      log: Callable[[str], None] = say, sleep: Callable[[float], None] = time.sleep
                      ) -> pd.DataFrame:
    for attempt in range(RETRIES + 1):
        try:
            return fetch_range(series, first, last, session)
        except (requests.RequestException, JsocError) as e:
            if attempt == RETRIES:
                raise JsocError(f"{series} HARP {first}-{last}: {e} (after {RETRIES} retries)") from e
            wait = BACKOFF_S[min(attempt, len(BACKOFF_S) - 1)]
            log(f"   {series} HARP {first}-{last}: {e}; retry in {wait}s")
            sleep(wait)
    raise AssertionError("unreachable")


def latest_harpnum(series: str, session: requests.Session | None = None) -> int:
    """Highest HARP number of the series (``[$]`` selects the last HARP)."""
    s = session or requests.Session()
    r = s.get(SHOW_INFO_URL, params={"ds": f"{sharp.SERIES[series]}[$]", "q": "1", "key": "HARPNUM"},
              timeout=120, headers={"User-Agent": USER_AGENT})
    nums = [int(x) for x in r.text.split() if x.isdigit()]
    if r.status_code != 200 or not nums:
        raise JsocError(f"Cannot read the last HARP number of {sharp.SERIES[series]}")
    return max(nums)


# ---------------------------------------------------------------------------------------- planning
def blocks_to_update(store: DataStore, series: str, last_harp: int, full: bool = False,
                     recent_days: int = RECENT_DAYS) -> list[int]:
    """Store blocks to (re)download: missing ones, and those whose data may still change.

    HARP numbers are assigned in time order and a HARP is observed from its creation, so a block
    is complete when, at the time its data were taken (``data_until``: download time, or end of
    the dump for blocks imported from it), its newest record was already ``recent_days`` old:
    its HARPs had ended and no new HARP number fell in the block. The block of the last HARP
    number is always updated.
    """
    starts = list(range(0, last_harp + 1, HARP_BLOCK))
    if full:
        return starts
    files = store.load_manifest()["datasets"].get(f"sharp/{series}", {}).get("files", {})
    todo = []
    for b in starts:
        info = files.get(block_name(b))
        if b == starts[-1] or info is None:
            todo.append(b)
            continue
        if info["rows"] == 0:
            continue                                     # no HARP of the series in this block
        if not (store.sharp_dir(series) / block_name(b)).exists():
            todo.append(b)
            continue
        newest, until = info.get("max_t_obs"), info.get("data_until")
        if newest is None or until is None or \
                pd.Timestamp(until) - pd.Timestamp(newest) < pd.Timedelta(days=recent_days):
            todo.append(b)
    return todo


# ---------------------------------------------------------------------------------------- update
def _cache_dir(store: DataStore, series: str) -> Path:
    return store.root / ".cache" / "jsoc" / series


def download_sharp(store: DataStore, series: str, full: bool = False, blocks: Iterable[int] | None = None,
                   workers: int = WORKERS, harps_per_request: int = HARPS_PER_REQUEST,
                   log: Callable[[str], None] = say,
                   progress: Callable[[int, int], None] | None = None) -> dict:
    """Download (or update) the SHARP records of ``series`` from JSOC into the store.

    ``full`` downloads every block; otherwise only ``blocks_to_update`` (or the given ``blocks``).
    ``progress(done, total)`` is called after every answered request.
    """
    t0 = time.time()
    session = requests.Session()
    last = latest_harpnum(series, session)
    todo = sorted(blocks) if blocks is not None else blocks_to_update(store, series, last, full=full)
    cache = _cache_dir(store, series)
    cache.mkdir(parents=True, exist_ok=True)
    ranges = [(b, lo, min(lo + harps_per_request - 1, b + HARP_BLOCK - 1))
              for b in todo for lo in range(b, b + HARP_BLOCK, harps_per_request) if lo <= last]
    pending = [r for r in ranges if not (cache / f"{r[1]:06d}-{r[2]:06d}.parquet").exists()]
    log(f"{sharp.SERIES[series]}: last HARP {last}; {len(todo)} blocks to update, "
        f"{len(pending)} of {len(ranges)} requests to send ({len(ranges) - len(pending)} cached)")
    done = len(ranges) - len(pending)
    if progress:
        progress(done, len(ranges))

    def get(r):
        _, lo, hi = r
        df = fetch_range_retry(series, lo, hi, session, log=log)
        table = sharp.to_table(sharp.normalize(df)) if len(df) else sharp.SCHEMA.empty_table()
        store.write_table(cache / f"{lo:06d}-{hi:06d}.parquet", table)      # atomic
        return r, table.num_rows

    written = {}

    def record(blk):
        # every block is registered as soon as it is written: an interrupted run keeps it
        store.record_dataset(f"sharp/{series}", {block_name(blk): written[blk]}, merge=True,
                             series=sharp.SERIES[series], time_scale="UTC (converted from TAI)")

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [ex.submit(get, r) for r in pending]
        for fut in as_completed(futures):
            (b, lo, hi), n = fut.result()
            done += 1
            log(f"   {series} HARP {lo}-{hi}: {n:,} records ({done}/{len(ranges)}, {time.time() - t0:.0f}s)")
            if progress:
                progress(done, len(ranges))
            # write the store blocks whose ranges are all cached
            for blk in sorted({x[0] for x in ranges} - set(written)):
                parts = [cache / f"{x[1]:06d}-{x[2]:06d}.parquet" for x in ranges if x[0] == blk]
                if all(p.exists() for p in parts):
                    written[blk] = _assemble_block(store, series, blk, parts)
                    record(blk)

    for blk in sorted({x[0] for x in ranges} - set(written)):      # nothing left to download
        parts = [cache / f"{x[1]:06d}-{x[2]:06d}.parquet" for x in ranges if x[0] == blk]
        written[blk] = _assemble_block(store, series, blk, parts)
        record(blk)
    entry = store.record_dataset(
        f"sharp/{series}", {}, merge=True,
        last_download={"at": now_iso(), "kind": "jsoc-show_info", "last_harpnum": last,
                       "blocks": [block_name(b) for b in sorted(written)],
                       "duration_s": round(time.time() - t0, 1)})
    log(f"{sharp.SERIES[series]}: {len(written)} blocks written in {time.time() - t0:.0f}s; "
        f"store {entry['rows']:,} records")
    return entry


def _assemble_block(store: DataStore, series: str, start: int, parts: list[Path]) -> dict:
    """Join the cached ranges of a block into the store file (Arrow only: types are preserved)."""
    table = pa.concat_tables([pq.read_table(p, schema=sharp.SCHEMA) for p in parts])
    path = store.sharp_dir(series) / block_name(start)
    info = {"source": "jsoc", "downloaded_at": now_iso(), "data_until": now_iso()}
    if table.num_rows == 0:
        # no HARP of this block exists in the series: no file
        if path.exists():
            path.unlink()
        info.update(rows=0, bytes=0, sha256=None, max_t_obs=None)
    else:
        table = table.sort_by([("HARPNUM", "ascending"), ("T_OBS", "ascending")])
        h = table.column("HARPNUM")
        if pc.min(h).as_py() < start or pc.max(h).as_py() >= start + HARP_BLOCK:
            raise ValueError(f"Records outside the HARP block starting at {start}")
        info.update(store.write_table(path, table), max_t_obs=pc.max(table.column("T_OBS")).as_py().isoformat())
    for p in parts:
        p.unlink()
    return info
