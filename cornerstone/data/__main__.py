"""Command line of the data layer.

    python -m cornerstone.data status [--root DIR]
    python -m cornerstone.data bootstrap-sqlite DB.sqlite [--root DIR] [--series DEF NRT] [--no-references]
"""
from __future__ import annotations

import argparse

import pandas as pd

from .bootstrap import bootstrap_from_sqlite
from .config import load_config
from .store import DataStore


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cornerstone.data", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="store folder (default: data_root of data_config.json)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="datasets present in the store")
    b = sub.add_parser("bootstrap-sqlite", help="create the store from a legacy SQLite database")
    b.add_argument("sqlite")
    b.add_argument("--series", nargs="+", default=["DEF", "NRT"], choices=["DEF", "NRT"])
    b.add_argument("--no-references", action="store_true", help="do not copy the legacy reference tables")
    a = ap.parse_args(argv)
    store = DataStore(a.root or load_config()["data_root"])
    if a.cmd == "bootstrap-sqlite":
        bootstrap_from_sqlite(a.sqlite, store, series=a.series, references=not a.no_references)
    m = store.load_manifest()
    print(f"Store: {store.root}")
    with pd.option_context("display.width", 160):
        for name, d in sorted(m["datasets"].items()):
            print(f"  {name:20s} {d['rows']:>12,} rows  {d['bytes'] / 1e6:9.1f} MB  updated {d['updated_at']}")
    if not m["datasets"]:
        print("  (empty)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
