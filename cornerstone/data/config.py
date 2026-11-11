"""Configuration of the data layer (``cornerstone/configs/data_config.json``).

``data_root`` is the folder of the Parquet store; a relative path is resolved against the
package root. The environment variable ``CORNERSTONE_DATA_ROOT`` overrides it (Docker, CI).
``bootstrap_url`` is the published Parquet snapshot (Google Drive folder) used to create the
store without converting the former dump; ``null`` until it is published.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PACKAGE_ROOT / "cornerstone" / "configs" / "data_config.json"
DEFAULTS = {"data_root": "datastore", "bootstrap_url": None}


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Data configuration with defaults; paths resolved to absolute paths."""
    p = Path(path) if path else CONFIG_PATH
    cfg = dict(DEFAULTS)
    if p.exists():
        cfg.update(json.loads(p.read_text()))
    root = os.environ.get("CORNERSTONE_DATA_ROOT") or cfg["data_root"]
    root = Path(root).expanduser()
    cfg["data_root"] = str(root if root.is_absolute() else PACKAGE_ROOT / root)
    return cfg


def default_store():
    """The store configured for this installation."""
    from .store import DataStore
    return DataStore(load_config()["data_root"])
