"""CORNERSTONE data layer: local Parquet store of SHARP keywords, GOES flares and HARP-NOAA maps.

Independent of the ML stack (no torch/ClearML imports). Main entry points:

- ``DataStore``: files, catalogue, manifest, reading with column/HARP/time selection;
- ``update_all``: download/update from the sources (JSOC SHARP DEF/NRT, JSOC HARP-NOAA map,
  NOAA SWPC GOES events); modules ``jsoc``, ``harp_noaa``, ``goes`` for each source;
- ``bootstrap_from_snapshot``: create the store from the published Parquet snapshot (Drive);
- ``bootstrap_from_sqlite``: create the store from a database of the former MySQL dump;
- ``default_store``: the store configured in ``cornerstone/configs/data_config.json``.

Command line: ``python -m cornerstone.data --help``.
"""
from .bootstrap import bootstrap_from_snapshot, bootstrap_from_sqlite, import_legacy_references, import_sharp_from_sqlite
from .config import default_store, load_config
from .store import DataStore
from .update import SOURCES, update_all

__all__ = ["DataStore", "bootstrap_from_snapshot", "bootstrap_from_sqlite", "import_sharp_from_sqlite", "import_legacy_references",
           "default_store", "load_config", "update_all", "SOURCES"]
