"""CORNERSTONE datasets: SHARP time windows of the labeled (t_X, HARP) pairs, from the Parquet store.

Replaces ``Cornerstone.create_hdf5_from_db`` (no database); see ``build.build_dataset``.
"""
from .build import build_dataset, label_name

__all__ = ["build_dataset", "label_name"]
