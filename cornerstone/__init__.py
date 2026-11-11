# cornerstone/__init__.py
"""CORNERSTONE: solar flare forecasting from SDO/HMI SHARP data.

``Cornerstone`` (database, labeling, datasets, training) is imported lazily: the data layer
(``cornerstone.data``) and the labeling functions do not need torch, ClearML or the ML stack,
and importing them must stay light.
"""

__all__ = ["Cornerstone"]


def __getattr__(name):
    if name == "Cornerstone":
        from .cornerstone import Cornerstone
        return Cornerstone
    raise AttributeError(f"module 'cornerstone' has no attribute {name!r}")
