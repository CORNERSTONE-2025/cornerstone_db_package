"""CORNERSTONE processing: from the raw store to the inputs of the labeling.

- ``QualityFilter``: which SHARP records are used ("good data"), as parameters;
- ``harp_spans`` / ``harp_regions``: observation spans and NOAA regions of the HARPs;
- ``FlareConfig`` / ``associate``: GOES events calibrated, one satellite per flare, associated
  to HARPs (many-to-many HARP <-> NOAA, ambiguity rules).

Independent of the ML stack; parameters are dataclasses with ``to_dict()`` for the metadata.
"""
from .flares import AMBIGUOUS_RULES, FlareConfig, associate, calibrate, flux_to_class, prepare_events
from .harps import harp_regions, harp_spans
from .quality import QualityFilter

__all__ = ["QualityFilter", "harp_spans", "harp_regions", "FlareConfig", "AMBIGUOUS_RULES", "associate",
           "prepare_events", "calibrate", "flux_to_class"]
