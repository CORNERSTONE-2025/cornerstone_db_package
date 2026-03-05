# utils/config_tools.py
from collections.abc import Mapping

def deep_update(base, upd):
    """Recursively update dict `base` in place with dict `upd`."""
    if not isinstance(upd, Mapping):
        return base
    for k, v in upd.items():
        if isinstance(v, Mapping) and isinstance(base.get(k), Mapping):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base
















