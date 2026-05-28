from __future__ import annotations

__all__ = ["dt", "mt", "pd", "lt"]


def __getattr__(name: str):
    if name == "dt":
        from nico2_lib import datasets as dt
        return dt
    if name == "lt":
        from nico2_lib import label_transfer as lt
        return lt
    if name == "mt":
        from nico2_lib import metrics as mt
        return mt
    if name == "pd":
        from nico2_lib import predictors as pd
        return pd
    raise AttributeError(f"module 'nico2_lib' has no attribute {name!r}")
