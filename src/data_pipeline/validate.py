from __future__ import annotations
import pandas as pd

def assert_no_na(df_or_s: pd.DataFrame | pd.Series, cols: list[str] | None = None) -> None:
    obj = df_or_s if cols is None else df_or_s[cols]
    if obj.isna().any().any():
        raise AssertionError("Found NaNs.")

def assert_monotonic_index(df_or_s: pd.DataFrame | pd.Series) -> None:
    idx = df_or_s.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise AssertionError("Index is not DatetimeIndex.")
    if not idx.is_monotonic_increasing:
        raise AssertionError("Index not monotonic increasing.")
