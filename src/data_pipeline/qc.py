from __future__ import annotations
import pandas as pd

def coverage(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for c in df.columns:
        s = df[c]
        idx = s.dropna().index
        out[c] = {
            "first": idx.min(),
            "last": idx.max(),
            "non_na_ratio": float(s.notna().mean()),
            "rows": int(s.size),
        }
    return pd.DataFrame(out).T

def assert_min_coverage(df: pd.DataFrame, cols: list[str], min_ratio: float = 0.5):
    poor = {c: df[c].notna().mean() for c in cols if c in df and df[c].notna().mean() < min_ratio}
    if poor:
        raise AssertionError(f"Low coverage: " + ", ".join(f"{k}={v:.2%}" for k,v in poor.items()))
