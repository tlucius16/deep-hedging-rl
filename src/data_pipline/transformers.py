from __future__ import annotations
import pandas as pd

def standardize_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure datetime index named 'date' and float values named 'value'."""
    out = df.copy()
    if "date" not in out.columns or "value" not in out.columns:
        raise ValueError("Input must have columns ['date','value']")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date")
    # coerce numeric; invalids -> NaN, then drop
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value"]).set_index("date")
    out.index.name = "date"
    return out

def resample_fill(s: pd.Series, freq: str = "D", method: str = "ffill") -> pd.Series:
    """Resample to target frequency and fill (default forward-fill)."""
    s = s.asfreq(freq)
    if method == "ffill":
        s = s.ffill()
    elif method == "bfill":
        s = s.bfill()
    return s

def compute_returns(price: pd.Series, kind: str = "log") -> pd.Series:
    """Compute simple or log returns for a price series."""
    if kind == "log":
        return (price.ffill().pct_change() + 1.0).apply(lambda x: 0.0 if x <= 0 else pd.NA).fillna(0).pipe(lambda s: (price.ffill()/price.ffill().shift(1)).apply(lambda x: 0 if x<=0 else pd.np.log(x)))
    if kind == "simple":
        return price.ffill().pct_change()
    raise ValueError("kind must be 'log' or 'simple'")

def rename_cols(s: pd.Series, new_name: str) -> pd.Series:
    s = s.copy()
    s.name = new_name
    return s
