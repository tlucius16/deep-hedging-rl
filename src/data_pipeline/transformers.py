from __future__ import annotations
import numpy as np
import pandas as pd
from .schemas import Cols, OptCols, SurfCols

# ---------- Time series utilities ----------

def standardize_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure columns ['date','value'] exist, parse date, coerce value to float,
    drop bad rows, de-duplicate by date (keep last), sort and set index.
    """
    out = df.copy()

    if Cols.DATE not in out.columns or Cols.VALUE not in out.columns:
        raise ValueError("Input must have columns ['date','value']")

    # Parse date; keep tz-naive; drop invalids
    out[Cols.DATE] = pd.to_datetime(out[Cols.DATE], errors="coerce")
    out = out.dropna(subset=[Cols.DATE])

    # Coerce numeric; drop invalids
    out[Cols.VALUE] = pd.to_numeric(out[Cols.VALUE], errors="coerce")
    out = out.dropna(subset=[Cols.VALUE])

    # Deduplicate on date (common with FRED, vendor exports)
    out = (
        out.sort_values(Cols.DATE)
           .drop_duplicates(subset=[Cols.DATE], keep="last")
           .set_index(Cols.DATE)
           .sort_index()
    )
    out.index.name = Cols.DATE
    return out


def resample_fill(s: pd.Series, freq: str = "D", method: str = "ffill") -> pd.Series:
    """
    Resample to a regular grid from s.index.min()..s.index.max() (inclusive),
    then fill gaps. More robust than Series.asfreq() alone for irregular inputs.
    """
    s = s.sort_index()
    if s.index.empty:
        return s

    idx = pd.date_range(start=s.index.min(), end=s.index.max(), freq=freq)
    s2 = s.reindex(idx)

    if method == "ffill":
        s2 = s2.ffill()
    elif method == "bfill":
        s2 = s2.bfill()
    elif method is None or method == "":
        pass
    else:
        raise ValueError("method must be 'ffill', 'bfill', or None")

    s2.index.name = s.index.name
    s2.name = s.name
    return s2


def compute_returns(price: pd.Series, kind: str = "log") -> pd.Series:
    """
    Compute simple or log returns. Safely handles non-positive/NaN prices.
    - simple: p_t / p_{t-1} - 1
    - log:    log(p_t) - log(p_{t-1}) with guard for p<=0
    """
    p = price.astype(float).ffill()
    if kind == "simple":
        return p.pct_change()

    if kind == "log":
        # mask non-positive before log
        positive = p > 0
        logp = pd.Series(np.where(positive, np.log(p), np.nan), index=p.index)
        r = logp.diff()
        return r

    raise ValueError("kind must be 'log' or 'simple'")


def rename_cols(s: pd.Series, new_name: str) -> pd.Series:
    s = s.copy()
    s.name = new_name
    return s


# ---------- Table standardizers for WRDS-style inputs ----------

def _coerce_numeric_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def standardize_option_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize an options-like table (prices OR volume aggregate).
    - Parses dates
    - Coerces numerics
    - Builds `mid` from bid/ask or last/premium if missing
    - Drops bad rows, but only for columns that actually exist
    """
    out = df.copy()

    # Dates
    for c in [OptCols.DATE, OptCols.EXPIRY]:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce")

    # Put/Call
    if OptCols.PUT_CALL in out.columns:
        pc = out[OptCols.PUT_CALL].astype(str).str.upper().str[0]
        out[OptCols.PUT_CALL] = pc.where(pc.isin(["C", "P"]))

    # Numerics
    num_cols = [
        OptCols.STRIKE, OptCols.BID, OptCols.ASK, OptCols.MID, OptCols.LAST,
        OptCols.IV, OptCols.DELTA, OptCols.GAMMA, OptCols.VEGA, OptCols.THETA,
        OptCols.OPEN_INT, OptCols.VOLUME,
    ]
    for c in num_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Synthesize MID if missing
    if OptCols.MID not in out.columns:
        if {OptCols.BID, OptCols.ASK}.issubset(out.columns):
            out[OptCols.MID] = (out[OptCols.BID] + out[OptCols.ASK]) / 2.0
        elif OptCols.LAST in out.columns:
            out[OptCols.MID] = out[OptCols.LAST]

    # Keep canonical subset that exists
    keep = [c for c in [
        OptCols.DATE, OptCols.UNDERLYING, OptCols.PUT_CALL, OptCols.EXPIRY,
        OptCols.STRIKE, OptCols.BID, OptCols.ASK, OptCols.MID, OptCols.LAST,
        OptCols.IV, OptCols.DELTA, OptCols.GAMMA, OptCols.VEGA, OptCols.THETA,
        OptCols.OPEN_INT, OptCols.VOLUME
    ] if c in out.columns]
    out = out[keep]

    # Drop bad rows — only for columns that actually exist
    drop_subset = []
    for c in (OptCols.DATE, OptCols.MID, OptCols.STRIKE):
        if c in out.columns:
            drop_subset.append(c)
    if drop_subset:
        out = out.dropna(subset=drop_subset, how="any")

    # Sort using available keys
    sort_keys = [c for c in [OptCols.DATE, OptCols.UNDERLYING, OptCols.EXPIRY, OptCols.STRIKE] if c in out.columns]
    if sort_keys:
        out = out.sort_values(sort_keys)

    return out.reset_index(drop=True)

def standardize_vol_surface_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a vol-surface long table (date, expiry|tenor_d, moneyness/delta bucket, iv).
    """
    out = df.copy()

    for c in [SurfCols.DATE, SurfCols.EXPIRY]:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce")

    out = _coerce_numeric_cols(out, [SurfCols.TENOR_D, SurfCols.MNY, SurfCols.IV])

    # keep minimal canonical set
    keep = [c for c in [SurfCols.DATE, SurfCols.UNDERLYING, SurfCols.EXPIRY, SurfCols.TENOR_D, SurfCols.MNY, SurfCols.IV] if c in out.columns]
    out = out[keep].dropna(subset=[SurfCols.DATE, SurfCols.IV], how="any")
    out = out.sort_values([c for c in [SurfCols.DATE, SurfCols.UNDERLYING, SurfCols.EXPIRY, SurfCols.TENOR_D, SurfCols.MNY] if c in out.columns])
    return out.reset_index(drop=True)
