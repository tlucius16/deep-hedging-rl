from __future__ import annotations


import pandas as pd
from pathlib import Path
from typing import Iterable


def _find_date_col(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    cols_map = {c.lower(): c for c in df.columns}
    # 1) try explicit candidates
    for c in candidates:
        if c.lower() in cols_map:
            return cols_map[c.lower()]
    # 2) try unnamed first column
    if df.columns[0] in ("", None) or str(df.columns[0]).startswith("Unnamed"):
        try:
            pd.to_datetime(df.iloc[:, 0], errors="raise")
            return df.columns[0]
        except Exception:
            pass
    # 3) try any column that parses mostly as datetime
    for c in df.columns:
        s = pd.to_datetime(df[c], errors="coerce")
        if s.notna().mean() > 0.9:
            return c
    # 4) try index
    if not isinstance(df.index, pd.DatetimeIndex):
        # sometimes date is already the index but not parsed
        idx = pd.to_datetime(df.index, errors="coerce")
        if idx.notna().mean() > 0.9:
            df.reset_index(inplace=True)
            df.rename(columns={"index": "date"}, inplace=True)
            return "date"
    raise KeyError(f"No date-like column found. Columns: {list(df.columns)}")

def _find_value_col(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    cols_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols_map:
            return cols_map[c.lower()]
    # fallback: prefer common price columns if present
    for c in ["Adj Close", "Close", "PX_LAST", "Price"]:
        if c in df.columns:
            return c
    # else pick the first numeric-looking column
    numcols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numcols:
        return numcols[0]
    raise KeyError(f"No candidate value column found. Columns: {list(df.columns)}")

def read_csv_flexible(
    path: Path,
    date_cols: Iterable[str],
    value_cols: Iterable[str],
    parse_first: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    dcol = _find_date_col(df, date_cols)
    vcol = _find_value_col(df, value_cols)
    out = df[[dcol, vcol]].rename(columns={dcol: "date", vcol: "value"})
    if parse_first:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out
