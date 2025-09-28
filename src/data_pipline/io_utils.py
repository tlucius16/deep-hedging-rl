from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Iterable

def read_csv_flexible(
    path: Path,
    date_cols: Iterable[str],
    value_cols: Iterable[str],
    parse_first: bool = True,
) -> pd.DataFrame:
    """
    Read a CSV with unknown column names (FRED/Yahoo/Bloomberg variants).
    Picks the first matching date column and first matching value column.
    Returns df with columns ['date','value'] (untyped index).
    """
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}  # case-insensitive map

    def pick(cands: Iterable[str]) -> str:
        for c in cands:
            if c.lower() in cols:
                return cols[c.lower()]
        raise KeyError(f"No candidate columns {list(cands)} found in {list(df.columns)}")

    dcol = pick(date_cols)
    vcol = pick(value_cols)
    out = df[[dcol, vcol]].rename(columns={dcol: "date", vcol: "value"})
    if parse_first:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out

def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)

def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)

def pick_first(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None

def coerce_numeric(s):
    return pd.to_numeric(s, errors="coerce")
