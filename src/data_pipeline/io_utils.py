from __future__ import annotations


# io_utils.py

import pandas as pd
from pathlib import Path
from typing import Iterable

# ---------- helpers to detect columns ----------

def pick_first(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Return the first existing column matching any of the candidate names (case-insensitive)."""
    cmap = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).lower()
        if key in cmap:
            return cmap[key]
    return None

def coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _find_date_col(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    # 1) explicit candidates
    col = pick_first(df, candidates)
    if col:
        return col
    # 2) unnamed/first column
    first = df.columns[0]
    if str(first).startswith("Unnamed"):
        try:
            pd.to_datetime(df.iloc[:, 0], errors="raise")
            return first
        except Exception:
            pass
    # 3) any column that mostly parses as dates
    for c in df.columns:
        if pd.to_datetime(df[c], errors="coerce").notna().mean() > 0.9:
            return c
    # 4) index might be the date
    if isinstance(df.index, pd.DatetimeIndex):
        df.reset_index(inplace=True)
        df.rename(columns={"index": "date"}, inplace=True)
        return "date"
    raise KeyError(f"No date-like column found. Columns: {list(df.columns)}")

def _find_value_col(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    col = pick_first(df, candidates)
    if col:
        return col
    for fallback in ("Adj Close", "Close", "PX_LAST", "Price", "VALUE"):
        if fallback in df.columns:
            return fallback
    numcols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numcols:
        return numcols[0]
    raise KeyError(f"No candidate value column found. Columns: {list(df.columns)}")

# ---------- file IO ----------

def read_csv_flexible(
    path: Path,
    date_cols: Iterable[str],
    value_cols: Iterable[str],
    parse_first: bool = True,
    skiprows: int = 0,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    dcol = _find_date_col(df, date_cols)
    vcol = _find_value_col(df, value_cols)
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

# io_utils.py (append these)
LFS_HEADER = "version https://git-lfs.github.com/spec/v1"

def is_lfs_pointer(path: Path, sniff: int = 160) -> bool:
    try:
        if path.stat().st_size > 512:  # real files will be larger
            return False
        head = path.read_text(errors="ignore")[:sniff]
        return LFS_HEADER in head
    except Exception:
        return False

def ensure_materialized(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    if is_lfs_pointer(path):
        raise RuntimeError(
            f"{path.name} looks like a Git LFS pointer.\n"
            "Run:\n"
            "  git lfs install\n"
            "  git lfs pull\n"
            "…then re-run."
        )
