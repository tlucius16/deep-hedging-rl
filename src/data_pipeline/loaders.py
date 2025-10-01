# loaders.py
from __future__ import annotations
from pathlib import Path
import io, zipfile
import pandas as pd

from .config import (
    RAW_DIR, DATASETS, FILENAMES,
    TIMESERIES_SOURCES, TIMESERIES_PARSING,
)
from .io_utils import (
    read_csv_flexible, pick_first, coerce_numeric, ensure_materialized
)
from .transformers import (
    standardize_timeseries, resample_fill, rename_cols,
    standardize_option_table, standardize_vol_surface_table,
)

# ---------- File readers ----------

def _read_any(path: Path) -> pd.DataFrame:
    """Read CSV/Parquet directly, or first CSV/Parquet member from a .zip."""
    ensure_materialized(path)
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            members = [n for n in zf.namelist()
                       if n.lower().endswith((".csv", ".parquet"))]
            if not members:
                raise ValueError(f"No CSV/Parquet inside {path.name}. Members: {zf.namelist()}")
            name = members[0]
            with zf.open(name) as fh:
                if name.lower().endswith(".csv"):
                    return pd.read_csv(fh)
                return pd.read_parquet(io.BytesIO(fh.read()))
    raise ValueError(f"Unsupported file extension: {path}")


# ---------- Flexible time-series loaders ----------

def load_timeseries_csv(
    file: str,
    date_candidates,
    value_candidates,
    freq: str = "D",
    rename_to: str | None = None,
    skiprows: int = 0,
) -> pd.Series:
    p = Path(RAW_DIR) / file
    ensure_materialized(p)

    # Use our flexible CSV reader that reduces to ['date','value']
    raw = read_csv_flexible(
        path=p,
        date_cols=date_candidates,
        value_cols=value_candidates,
        skiprows=skiprows,
    )
    std = standardize_timeseries(raw)
    s = std["value"]
    s = resample_fill(s, freq=freq, method="ffill")
    if rename_to:
        s = rename_cols(s, rename_to)
    return s


def load_dataset(name: str) -> pd.Series:
    cfg = DATASETS[name]
    return load_timeseries_csv(
        file=cfg["file"],
        date_candidates=cfg["date_candidates"],
        value_candidates=cfg["value_candidates"],
        freq=cfg.get("freq", "D"),
        rename_to=cfg["rename"]["value"],
        skiprows=cfg.get("skiprows", 0),
    )

def load_timeseries_flexible(name: str) -> pd.Series | None:
    """
    Try multiple candidate files for `name` (SPY/GSPC/VIX/DGS10).
    Returns standardized Series or None (with warnings) if none work.
    """
    srcs = TIMESERIES_SOURCES.get(name, [])
    parse = TIMESERIES_PARSING[name]

    for fname in srcs:
        p = Path(RAW_DIR) / fname
        if not p.exists():
            continue
        try:
            # (We call _read_any only to validate readability; actual parsing
            # uses read_csv_flexible which honors skiprows and candidates.)
            _ = _read_any(p)
            df_two = read_csv_flexible(
                path=p,
                date_cols=parse["date_candidates"],
                value_cols=parse["value_candidates"],
                skiprows=parse.get("skiprows", 0),
            )
            std = standardize_timeseries(df_two)
            s = std["value"]
            s = resample_fill(s, freq=parse.get("freq", "D"), method="ffill")
            s = rename_cols(s, parse["rename_to"])
            return s
        except Exception as e:
            print(f"⚠️ {name}: failed reading {fname}: {e}")
            continue

    print(f"⚠️ {name}: no usable source found in {srcs}")
    return None

# ---------- Helpers for WRDS table standardization ----------

def _rename_by_candidates(df: pd.DataFrame, mapping: dict[str, list[str]]) -> pd.DataFrame:
    cols = {str(c).strip(): c for c in df.columns}
    out = df.copy()
    for canon, cands in mapping.items():
        for cand in cands:
            if cand in cols:
                out = out.rename(columns={cols[cand]: canon})
                break
    return out


# ---------- Options/volume/vol-surface loaders (WRDS, possibly zipped) ----------

def load_standardized_options(ticker: str) -> pd.DataFrame:
    key  = f"std_opts_{ticker.lower()}"
    path = Path(RAW_DIR) / FILENAMES[key]
    df   = _read_any(path)
    df.columns = [str(c).strip() for c in df.columns]

    # Matches your StandardizedOptionsSPX.zip
    cand_map = {
        "date":        ["date","Date","DATE"],
        "underlying":  ["ticker","underlying","root","symbol","TICKER"],
        "put_call":    ["cp_flag","put_call","CALL_PUT","OptionType"],
        # no explicit expiry; we’ll build it from days
        "DTE":         ["days","days_to_expiration","DTE"],
        "strike":      ["strike_price","strike","STRIKE","K"],
        # you only have 'premium' (no bid/ask), treat as last/mid
        "last":        ["premium","PX_LAST","last","Last","Premium"],
        "mid":         ["mid","PX_MID","Mid","midpoint","MIDPOINT"],
        "iv":          ["impl_volatility","iv","IV","ImpliedVol","IVOL"],
        "delta":       ["delta","DELTA"],
        "gamma":       ["gamma","GAMMA"],
        "theta":       ["theta","THETA"],
        "vega":        ["vega","VEGA"],
        "open_interest":["open_interest","OPEN_INT","OI"],
        "volume":      ["volume","VOLUME"],
        # sometimes present
        "forward_price":["forward_price","ForwardPrice"],
    }

    df = _rename_by_candidates(df, cand_map)

    # Build expiry from date + days if missing
    if "expiry" not in df.columns and "DTE" in df.columns and "date" in df.columns:
        df["expiry"] = pd.to_datetime(df["date"], errors="coerce") + pd.to_timedelta(
            pd.to_numeric(df["DTE"], errors="coerce"), unit="D"
        )

    # If no mid, synthesize from bid/ask OR use last/premium
    if "mid" not in df.columns:
        if {"bid","ask"}.issubset(df.columns):
            df["mid"] = (pd.to_numeric(df["bid"], errors="coerce") + pd.to_numeric(df["ask"], errors="coerce"))/2
        elif "last" in df.columns:
            df["mid"] = pd.to_numeric(df["last"], errors="coerce")

    from .transformers import standardize_option_table
    return standardize_option_table(df)

def load_option_volume_wrds(ticker: str) -> pd.DataFrame:
    key  = f"opt_volume_{ticker.lower()}"
    path = Path(RAW_DIR) / FILENAMES[key]
    df   = _read_any(path)
    df.columns = [str(c).strip() for c in df.columns]

    cand_map = {
        "date":        ["date","Date","DATE"],
        "underlying":  ["ticker","underlying","root","symbol","TICKER"],
        "put_call":    ["cp_flag","put_call","CALL_PUT","OptionType"],
        "volume":      ["volume","VOLUME"],
        "open_interest":["open_interest","OPEN_INT","OI"],
        # if a variant ever has days:
        "DTE":         ["days","days_to_expiration","DTE"],
    }
    df = _rename_by_candidates(df, cand_map)

    if "expiry" not in df.columns and {"DTE","date"}.issubset(df.columns):
        df["expiry"] = pd.to_datetime(df["date"], errors="coerce") + pd.to_timedelta(
            pd.to_numeric(df["DTE"], errors="coerce"), unit="D"
        )

    from .transformers import standardize_option_table
    # Use the same cleaner to coerce types; it will ignore absent cols.
    return standardize_option_table(df)

def load_vol_surface_wrds(ticker: str) -> pd.DataFrame:
    key  = f"vol_surface_{ticker.lower()}"
    path = Path(RAW_DIR) / FILENAMES[key]
    df   = _read_any(path)
    df.columns = [str(c).strip() for c in df.columns]

    cand_map = {
        "date":       ["date","Date","DATE"],
        "underlying": ["ticker","underlying","root","symbol","TICKER"],
        "tenor_d":    ["days","tenor_d","DaysToExp","DTE","days_to_expiration"],
        "moneyness":  ["delta","Delta","DELTA","DeltaBucket"],
        "iv":         ["impl_volatility","iv","IV","ImpliedVol","IVOL"],
        "expiry":     ["expiration","expiry","exdate","EXDATE"],  # if present
    }
    df = _rename_by_candidates(df, cand_map)

    return standardize_vol_surface_table(df)

def load_hist_vol_wrds(ticker: str) -> pd.DataFrame:
    key  = f"hist_vol_{ticker.lower()}"
    path = Path(RAW_DIR) / FILENAMES[key]
    df   = _read_any(path)
    df.columns = [str(c).strip() for c in df.columns]
    # Minimal normalization; users may rename in downstream builders
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

def load_forward_prices_wrds(ticker: str) -> pd.DataFrame:
    key  = f"fwd_{ticker.lower()}"
    path = Path(RAW_DIR) / FILENAMES[key]
    df   = _read_any(path)
    df.columns = [str(c).strip() for c in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df
