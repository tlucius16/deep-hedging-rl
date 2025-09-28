from __future__ import annotations
from pathlib import Path
import pandas as pd
from .config import RAW_DIR, DATASETS
from .io_utils import read_csv_flexible
from .transformers import standardize_timeseries, resample_fill, rename_cols

def load_timeseries_csv(
    file: str,
    date_candidates: list[str],
    value_candidates: list[str],
    freq: str = "D",
    rename_to: str | None = None,
) -> pd.Series:
    raw = read_csv_flexible(
        path=Path(RAW_DIR) / file,
        date_cols=date_candidates,
        value_cols=value_candidates,
    )
    std = standardize_timeseries(raw)  # index=date, col=value
    ser = std["value"]
    ser = resample_fill(ser, freq=freq, method="ffill")
    if rename_to:
        ser = rename_cols(ser, rename_to)
    return ser

def load_dataset(name: str) -> pd.Series:
    """Load by logical dataset name defined in config.DATASETS."""
    cfg = DATASETS[name]
    return load_timeseries_csv(
        file=cfg["file"],
        date_candidates=cfg["date_candidates"],
        value_candidates=cfg["value_candidates"],
        freq=cfg.get("freq", "D"),
        rename_to=cfg["rename"]["value"],
    )

from .io_utils import read_csv_flexible, pick_first, coerce_numeric

def _read_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize column names to keep downstream simpler (still keep originals)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def load_hvol() -> pd.Series:
    from .config import DATASETS, RAW_DIR
    cfg = DATASETS["hvol"]
    ser = load_timeseries_csv(
        file=cfg["file"],
        date_candidates=cfg["date_candidates"],
        value_candidates=cfg["value_candidates"],
        freq=cfg["freq"],
        rename_to=cfg["rename"]["value"],
    )
    return ser

def load_forward_prices() -> pd.Series:
    from .config import DATASETS, RAW_DIR
    cfg = DATASETS["fwd"]
    ser = load_timeseries_csv(
        file=cfg["file"],
        date_candidates=cfg["date_candidates"],
        value_candidates=cfg["value_candidates"],
        freq=cfg["freq"],
        rename_to=cfg["rename"]["value"],
    )
    return ser

def load_option_prices() -> pd.DataFrame:
    """Return tidy options prices with canonical columns."""
    from .config import DATASETS, RAW_DIR
    cfg = DATASETS["op_prices"]
    df = _read_df(Path(RAW_DIR) / cfg["file"])
    c = cfg["cands"]

    # pick columns (some may be None, we'll handle)
    date   = pick_first(df, c["date"])
    und    = pick_first(df, c["underlying"])
    cp     = pick_first(df, c["put_call"])
    exp    = pick_first(df, c["expiry"])
    strike = pick_first(df, c["strike"])
    bid    = pick_first(df, c["bid"]);    ask   = pick_first(df, c["ask"])
    mid    = pick_first(df, c["mid"]);    last  = pick_first(df, c["last"])
    iv     = pick_first(df, c["iv"]);     delta = pick_first(df, c["delta"])
    gamma  = pick_first(df, c["gamma"]);  vega  = pick_first(df, c["vega"])
    theta  = pick_first(df, c["theta"])
    oi     = pick_first(df, c["open_interest"])

    keep = [date, und, cp, exp, strike, bid, ask, mid, last, iv, delta, gamma, vega, theta, oi]
    keep = [k for k in keep if k is not None]
    out = df[keep].copy()

    # rename to canonical
    rename_map = {
        date: "date", und: "underlying", cp: "put_call", exp: "expiry",
        strike: "strike", bid: "bid", ask: "ask", mid: "mid", last: "last",
        iv: "iv", delta: "delta", gamma: "gamma", vega: "vega",
        theta: "theta", oi: "open_interest",
    }
    out = out.rename(columns={k:v for k,v in rename_map.items() if k is not None})

    # typing
    out["date"]   = pd.to_datetime(out["date"], errors="coerce")
    if "expiry" in out: out["expiry"] = pd.to_datetime(out["expiry"], errors="coerce")
    for col in ["strike","bid","ask","mid","last","iv","delta","gamma","vega","theta","open_interest"]:
        if col in out: out[col] = coerce_numeric(out[col])

    # compute mid if missing
    if "mid" not in out and {"bid","ask"}.issubset(out.columns):
        out["mid"] = (out["bid"] + out["ask"]) / 2.0

    # normalize put/call values to "C"/"P"
    if "put_call" in out:
        out["put_call"] = out["put_call"].astype(str).str.upper().str[0].replace({"C":"C","P":"P"})

    return out.dropna(subset=["date","strike","mid"], how="any")

def load_option_volume() -> pd.DataFrame:
    """Return tidy daily option volumes & OI."""
    from .config import DATASETS, RAW_DIR
    cfg = DATASETS["op_volume"]
    df = _read_df(Path(RAW_DIR) / cfg["file"])
    c = cfg["cands"]

    cols = {k: pick_first(df, v) for k, v in c.items()}
    out = df[[v for v in cols.values() if v is not None]].rename(columns={v:k for k,v in cols.items() if v})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in ["strike","volume","open_interest"]:
        if col in out: out[col] = coerce_numeric(out[col])
    return out.dropna(subset=["date","strike"], how="any")

def load_vol_surface() -> pd.DataFrame:
    """Return tidy vol surface: date, underlying, moneyness, tenor_d, iv, [expiry]."""
    from .config import DATASETS, RAW_DIR
    cfg = DATASETS["vol_surface"]
    df = _read_df(Path(RAW_DIR) / cfg["file"])
    c = cfg["cands"]

    cols = {k: pick_first(df, v) for k, v in c.items()}
    out = df[[v for v in cols.values() if v is not None]].rename(columns={v:k for k,v in cols.items() if v})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if "expiry" in out: out["expiry"] = pd.to_datetime(out["expiry"], errors="coerce")
    for col in ["moneyness","tenor_d","iv"]:
        if col in out: out[col] = coerce_numeric(out[col])
    return out.dropna(subset=["date","iv"], how="any")
