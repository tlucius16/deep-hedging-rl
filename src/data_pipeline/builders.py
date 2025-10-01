from __future__ import annotations
import pandas as pd
from pathlib import Path
from .config import PROCESSED_DIR
from .loaders import (
    load_dataset,
    load_standardized_options,
    load_option_volume_wrds,
    load_vol_surface_wrds,
    load_hist_vol_wrds,
    load_forward_prices_wrds,
    load_timeseries_flexible,
)
from .io_utils import save_parquet, save_csv

# builders.py (adjust build_market_daily to use flexible loader)

def build_market_daily(save: bool = True) -> pd.DataFrame:
    spy   = load_timeseries_flexible("spy")
    gspc  = load_timeseries_flexible("gspc")
    vix   = load_timeseries_flexible("vix")
    dgs10 = load_timeseries_flexible("dgs10")

    parts = [x for x in [spy, gspc, vix, dgs10] if x is not None]
    if not parts:
        raise RuntimeError("No market series could be loaded. Check LFS and raw files.")

    df = pd.concat(parts, axis=1).dropna(how="all").sort_index()

    # Soft checks with warnings instead of hard asserts:
    if "close_spy" in df and df["close_spy"].notna().mean() <= 0.90:
        print("⚠️ SPY has many NaNs — verify source file/columns. Proceeding anyway.")
    if "close_spy" not in df and "close_gspc" not in df:
        print("⚠️ Neither SPY nor GSPC loaded — panel will lack an equity price series.")

    if save:
        save_parquet(df, Path(PROCESSED_DIR) / "market_daily.parquet")
        save_csv(df, Path(PROCESSED_DIR) / "market_daily.csv")
    return df

def build_options_snapshot(ticker: str = "spy", save: bool = True) -> pd.DataFrame:
    px  = load_standardized_options(ticker)        # from StandardizedOptions*.zip
    vol = load_option_volume_wrds(ticker)          # from OptionVolume*.zip
    key = [c for c in ["date","underlying","put_call","expiry","strike"] if c in px.columns]
    df = px.merge(vol, on=[k for k in key if k in vol.columns], how="left")
    if save:
        save_parquet(df, PROCESSED_DIR / f"options_snapshot_{ticker}.parquet")
        save_csv(df, PROCESSED_DIR / f"options_snapshot_{ticker}.csv")
    return df

def build_vol_surface_long(ticker: str = "spy", save: bool = True) -> pd.DataFrame:
    vs = load_vol_surface_wrds(ticker)
    if save:
        save_parquet(vs, PROCESSED_DIR / f"vol_surface_{ticker}.parquet")
        save_csv(vs, PROCESSED_DIR / f"vol_surface_{ticker}.csv")
    return vs

def build_market_plus_hvol_fwd(ticker: str = "spy", save: bool = True) -> pd.DataFrame:
    mkt = build_market_daily(save=False)

    # ---------- Historical vol: keep only date/days/volatility and pivot ----------
    h = load_hist_vol_wrds(ticker)
    if h is None or len(h) == 0:
        H = pd.DataFrame(index=mkt.index)
    else:
        # keep minimal
        keep_h = [c for c in ["date", "days", "volatility"] if c in h.columns]
        h = h[keep_h].copy()
        if "date" in h: h["date"] = pd.to_datetime(h["date"], errors="coerce")
        # pivot to hvol_{Xd}
        if {"date", "days", "volatility"}.issubset(h.columns):
            h["days"] = pd.to_numeric(h["days"], errors="coerce")
            H = (
                h.dropna(subset=["date", "days", "volatility"])
                 .pivot_table(index="date", columns="days", values="volatility", aggfunc="mean")
            )
            # rename columns to hvol_{Xd}
            H.columns = [f"hvol_{int(d)}d" for d in H.columns]
        elif {"date", "volatility"}.issubset(h.columns):
            H = (h.dropna(subset=["date", "volatility"])
                   .set_index("date")
                   .rename(columns={"volatility": "hvol"}))
        else:
            H = pd.DataFrame(index=mkt.index)

    # ---------- Forward price: pick front (min positive tenor) and call it fwd_front ----------
    f = load_forward_prices_wrds(ticker)
    if f is None or len(f) == 0:
        F = pd.DataFrame(index=mkt.index)
    else:
        keep_f = [c for c in ["date", "expiry", "fwd_price", "ForwardPrice"] if c in f.columns]
        f = f[keep_f].copy()
        if "ForwardPrice" in f.columns and "fwd_price" not in f.columns:
            f = f.rename(columns={"ForwardPrice": "fwd_price"})
        if "date" in f:   f["date"] = pd.to_datetime(f["date"], errors="coerce")
        if "expiry" in f: f["expiry"] = pd.to_datetime(f["expiry"], errors="coerce")

        if {"date", "expiry", "fwd_price"}.issubset(f.columns):
            f["days"] = (f["expiry"] - f["date"]).dt.days
            f = f.loc[f["days"].ge(0)]
            F = (
                f.sort_values(["date", "days"])
                 .groupby("date", as_index=False).first()[["date", "fwd_price"]]
                 .set_index("date")
                 .rename(columns={"fwd_price": "fwd_front"})
            )
        elif {"date", "fwd_price"}.issubset(f.columns):
            F = f.dropna(subset=["date"]).set_index("date").rename(columns={"fwd_price": "fwd_front"})
        else:
            F = pd.DataFrame(index=mkt.index)

    out = mkt.join(H, how="left").join(F, how="left")

    if save:
        save_parquet(out, Path(PROCESSED_DIR) / f"market_extended_{ticker}.parquet")
        save_csv(out, Path(PROCESSED_DIR) / f"market_extended_{ticker}.csv")
    return out


