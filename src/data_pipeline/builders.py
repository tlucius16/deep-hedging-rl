from __future__ import annotations
from pathlib import Path
import pandas as pd
from .config import PROCESSED_DIR
from .loaders import load_dataset
from .validate import assert_no_na, assert_monotonic_index
from .io_utils import save_parquet, save_csv

def build_market_daily(save: bool = True) -> pd.DataFrame:
    """
    Build a daily market panel with SPY, GSPC, VIX, DGS10.
    Extend by adding more series in config.DATASETS and then joining here.
    """
    spy  = load_dataset("spy")
    gspc = load_dataset("gspc")
    vix  = load_dataset("vix")
    dgs10 = load_dataset("dgs10")

    df = pd.concat([spy, gspc, vix, dgs10], axis=1)
    df = df.dropna(how="all")  # allow some NaNs; most series are daily

    assert_monotonic_index(df)
    # Don’t force drop of NaNs globally—financial series sometimes missing on holidays.
    # But we can ensure major columns are mostly filled:
    assert df["close_spy"].notna().mean() > 0.95, "Too many NaNs in SPY"

    if save:
        save_parquet(df, Path(PROCESSED_DIR) / "market_daily.parquet")
        save_csv(df, Path(PROCESSED_DIR) / "market_daily.csv")
    return df

from .loaders import load_option_prices, load_option_volume, load_vol_surface, load_hvol, load_forward_prices

def build_options_snapshot(save: bool = True) -> pd.DataFrame:
    """Join option prices and volume/OI on common keys (date, expiry, strike, put_call)."""
    px = load_option_prices()
    vol = load_option_volume()
    key = ["date","underlying","put_call","expiry","strike"]
    df = px.merge(vol, on=[k for k in key if k in vol.columns], how="left")
    if save:
        save_parquet(df, PROCESSED_DIR / "options_snapshot.parquet")
        save_csv(df, PROCESSED_DIR / "options_snapshot.csv")
    return df

def build_vol_surface_long(save: bool = True) -> pd.DataFrame:
    """Clean vol surface for plotting/calibration."""
    vs = load_vol_surface()
    if save:
        save_parquet(vs, PROCESSED_DIR / "vol_surface.parquet")
        save_csv(vs, PROCESSED_DIR / "vol_surface.csv")
    return vs

def build_market_plus_hvol_fwd(save: bool = True) -> pd.DataFrame:
    """Market panel extended with historical vol and forwards."""
    mkt = build_market_daily(save=False)
    h  = load_hvol()
    f  = load_forward_prices()
    df = mkt.join(h, how="left").join(f, how="left")
    if save:
        save_parquet(df, PROCESSED_DIR / "market_extended.parquet")
        save_csv(df, PROCESSED_DIR / "market_extended.csv")
    return df
