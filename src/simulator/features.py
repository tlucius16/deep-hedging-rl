# simulator/features.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow.dataset as ds

# ---------- helpers ----------
def _prep_base(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if "put_call" in x:
        x["put_call"] = (x["put_call"].astype(str)
                           .str.strip().str.upper().str[0]
                           .map({"C":"C","P":"P"}))
    if "delta" in x and x["delta"].abs().quantile(0.99) > 2:
        x["delta"] = x["delta"] / 100.0  # auto-rescale if vendor used ±100
    return x

def _pick_atm_tolerant(df: pd.DataFrame, target_dte: int, base_delta=0.50):
    x = _prep_base(df)
    for dte_tol, d_tol in [(15, 0.15), (25, 0.20), (35, 0.25)]:
        z = x[x["tenor_d"].sub(target_dte).abs() <= dte_tol].copy()
        if z.empty: continue
        z["abs_delta"] = z["delta"].abs()
        z = z[z["abs_delta"].sub(base_delta).abs() <= d_tol]
        if z.empty: continue
        z["dte_diff"] = (z["tenor_d"] - target_dte).abs()
        z["atm_diff"] = (z["abs_delta"] - base_delta).abs()
        z["spread"]   = z["ask"] - z["bid"]
        y = (z.sort_values(["date","dte_diff","atm_diff","spread"])
               .drop_duplicates("date"))
        if len(y): return y[["date","iv"]]
    # fallback: nearest-by-delta within widest DTE
    z = x[x["tenor_d"].sub(target_dte).abs() <= 35].copy()
    if z.empty: return pd.DataFrame(columns=["date","iv"])
    z["abs_delta"] = z["delta"].abs()
    z["atm_diff"]  = (z["abs_delta"] - base_delta).abs()
    z["spread"]    = z["ask"] - z["bid"]
    y = (z.sort_values(["date","atm_diff","spread"])
           .drop_duplicates("date"))
    return y[["date","iv"]]

def _pick_25d_wings_by_date(df: pd.DataFrame, target_dte=30):
    x = _prep_base(df); x["abs_delta"] = x["delta"].abs()
    out = None
    for dte_tol, dlt in [(15, 0.05), (25, 0.08), (35, 0.10)]:
        z = x[x["tenor_d"].sub(target_dte).abs() <= dte_tol].copy()
        if z.empty: continue
        z = z[z["abs_delta"].sub(0.25).abs() <= dlt]
        if z.empty: continue
        z["dte_diff"] = (z["tenor_d"] - target_dte).abs()
        z["d_diff"]   = (z["abs_delta"] - 0.25).abs()
        z["spread"]   = z["ask"] - z["bid"]
        best = (z.sort_values(["date","put_call","dte_diff","d_diff","spread"])
                 .groupby(["date","put_call"], as_index=False)
                 .first())
        wide = (best.pivot(index="date", columns="put_call", values="iv")
                     .rename(columns={"P":"iv_put25_30d", "C":"iv_call25_30d"}))
        out = wide if out is None else out.combine_first(wide)

    if out is None:
        z = x[x["tenor_d"].sub(target_dte).abs() <= 35].copy()
        if z.empty:
            return pd.DataFrame(columns=["date","iv_put25_30d","iv_call25_30d","iv_skew_30d"])
        z["d_diff"] = (z["abs_delta"] - 0.25).abs()
        z["spread"] = z["ask"] - z["bid"]
        best = (z.sort_values(["date","put_call","d_diff","spread"])
                 .groupby(["date","put_call"], as_index=False)
                 .first())
        out = (best.pivot(index="date", columns="put_call", values="iv")
                   .rename(columns={"P":"iv_put25_30d", "C":"iv_call25_30d"}))

    out = out.reset_index()
    out["iv_skew_30d"] = out["iv_put25_30d"] - out["iv_call25_30d"]
    return out[["date","iv_put25_30d","iv_call25_30d","iv_skew_30d"]]

# ---------- public API ----------
def make_option_features(clean_dir: Path, prefix: str):
    """
    From cleaned parquet parts -> daily features:
      iv_atm_30d_{prefix}, iv_atm_91d_{prefix}, iv_ts_slope_{prefix},
      iv_put25_30d_{prefix}, iv_call25_30d_{prefix}, iv_skew_30d_{prefix}
    """
    dset = ds.dataset(clean_dir, format="parquet")
    cols = ["date","tenor_d","put_call","bid","ask","iv","delta"]
    base = dset.to_table(columns=[c for c in cols if c in dset.schema.names]).to_pandas()

    iv30 = _pick_atm_tolerant(base, 30).rename(columns={"iv": f"iv_atm_30d_{prefix}"})
    iv91 = _pick_atm_tolerant(base, 91).rename(columns={"iv": f"iv_atm_91d_{prefix}"})
    feats = iv30.merge(iv91, on="date", how="outer")
    feats[f"iv_ts_slope_{prefix}"] = feats[f"iv_atm_91d_{prefix}"] - feats[f"iv_atm_30d_{prefix}"]

    wings = _pick_25d_wings_by_date(base, target_dte=30).rename(columns={
        "iv_put25_30d": f"iv_put25_30d_{prefix}",
        "iv_call25_30d": f"iv_call25_30d_{prefix}",
        "iv_skew_30d": f"iv_skew_30d_{prefix}",
    })

    out = (feats.merge(wings, on="date", how="left")
                 .sort_values("date").reset_index(drop=True))
    return out
