# dynamics.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, Sequence

def ensure_ret_fwd(df: pd.DataFrame, price_col: str) -> pd.Series:
    """
    Utility for upstream use (optional): compute forward return from a price column.
    Not used by the env directly (env expects 'ret_fwd' to be present).
    """
    px = pd.to_numeric(df[price_col], errors="coerce")
    r = px.pct_change().shift(-1)
    return r

def basic_dynamics_view(df: pd.DataFrame, features: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    If you want to precompute (X, R) outside the env:
      returns feature matrix X and forward returns R from df[features + 'ret_fwd'].
    """
    req = list(features) + ["ret_fwd"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise KeyError(f"dynamics.basic_dynamics_view: missing {missing}")
    g = df.dropna(subset=["ret_fwd"])
    X = g[features].values.astype(float)
    R = g["ret_fwd"].values.astype(float)
    return X, R
