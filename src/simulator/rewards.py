# rewards.py
from __future__ import annotations
import numpy as np
from typing import Dict

def pnl_only(pnl: float, info: Dict) -> float:
    """Raw per-step P&L."""
    return float(pnl)

def log_utility(pnl: float, info: Dict) -> float:
    """
    Δ log(NAV) as reward. Requires 'nav' and 'pnl' in info.
    """
    nav = float(info["nav"])
    prev = max(1e-12, nav / (1.0 + float(info["pnl"])))
    return float(np.log(max(nav, 1e-12)) - np.log(prev))

def mean_variance(pnl: float, info: Dict, lam: float = 5.0) -> float:
    """Per-step mean-variance proxy."""
    return float(pnl - lam * (pnl ** 2))

def downside_focus(pnl: float, info: Dict, kappa: float = 5.0) -> float:
    """Penalize losses more than gains."""
    return float(pnl if pnl >= 0 else pnl * (1.0 + kappa))
