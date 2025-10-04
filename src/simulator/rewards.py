from __future__ import annotations
from typing import Dict, Any
import numpy as np

def pnl_only(pnl: float, info: Dict[str, Any]) -> float:
    """Reward = raw PnL per step (includes txn costs)."""
    return float(pnl)

def pnl_minus_var(pnl: float, info: Dict[str, Any], lam: float = 0.1, window: int = 50) -> float:
    """
    Risk-aware reward: pnl - λ * rolling variance of pnl.
    You pass lam/window by currying or via a closure when wiring the reward_fn.
    """
    # Expect the caller to maintain a rolling buffer in info if desired.
    # For a stateless default, we just penalize absolute pnl as a proxy:
    return float(pnl - lam * (abs(pnl)))
