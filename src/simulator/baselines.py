# baselines.py
from __future__ import annotations
import numpy as np
from typing import Callable

def no_hedge_policy() -> Callable[[np.ndarray], float]:
    """Always zero position."""
    return lambda obs: 0.0

def momentum_policy(feature_idx: int = 0, k: float = 1.0) -> Callable[[np.ndarray], float]:
    """
    Uses sign of last-step log return of the chosen feature.
    Pure function of observation window -> action.
    """
    def policy(obs: np.ndarray) -> float:
        x = obs[:, feature_idx]
        if len(x) < 2:
            return 0.0
        r = np.diff(np.log(np.clip(x, 1e-12, None)))
        sig = np.sign(r[-1])
        return float(np.clip(k * sig, -1.0, 1.0))
    return policy

def volatility_targeting(feature_idx: int = 0, ann_vol_target: float = 0.15) -> Callable[[np.ndarray], float]:
    """
    Hedge size inversely proportional to recent realized volatility of the selected feature.
    - If realized vol > target → increase hedge (reduce exposure).
    - If realized vol < target → decrease hedge (take more exposure).
    """
    def policy(obs: np.ndarray) -> float:
        x = obs[:, feature_idx]
        if len(x) < 5:  # at least a few obs for stdev
            return 0.0
        r = np.diff(np.log(np.clip(x, 1e-12, None)))
        vol = np.nanstd(r, ddof=1) * np.sqrt(252.0)
        if vol <= 1e-6 or np.isnan(vol):
            return 0.0

        # Hedge fraction = 1 - (target_vol / realized_vol)
        # → if vol > target → hedge up (approach 1)
        # → if vol < target → hedge down (approach 0)
        h = 1.0 - (ann_vol_target / vol)
        return float(np.clip(h, -1.0, 1.0))  # hedge ratio bounded
    return policy


def delta_hedge_policy(delta_fn: Callable[[np.ndarray], float], scale: float = 1.0) -> Callable[[np.ndarray], float]:
    """
    Wrap a delta estimator (from obs window) into a policy that sets hedge = -scale * delta.
    """
    def policy(obs: np.ndarray) -> float:
        try:
            d = float(delta_fn(obs))
        except Exception:
            d = 0.0
        return float(np.clip(-scale * d, -1.0, 1.0))
    return policy
