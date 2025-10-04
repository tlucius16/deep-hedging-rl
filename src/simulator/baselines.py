from __future__ import annotations
from typing import Callable
import numpy as np
import pandas as pd

# policy signatures: (obs: np.ndarray) -> float

def no_hedge_policy() -> Callable[[np.ndarray], float]:
    """Always 0 position."""
    return lambda obs: 0.0

def momentum_policy(returns: np.ndarray, k: float = 1.0) -> Callable[[np.ndarray], float]:
    """
    Hedge in the direction of last return (sign), scaled by k units.
    `returns` should be the array of hedge instrument pct returns aligned with env time.
    """
    state = {"t": None}  # we will set t from external env each call if provided

    def _policy(obs: np.ndarray) -> float:
        t = state.get("t", None)
        # Fallback if caller didn't provide t externally:
        if t is None or t <= 0:
            return 0.0
        r_lag = returns[t - 1]
        return float(k * np.sign(r_lag))
    return _policy

def delta_hedge_policy(delta_series: pd.Series, scale: float = 1.0) -> Callable[[np.ndarray], float]:
    """
    Hedge using option delta from panel: a_t = -scale * delta_t
    Pass a pandas Series (indexed like the panel).
    """
    def _policy_at_t(t: int) -> float:
        try:
            d = float(delta_series.iloc[t])
        except Exception:
            d = 0.0
        return float(-scale * d)

    # wrapper to match (obs)->float signature; user sets env.t externally if needed
    def _policy(obs) -> float:
        # By default, this returns 0. You should call _policy_at_t(env.t) in your loop,
        # or adapt env.rollout to pass t into policies. Kept simple here.
        return 0.0

    # expose helper for manual loops
    _policy.at_t = _policy_at_t  # type: ignore[attr-defined]
    return _policy
