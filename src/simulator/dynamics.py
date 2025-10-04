from __future__ import annotations
import numpy as np

def gbm_paths(s0: float, mu: float, sigma: float, dt: float, n_steps: int, n_paths: int, seed: int | None = None):
    """
    Simple GBM path generator.
    Returns array shape (n_steps+1, n_paths) with first row = s0.
    """
    rng = np.random.default_rng(seed)
    s = np.zeros((n_steps + 1, n_paths), dtype=float)
    s[0, :] = s0
    drift = (mu - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    z = rng.standard_normal(size=(n_steps, n_paths))
    for t in range(n_steps):
        s[t + 1, :] = s[t, :] * np.exp(drift + vol * z[t, :])
    return s
