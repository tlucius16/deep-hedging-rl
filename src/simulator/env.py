# env.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Callable, Sequence, Optional, Tuple, Dict

StepOut = Tuple[np.ndarray, float, bool, Dict]

class HedgingEnv:
    """
    Minimal, leak-free environment for hedging experiments.

    - Observations: last `window` rows of selected `features`, up to time t (exclusive).
    - Reward: user-supplied function of (pnl, info), where info includes ret, nav, pos, cost, pnl.
    - Costs: charged on CHANGE in position (Δpos) using txn_cost_bps.
    - Returns: uses forward return R[t] = r(t→t+1) — no look-ahead.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        features: Sequence[str],
        reward_fn: Callable[[float, Dict], float],
        window: int = 60,
        txn_cost_bps: float = 0.5,
        pos_limit: float = 1.0,
        scaler: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        rng_seed: int = 0,
    ):
        """
        Args:
            df: DataFrame containing `features` and a column 'ret_fwd' (forward return).
            features: columns to expose in the observation window.
            reward_fn: function (pnl: float, info: dict) -> float.
            window: number of timesteps in the observation window (int >= 1).
            txn_cost_bps: cost in basis points per unit of |Δposition|.
            pos_limit: hard clip on position size in [-pos_limit, +pos_limit].
            scaler: optional callable to transform obs window; signature (ndarray)->ndarray.
            rng_seed: RNG seed for any stochastic choices (kept for consistency).
        """
        self.df = df.copy()
        self.features = list(features)
        self.window = int(window)
        self.txn_cost_bps = float(txn_cost_bps)
        self.pos_limit = float(pos_limit)
        self.scaler = scaler
        self.rng = np.random.default_rng(rng_seed)
        self.reward_fn = reward_fn

        # --- basic validations ---
        req_cols = set(self.features + ["ret_fwd"])
        missing = [c for c in req_cols if c not in self.df.columns]
        if missing:
            raise KeyError(f"HedgingEnv: missing columns in df: {missing}")
        if not isinstance(self.df.index, pd.DatetimeIndex):
            # allow integer index but always sort to keep order deterministic
            self.df = self.df.sort_index()
        else:
            self.df = self.df.sort_index()

        # --- build state arrays ---
        x = self.df[self.features]
        # Drop rows where reward cannot be computed (NaN forward return)
        ok = self.df["ret_fwd"].notna()
        x = x.loc[ok]
        R = self.df.loc[ok, "ret_fwd"]

        self.X = x.values.astype(float)
        self.R = R.values.astype(float)
        self.T = len(self.R)

        if self.window >= self.T:
            raise ValueError("HedgingEnv: window too large for available series length.")

        # runtime state
        self.t = None
        self.nav = None
        self.pos = None
        self.done = None

    # ----- API -----
    def reset(self) -> np.ndarray:
        self.t = self.window
        self.nav = 1.0
        self.pos = 0.0
        self.done = False
        return self._obs()

    def _obs(self) -> np.ndarray:
        w = self.X[self.t - self.window : self.t]
        if self.scaler is not None:
            w = self.scaler(w)
        # defend against any accidental NaNs/Infs in features
        return np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

    def step(self, action: float) -> StepOut:
        if self.done:
            # Gym-like behavior: after done, return same obs and zero reward
            return self._obs(), 0.0, True, {"ret": 0.0, "pnl": 0.0, "pos": self.pos, "nav": self.nav, "cost": 0.0}

        a = float(np.clip(action, -self.pos_limit, self.pos_limit))
        r = float(self.R[self.t])  # forward return t→t+1

        dpos = a - self.pos
        cost = (self.txn_cost_bps * 1e-4) * abs(dpos)  # bps -> proportion
        pnl = a * r - cost

        self.nav *= (1.0 + pnl)
        self.pos = a
        self.t += 1
        self.done = self.t >= (self.T - 1)  # last usable forward return index is T-1

        info = {"ret": r, "pnl": pnl, "pos": self.pos, "nav": self.nav, "cost": cost}
        reward = float(self.reward_fn(pnl, info))
        return self._obs(), reward, self.done, info

    # Convenience rollout for quick baselines/tests
    def rollout(self, policy: Callable[[np.ndarray], float], max_steps: Optional[int] = None) -> Dict[str, np.ndarray]:
        obs = self.reset()
        rewards, navs, positions, rets, pnls, costs = [], [self.nav], [self.pos], [], [], []
        steps = 0
        limit = max_steps or (self.T - self.window - 1)
        while not self.done and steps < limit:
            a = float(policy(obs))
            obs, rwd, done, info = self.step(a)
            rewards.append(rwd)
            navs.append(info["nav"])
            positions.append(info["pos"])
            rets.append(info["ret"])
            pnls.append(info["pnl"])
            costs.append(info["cost"])
            steps += 1
        return {
            "rewards": np.asarray(rewards),
            "nav": np.asarray(navs),
            "positions": np.asarray(positions),
            "rets": np.asarray(rets),
            "pnls": np.asarray(pnls),
            "costs": np.asarray(costs),
            "steps": steps,
        }
