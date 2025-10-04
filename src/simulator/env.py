from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd

RewardFn = Callable[[float, Dict[str, Any]], float]

@dataclass
class HedgingConfig:
    """
    Basic environment config.
    - hedge_price_col: column in panel used to mark to market the hedge instrument (e.g., 'close_spy')
    - state_cols: feature columns exposed to the agent at each step
    - txn_cost_bps: one-way transaction cost per notional (bps)
    - pos_limit: hard cap on absolute hedge units to avoid runaway policies
    """
    hedge_price_col: str = "close_spy"
    state_cols: Tuple[str, ...] = ("close_spy", "vix", "hvol_10d", "rate_10y")
    txn_cost_bps: float = 0.5
    pos_limit: float = 10_000.0  # units of hedge instrument
    start_index: int = 1         # start at 1 so we have t-1 for returns/pnl

class HedgingEnv:
    """
    Deterministic, historical replay environment for deep hedging.

    - Observation: panel.loc[t, state_cols] (numpy array)
    - Action: hedge position in hedge instrument (float, units)
    - Reward: computed via injected reward_fn(pnl, info)
    - Episode ends at the last index of the panel.

    Notes:
    - You pass in a *time-indexed* pandas DataFrame (panel) with at least `hedge_price_col` and `state_cols`.
    - All alignment/NaN handling should be done upstream; we assert columns exist here.
    """

    def __init__(self, panel: pd.DataFrame, cfg: HedgingConfig, reward_fn: RewardFn):
        self.panel = panel.copy()
        self.cfg = cfg
        self.reward_fn = reward_fn

        missing = [c for c in (cfg.hedge_price_col, *cfg.state_cols) if c not in self.panel.columns]
        if missing:
            raise ValueError(f"Panel missing required columns: {missing}")

        # internals
        self.t: Optional[int] = None
        self.done: bool = False
        self.pos: float = 0.0
        self.nav: float = 0.0

        # vectors for speed
        self._price = self.panel[cfg.hedge_price_col].astype(float).values
        self._state = self.panel[list(cfg.state_cols)].astype(float).values

    # -------------- Public API --------------
    @property
    def n_steps(self) -> int:
        return len(self.panel)

    def reset(self, t0: Optional[int] = None) -> np.ndarray:
        self.t = int(t0 if t0 is not None else self.cfg.start_index)
        if not (0 <= self.t < self.n_steps):
            raise IndexError(f"t0 {self.t} out of bounds [0, {self.n_steps-1}]")
        self.done = False
        self.pos = 0.0
        self.nav = 0.0
        return self._state[self.t].copy()

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("step() called after episode is done. Call reset().")
        if self.t is None:
            raise RuntimeError("Call reset() before step().")

        # clamp action to prevent crazy positions
        a = float(np.clip(action, -self.cfg.pos_limit, self.cfg.pos_limit))
        t = self.t

        # price now and next
        if t + 1 >= self.n_steps:
            self.done = True
            return self._state[t].copy(), 0.0, True, {"reason": "eof"}

        px_t = self._price[t]
        px_n = self._price[t + 1]

        # transaction cost on change in position
        d_pos = a - self.pos
        txn_cost = abs(d_pos) * px_t * (self.cfg.txn_cost_bps / 1e4)

        # pnl from holding current pos over [t, t+1]
        pnl_underlying = self.pos * (px_n - px_t)
        pnl = pnl_underlying - txn_cost

        self.nav += pnl
        self.pos = a
        self.t += 1

        info = {
            "t": self.t,
            "price_t": px_t,
            "price_next": px_n,
            "pnl_underlying": pnl_underlying,
            "txn_cost": txn_cost,
            "pos": self.pos,
            "nav": self.nav,
        }
        reward = float(self.reward_fn(pnl, info))
        self.done = (self.t + 1 >= self.n_steps)
        obs_next = self._state[self.t].copy()
        return obs_next, reward, self.done, info

    # convenience run helper (handy for baselines / smoke tests)
    def rollout(self, policy: Callable[[np.ndarray], float], max_steps: Optional[int] = None):
        obs = self.reset()
        rewards, navs, positions = [], [], []
        steps = 0
        horizon = max_steps if max_steps is not None else (self.n_steps - self.cfg.start_index - 1)
        for _ in range(horizon):
            a = float(policy(obs))
            obs, r, done, info = self.step(a)
            rewards.append(r)
            navs.append(info["nav"])
            positions.append(info["pos"])
            steps += 1
            if done:
                break
        return {
            "rewards": np.array(rewards),
            "nav": np.array(navs),
            "positions": np.array(positions),
            "steps": steps,
        }
