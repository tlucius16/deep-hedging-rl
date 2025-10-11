# simulator/build_panel.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
from .features import make_option_features

def build_sim_panel(
    market_df: pd.DataFrame,
    spx_clean_dir: Path,
    include_spy: bool = False,
    spy_clean_dir: Path | None = None,
    act_at_open: bool = False,
    ffill_limit: int = 2,
):
    fe_spx = make_option_features(spx_clean_dir, "spx")

    panel = market_df.reset_index()
    panel = panel.rename(columns={panel.columns[0]: "date"}) if "date" not in panel else panel
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel = panel.merge(fe_spx, on="date", how="left")

    if include_spy and spy_clean_dir is not None:
        from .features import make_option_features as _mk
        fe_spy = _mk(spy_clean_dir, "spy")
        panel = panel.merge(fe_spy, on="date", how="left")

    feat_cols = [c for c in panel.columns if c.startswith(("iv_atm","iv_ts_slope","iv_skew"))]
    if ffill_limit:
        panel[feat_cols] = panel[feat_cols].ffill(limit=ffill_limit)

    # env expects 'ret_fwd'
    panel = panel.sort_values("date")
    panel["ret_fwd"] = panel["close_spy"].pct_change().shift(-1)
    panel = panel.dropna(subset=["ret_fwd"]).reset_index(drop=True)

    if act_at_open:
        panel[feat_cols] = panel[feat_cols].shift(1)
        panel = panel.dropna(subset=feat_cols).reset_index(drop=True)

    # default feature set (trim if some aren't present)
    state_cols = [
        "iv_atm_30d_spx","iv_ts_slope_spx","iv_skew_30d_spx",
        "vix","rate_10y","rv_21d","hvol_30d","hvol_91d"
    ]
    state_cols = [c for c in state_cols if c in panel.columns]

    return panel, state_cols
