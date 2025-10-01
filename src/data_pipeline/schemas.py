from __future__ import annotations
from dataclasses import dataclass

# Primitive column names used during staging
@dataclass(frozen=True)
class Cols:
    DATE = "date"
    VALUE = "value"

# Canonical names for final, standardized series/tables
@dataclass(frozen=True)
class Canonical:
    # time series
    SPY_CLOSE = "close_spy"
    GSPC_CLOSE = "close_gspc"
    VIX = "vix"
    DGS10 = "rate_10y"

    # common macro/market extras (add as needed)
    HVOL = "hvol"
    FWD_PRICE = "fwd_price"

# Canonical column set for options-like tables
@dataclass(frozen=True)
class OptCols:
    DATE = "date"
    UNDERLYING = "underlying"
    PUT_CALL = "put_call"     # 'C' or 'P'
    EXPIRY = "expiry"
    STRIKE = "strike"
    BID = "bid"
    ASK = "ask"
    MID = "mid"
    LAST = "last"
    IV = "iv"
    DELTA = "delta"
    GAMMA = "gamma"
    VEGA = "vega"
    THETA = "theta"
    OPEN_INT = "open_interest"
    VOLUME = "volume"

# Canonical for vol surface-style tables
@dataclass(frozen=True)
class SurfCols:
    DATE = "date"
    UNDERLYING = "underlying"
    EXPIRY = "expiry"     # optional if tenor provided
    TENOR_D = "tenor_d"   # days-to-expiry
    MNY = "moneyness"     # or delta bucket, but we keep the name generic
    IV = "iv"
