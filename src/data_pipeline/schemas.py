from dataclasses import dataclass

@dataclass(frozen=True)
class Cols:
    DATE = "date"
    VALUE = "value"

# Add canonical names you’ll use after standardization
@dataclass(frozen=True)
class Canonical:
    SPY_CLOSE = "close_spy"
    GSPC_CLOSE = "close_gspc"
    VIX = "vix"
    DGS10 = "rate_10y"
