from __future__ import annotations
from pathlib import Path

# project root assumed to be repo root when running scripts from there
ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# Map logical dataset names → filename in data/raw + column mapping
# Adjust these filenames to match exactly what you have in your screenshot
DATASETS = {
    "spy": {
        "file": "SPY_data.csv",
        # possible columns across sources -> standardized name 'close'
        "date_candidates": ["Date", "DATE", "date", "Trade Date"],
        "value_candidates": ["Adj Close", "Close", "PX_LAST", "VALUE", "close"],
        "rename": {"value": "close_spy"},
        "freq": "D",
        "skiprows": 2,  # if needed, e.g. for header rows
    },
    "gspc": {
        "file": "GSPC_data.csv",
        "date_candidates": ["Date", "DATE", "date"],
        "value_candidates": ["Adj Close", "Close", "PX_LAST", "VALUE", "close"],
        "rename": {"value": "close_gspc"},
        "freq": "D",
        "sikprows": 2,  # if needed, e.g. for header rows
    },
    "vix": {
        "file": "VIXCLS.csv",  # FRED usually
        "date_candidates": ["DATE", "Date", "date"],
        "value_candidates": ["VIXCLS", "VALUE", "Close", "PX_LAST"],
        "rename": {"value": "vix"},
        "freq": "D",
    },
    "dgs10": {
        "file": "DGS10.csv",   # FRED 10y treasury
        "date_candidates": ["DATE", "Date", "date"],
        "value_candidates": ["DGS10", "VALUE", "PX_LAST"],
        "rename": {"value": "rate_10y"},
        "freq": "D",
    },
    # add more here as needed:
    # "hvol": { "file": "hvol.csv", ... },
    # "vsurf": { "file": "vsurfd.csv", ... },
}

DATASETS.update({
    # ----- scalar / single-series time series -----
    "hvol": {
        "file": "hvold.csv",
        "date_candidates": ["date", "DATE"],
        "value_candidates": ["hvol", "HIST_VOL", "RealizedVol", "HVOL"],
        "rename": {"value": "hvol_20d"},   # adjust if 10d/30d etc.
        "freq": "D",
    },
    "fwd": {
        "file": "fwdprd.csv",
        "date_candidates": ["date", "DATE"],
        "value_candidates": ["forward", "FWD_PX", "FWD_PRICE", "VALUE"],
        "rename": {"value": "fwd_price"},
        "freq": "D",
    },

    # ----- option datasets (tabular) -----
    "op_prices": {
        "file": "opprcd.csv",
        # typical columns; loaders will pick whatever exists
        "cands": {
            "date": ["date", "DATE"],
            "underlying": ["underlying", "UNDERLYING", "TICKER", "ROOT"],
            "put_call": ["cp_flag", "PUT_CALL", "OptionType"],
            "expiry": ["expiry", "EXPIRATION", "EXPIRY", "OPT_EXPIRE_DT"],
            "strike": ["strike", "STRIKE", "STRIKE_PX"],
            "bid": ["bid", "BID"],
            "ask": ["ask", "ASK"],
            "mid": ["mid", "MID", "PX_MID"],
            "last": ["last", "LAST", "PX_LAST"],
            "iv": ["iv", "IMPLIED_VOL", "IVOL"],
            "delta": ["delta", "DELTA"],
            "gamma": ["gamma", "GAMMA"],
            "vega": ["vega", "VEGA"],
            "theta": ["theta", "THETA"],
            "open_interest": ["open_interest", "OPEN_INT", "OI"],
        }
    },
    "op_volume": {
        "file": "opvold.csv",
        "cands": {
            "date": ["date", "DATE"],
            "underlying": ["underlying", "UNDERLYING", "TICKER", "ROOT"],
            "put_call": ["cp_flag", "PUT_CALL", "OptionType"],
            "expiry": ["expiry", "EXPIRATION", "EXPIRY", "OPT_EXPIRE_DT"],
            "strike": ["strike", "STRIKE", "STRIKE_PX"],
            "volume": ["volume", "VOLUME"],
            "open_interest": ["open_interest", "OPEN_INT", "OI"],
        }
    },
    "vol_surface": {
        "file": "vsurfd.csv",
        "cands": {
            "date": ["date", "DATE"],
            "underlying": ["underlying", "UNDERLYING", "TICKER", "ROOT"],
            "moneyness": ["moneyness", "MNYNESS", "MONEYN", "DeltaBucket"],
            "tenor_d": ["tenor_d", "TENOR_D", "DaysToExp", "DAYS_TO_EXP"],
            "expiry": ["expiry", "EXPIRY", "EXPIRATION", "OPT_EXPIRE_DT"],  # optional if tenor supplied
            "iv": ["iv", "IMPLIED_VOL", "IVOL"],
        }
    },
})

DATASETS.update({
    "spy": {
        "file": "SPY_data.csv",
        "date_candidates": ["Date", "DATE", "date", "Trade Date", "Unnamed: 0", "index"],
        "value_candidates": ["Adj Close", "Close", "PX_LAST", "VALUE", "Price"],
        "rename": {"value": "close_spy"},
        "freq": "D",
    },
    "gspc": {
        "file": "GSPC_data.csv",
        "date_candidates": ["Date", "DATE", "date", "Trade Date", "Unnamed: 0", "index"],
        "value_candidates": ["Adj Close", "Close", "PX_LAST", "VALUE", "Price"],
        "rename": {"value": "close_gspc"},
        "freq": "D",
    },
    # keep vix, dgs10, etc. as before
})

