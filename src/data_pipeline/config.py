from __future__ import annotations
from pathlib import Path

def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / ".git").exists():
            return parent
    return Path(__file__).resolve().parents[2]

ROOT = _repo_root()

RAW_DIR       = ROOT / "data" / "ProjectData"   # ⬅️ moved from data/raw
PROCESSED_DIR = ROOT / "data" / "processed"
ALIGNED   = PROCESSED_DIR / "aligned"
ALIGNED_SPX  = ALIGNED /  "spx"
ALIGNED_SPY  = ALIGNED / "spy"
CLEANED   = PROCESSED_DIR / "cleaned"
RAW_OPT = ROOT / "data" / "raw" 
RAW_SPX = RAW_OPT / "options_spx"
RAW_SPY = RAW_OPT / "options_spy"

for d in (PROCESSED_DIR, ALIGNED, CLEANED, ALIGNED_SPX, ALIGNED_SPY):
    d.mkdir(parents=True, exist_ok=True)

FILENAMES = {
    "dgs10":                "DGS10.csv",
    "vix":                  "VIXCLS.csv",
    "fwd_spy":              "ForwardPriceSPY.zip",
    "fwd_spx":              "ForwardPriceSPX.zip",
    "std_opts_spy":         "StandardizedOptionsSPY.zip",
    "std_opts_spx":         "StandardizedOptionsSPX.zip",
    "opt_volume_spy":       "OptionVolumeSPY.zip",
    "opt_volume_spx":       "OptionVolumeSPX.zip",
    "vol_surface_spy":      "VolatilitySurfaceSPY.zip",
    "vol_surface_spx":      "VolatilitySurfaceSPX.zip",
    "hist_vol_spy":         "HistoricalVolatilitySPY.zip",
    "hist_vol_spx":         "HistoricalVolatilitySPX.zip",
}

# Keep legacy timeseries configs (used by loaders.load_dataset)
DATASETS = {
    "spy": {
        "file": "SPY_data.csv",   # if you keep this legacy file; else remove this block
        "date_candidates": ["Date","DATE","date","Trade Date","Unnamed: 0","index"],
        "value_candidates": ["Adj Close","Close","PX_LAST","VALUE","Price"],
        "rename": {"value": "close_spy"},
        "freq": "D",
        "skiprows": 2,
    },
    "gspc": {
        "file": "GSPC_data.csv",
        "date_candidates": ["Date","DATE","date","Trade Date","Unnamed: 0","index"],
        "value_candidates": ["Adj Close","Close","PX_LAST","VALUE","Price"],
        "rename": {"value": "close_gspc"},
        "freq": "D",
        "skiprows": 2,
    },
    "vix": {
        "file": FILENAMES["vix"],
        "date_candidates": ["DATE","Date","date"],
        "value_candidates": ["VIXCLS","VALUE","Close","PX_LAST"],
        "rename": {"value": "vix"},
        "freq": "D",
    },
    "dgs10": {
        "file": FILENAMES["dgs10"],
        "date_candidates": ["DATE","Date","date"],
        "value_candidates": ["DGS10","VALUE","PX_LAST"],
        "rename": {"value": "rate_10y"},
        "freq": "D",
    },
}
# config.py (add this block; keep your existing constants)

# Ordered list of candidate files per dataset. First one found & readable wins.
TIMESERIES_SOURCES = {
    "spy": [
        "SPY_data.csv",                # legacy local CSV
        "SPY.parquet",                 # raw parquet (if you upload)
        "SPY.zip",                     # zipped single CSV/Parquet
        # add a vendor export name if you later change it
    ],
    "gspc": [
        "GSPC_data.csv",
        "GSPC.parquet",
        "GSPC.zip",
    ],
    # vix/dgs10 typically fixed names; but you can add alternates too:
    "vix":  ["VIXCLS.csv", "VIXCLS.parquet", "VIXCLS.zip"],
    "dgs10":["DGS10.csv",  "DGS10.parquet",  "DGS10.zip"],
}

# For flexible parsing we still need candidate column names:
TIMESERIES_PARSING = {
    "spy": {
        "date_candidates":  ["Date","DATE","date","Trade Date","Unnamed: 0","index"],
        "value_candidates": ["Adj Close","Close","PX_LAST","VALUE","Price","close"],
        "rename_to": "close_spy",
        "freq": "D",
        "skiprows": 0,  # set to 2 if your legacy file has 2 header rows
    },
    "gspc": {
        "date_candidates":  ["Date","DATE","date","Trade Date","Unnamed: 0","index"],
        "value_candidates": ["Adj Close","Close","PX_LAST","VALUE","Price","close"],
        "rename_to": "close_gspc",
        "freq": "D",
        "skiprows": 0,
    },
    "vix": {
        "date_candidates":  ["DATE","Date","date"],
        "value_candidates": ["VIXCLS","VALUE","Close","PX_LAST","vix"],
        "rename_to": "vix",
        "freq": "D",
        "skiprows": 0,
    },
    "dgs10": {
        "date_candidates":  ["DATE","Date","date"],
        "value_candidates": ["DGS10","VALUE","PX_LAST","rate","yield"],
        "rename_to": "rate_10y",
        "freq": "D",
        "skiprows": 0,
    },
}
