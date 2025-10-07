#!/usr/bin/env python3
"""
Data cleaning pipeline (generic + options-aware).

This module is **library-first** so it can be:
- imported by notebooks (run_pipeline),
- invoked by your package CLI via `attach_subparser(...)`,
- or run standalone (`python -m data_pipeline.clean_data --help`).

It intentionally stays conservative: only basic hygiene + light domain checks.
"""

from __future__ import annotations
import os
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil import parser
from data_pipeline.config import RAW_DIR, CLEANED, ALIGNED
import argparse


def load_csv_with_date(path):
    """Read a CSV and ensure there is a proper 'date' column of dtype datetime64[ns]."""
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        # fall back: treat first column as date-like index
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.reset_index().rename(columns={"index": "date"})
    # drop rows with invalid dates
    df = df[df["date"].notna()]
    # sort & dedup on date if needed (keep first for identical duplicates)
    df = df.sort_values("date")
    df = df[~df["date"].duplicated(keep="first")] if df["date"].is_unique is False else df
    return df

def load_market(start_date=None):
    mkt_path = os.path.join(CLEANED, "market_daily_clean.csv")
    mkt = load_csv_with_date(mkt_path)
    if start_date:
        mkt = mkt[mkt["date"] >= pd.to_datetime(start_date)]
    return mkt

def row_preserving_join_with_market(df, market_df):
    """Inner-join market columns onto df by date, preserving all df rows for those dates."""
    # avoid column collisions: we won't rename, just rely on union; market cols have distinct names (close_spy, close_gspc, etc.)
    merged = df.merge(market_df, on="date", how="inner")
    return merged

def safe_load(path):
    if os.path.exists(path):
        return load_csv_with_date(path)
    else:
        print(f"⚠️ Skipping (not found): {path}")
        return None
