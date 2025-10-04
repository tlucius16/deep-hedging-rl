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

def build_combined(symbol, start_date):
    """
    Build one combined table for a symbol ('spx' or 'spy'):
      - load market_daily (trimmed to start_date)
      - load symbol-specific cleaned datasets
      - merge market cols into each dataset by date (row-preserving)
      - concatenate all rows together (add 'source' column), keep ALL columns
    """
    out_dir = os.path.join(ALIGNED, symbol)
    os.makedirs(out_dir, exist_ok=True)

    # market (trimmed)
    mkt = load_market(start_date=start_date)

    # expected cleaned inputs for each symbol
    parts = []
    sources = []

    if symbol == "spx":
        files = [
            ("options_snapshot_spx_clean.csv", "options_snapshot_spx"),
            ("vol_surface_spx_clean.csv",     "vol_surface_spx"),
            ("market_extended_spx_clean.csv", "market_extended_spx"),  # optional if present
        ]
    else:  # spy
        files = [
            ("options_snapshot_spy_clean.csv", "options_snapshot_spy"),
            ("vol_surface_spy_clean.csv",      "vol_surface_spy"),
            ("market_extended_spy_clean.csv",  "market_extended_spy"), # optional if present
        ]

    # 1) row-preserving merge of each dataset with market
    for fname, tag in files:
        path = os.path.join(CLEANED, fname)
        df = safe_load(path)
        if df is None:
            continue
        merged = row_preserving_join_with_market(df, mkt)
        merged.insert(0, "source", tag)
        parts.append(merged)
        sources.append(tag)
        print(f"✅ {tag}: {merged.shape[0]:,} rows, {merged.shape[1]} cols")

    # 2) also include market_daily rows themselves (one row/day) if you want them in the unified file
    mkt_rows = mkt.copy()
    mkt_rows.insert(0, "source", "market_daily")
    parts.append(mkt_rows)
    sources.append("market_daily")
    print(f"✅ market_daily: {mkt_rows.shape[0]:,} rows, {mkt_rows.shape[1]} cols")

    # 3) union all rows, keeping ALL columns (outer-join on columns by concat)
    if not parts:
        raise RuntimeError(f"No datasets found for {symbol.upper()}.")
    combined = pd.concat(parts, axis=0, ignore_index=True, sort=False)

    # Ensure 'date' is present & at front (and source second) for readability
    cols = combined.columns.tolist()
    ordered = ["source", "date"] + [c for c in cols if c not in ("source", "date")]
    combined = combined[ordered]

    # 4) save
    out_csv = os.path.join(out_dir, f"combined_{symbol}_all.csv")
    out_parq = os.path.join(out_dir, f"combined_{symbol}_all.parquet")
    combined.to_csv(out_csv, index=False)
    combined.to_parquet(out_parq, index=False)

    print(f"\n📦 {symbol.upper()} combined saved:")
    print(f"   - {out_csv}")
    print(f"   - {out_parq}")
    print(f"   Shape: {combined.shape[0]:,} rows × {combined.shape[1]} cols")
    print(f"   Date range: {combined['date'].min()} → {combined['date'].max()}")
    # quick per-source counts
    print("\nRow counts by source:")
    print(combined["source"].value_counts().to_string())
