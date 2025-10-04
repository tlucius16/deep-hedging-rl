
# [project.scripts]
# dh-pipeline = "data_pipeline.cli:main"

from __future__ import annotations
import argparse
from .validate import validate_raw
from .builders import (
    build_market_daily, build_options_snapshot,
    build_vol_surface_long, build_market_plus_hvol_fwd,
)

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
    "task",
    choices=[
        "validate","market",
        "options_spy","options_spx",
        "surface_spy","surface_spx",
        "extended_spy","extended_spx",
        "all"
    ]
)
    args = p.parse_args()

    if args.task in {"validate","all"}: validate_raw()
    if args.task in {"market","all"}:   build_market_daily(save=True)
    if args.task in {"options_spy","all"}: build_options_snapshot("spy", save=True)
    if args.task in {"options_spx","all"}: build_options_snapshot("spx", save=True)
    if args.task in {"surface_spy","all"}: build_vol_surface_long("spy", save=True)
    if args.task in {"surface_spx","all"}: build_vol_surface_long("spx", save=True)
    if args.task in {"extended_spy","all"}: build_market_plus_hvol_fwd("spy", save=True)
    if args.task in {"extended_spx","all"}: build_market_plus_hvol_fwd("spx", save=True)

if __name__ == "__main__":
    main()


