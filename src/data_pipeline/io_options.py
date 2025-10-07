# src/data_pipeline/io_options.py
from __future__ import annotations
import pandas as pd
import pyarrow.dataset as ds
import pyarrow as pa
from pathlib import Path
from typing import Iterable, Optional, Union, Sequence

# ---- sensible defaults for large option panels ----
DEFAULT_COLS: Sequence[str] = (
    "date","underlying","put_call","expiry","strike",
    "bid","ask","mid","last","iv","delta","gamma","vega","theta",
    "open_interest","volume"
)

def load_option_panel(
    root: Union[str, Path],
    ticker: Optional[str] = None,                 # e.g., "SPY" or "SPX"
    start: Optional[str] = None,                  # "YYYY-MM-DD"
    end: Optional[str] = None,                    # "YYYY-MM-DD"
    columns: Optional[Iterable[str]] = None,      # subset of DEFAULT_COLS
    repartition_to_pandas: bool = True,           # collect to pandas at the end
) -> pd.DataFrame:
    """
    Load large option price data (OptionMetrics-like) directly from Parquet without snapshots.
    Applies column projection + predicate pushdown for speed/memory.

    Supports a directory of partitioned parquet files (recommended).
    """
    root = Path(root)
    if columns is None:
        columns = DEFAULT_COLS

    # Arrow dataset (handles partitioned directories)
    dataset = ds.dataset(root, format="parquet")

    # ---- build filters (pushdown) ----
    filters = []
    if ticker is not None and "underlying" in dataset.schema.names:
        filters.append(ds.field("underlying") == ticker.upper())

    # normalize date bounds if present
    if "date" in dataset.schema.names:
        if start:
            filters.append(ds.field("date") >= pd.to_datetime(start))
        if end:
            filters.append(ds.field("date") <= pd.to_datetime(end))

    # ---- scan with projection + filters ----
    scan = dataset.to_table(
        columns=[c for c in columns if c in dataset.schema.names],
        filter=ds.and_(*filters) if filters else None
    )

    # ---- convert to pandas with compact dtypes ----
    if repartition_to_pandas:
        df = scan.to_pandas(types_mapper=_arrow_to_pandas_dtype)
        # basic hygiene
        if "date" in df:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
        # compact categoricals
        for cat_col in ("underlying","put_call"):
            if cat_col in df:
                df[cat_col] = df[cat_col].astype("category")
        # downcast numerics where safe
        for num_col in ("strike","bid","ask","mid","last","iv","delta","gamma","vega","theta"):
            if num_col in df and pd.api.types.is_float_dtype(df[num_col]):
                df[num_col] = pd.to_numeric(df[num_col], downcast="float")
        for num_col in ("open_interest","volume"):
            if num_col in df and pd.api.types.is_integer_dtype(df[num_col]):
                df[num_col] = pd.to_numeric(df[num_col], downcast="unsigned")
        return df

    # If you prefer to keep as Arrow Table:
    return scan


def _arrow_to_pandas_dtype(arrow_type: pa.DataType):
    """Map Arrow types to compact pandas dtypes."""
    if pa.types.is_integer(arrow_type):
        # keep ints compact; later we may downcast more
        return pd.Int64Dtype()
    if pa.types.is_boolean(arrow_type):
        return pd.BooleanDtype()
    if pa.types.is_float32(arrow_type) or pa.types.is_float16(arrow_type):
        return "float32"
    if pa.types.is_float(arrow_type):
        return "float64"
    return None  # fallback default
