# schema_mapper.py

from __future__ import annotations
from typing import Dict, List
import pandas as pd
from config import (
    DEFAULT_TIME_COL_CANDIDATES,
    DEFAULT_REGION_COL_CANDIDATES,
    DEFAULT_VALUE_COL_CANDIDATES,
    DEFAULT_METRIC_COL_CANDIDATES,
)


def _guess_column(cols: List[str], candidates: List[str]) -> str | None:
    lowered = [c.lower() for c in cols]
    for cand in candidates:
        for c in cols:
            if cand in c.lower():
                return c
    return None


def suggest_schema(df: pd.DataFrame) -> Dict[str, str | None]:
    cols = list(df.columns)

    time_col = _guess_column(cols, DEFAULT_TIME_COL_CANDIDATES)
    region_col = _guess_column(cols, DEFAULT_REGION_COL_CANDIDATES)
    metric_col = _guess_column(cols, DEFAULT_METRIC_COL_CANDIDATES)

    # value column: choose a numeric col if not obvious
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    value_col = _guess_column(cols, DEFAULT_VALUE_COL_CANDIDATES)
    if value_col is None and num_cols:
        value_col = num_cols[0]

    return {
        "time": time_col,
        "region": region_col,
        "metric": metric_col,
        "value": value_col,
    }


def build_canonical_long(
    df: pd.DataFrame,
    time_col: str | None,
    region_col: str | None,
    metric_col: str | None,
    value_col: str | None,
) -> pd.DataFrame:
    """
    Returns a long-format frame with: time, region, metric, value, plus extras.
    """
    tmp = df.copy()

    if time_col is None:
        tmp["time"] = range(len(tmp))
        time_col = "time"

    if region_col is None:
        tmp["region"] = "All"
        region_col = "region"

    if metric_col is None:
        # If value_col is set, label metric by value_col
        label = value_col if value_col is not None else "metric"
        tmp["metric"] = label
        metric_col = "metric"

    if value_col is None:
        # Try to pick any numeric column
        num_cols = [c for c in tmp.columns if pd.api.types.is_numeric_dtype(tmp[c])]
        if not num_cols:
            raise ValueError("No numeric column found for value.")
        value_col = num_cols[0]

    canonical = tmp[[time_col, region_col, metric_col, value_col]].copy()
    canonical.columns = ["time", "region", "metric", "value"]

    return canonical
