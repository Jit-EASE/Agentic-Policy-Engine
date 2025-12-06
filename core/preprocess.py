# preprocess.py

from __future__ import annotations
import pandas as pd


def coerce_time(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    try:
        out["time"] = pd.to_datetime(out["time"], errors="coerce")
        if out["time"].isna().all():
            # fallback to integer index
            out["time"] = range(len(out))
    except Exception:
        out["time"] = range(len(out))
    return out


def filter_series(df: pd.DataFrame, region: str | None, metric: str | None) -> pd.DataFrame:
    out = df.copy()
    if region is not None and region != "All":
        out = out[out["region"] == region]
    if metric is not None and metric != "All":
        out = out[out["metric"] == metric]
    return out.sort_values("time").reset_index(drop=True)


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.dropna(subset=["value"])
    return out
