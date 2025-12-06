# wavelet_engine.py

from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np
import pywt


def wavelet_decompose(series: pd.Series, wavelet: str = "db4", level: int | None = None) -> Dict[str, pd.Series]:
    """
    Decompose a series into approximate trend and detail (shocks).
    """
    y = series.values.astype(float)

    coeffs = pywt.wavedec(y, wavelet, level=level)
    # Approximation is coeffs[0], details are coeffs[1:]
    approx = pywt.waverec([coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]], wavelet)
    approx = approx[: len(y)]
    shocks = y - approx

    approx_s = pd.Series(approx, index=series.index, name=f"{series.name}_trend")
    shocks_s = pd.Series(shocks, index=series.index, name=f"{series.name}_shocks")

    return {"trend": approx_s, "shocks": shocks_s}
