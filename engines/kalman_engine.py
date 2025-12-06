# kalman_engine.py

from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd


def univariate_kalman_smoother(
    series: pd.Series,
    process_var: float = 1.0,
    meas_var: float = 4.0,
) -> Dict[str, pd.Series]:
    """
    Minimal univariate Kalman filter + RTS smoother.
    """
    y = series.values.astype(float)
    n = len(y)
    if n == 0:
        raise ValueError("Empty series for Kalman.")

    # Prior
    x_est = np.zeros(n)
    P = np.zeros(n)
    x_est[0] = y[0]
    P[0] = 1.0

    Q = process_var
    R = meas_var
    A = 1.0  # identity model
    H = 1.0

    # Filter forward
    for t in range(1, n):
        # Predict
        x_pred = A * x_est[t - 1]
        P_pred = A * P[t - 1] * A + Q

        # Update
        K = P_pred * H / (H * P_pred * H + R)
        x_est[t] = x_pred + K * (y[t] - H * x_pred)
        P[t] = (1 - K * H) * P_pred

    # RTS smoother
    x_smooth = np.zeros(n)
    x_smooth[-1] = x_est[-1]
    for t in range(n - 2, -1, -1):
        C = P[t] * A / (A**2 * P[t] + Q)
        x_smooth[t] = x_est[t] + C * (x_smooth[t + 1] - A * x_est[t])

    smooth_series = pd.Series(x_smooth, index=series.index, name=f"{series.name}_smooth")
    filtered_series = pd.Series(x_est, index=series.index, name=f"{series.name}_filtered")

    return {"filtered": filtered_series, "smoothed": smooth_series}
