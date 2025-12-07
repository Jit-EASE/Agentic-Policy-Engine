# engines/kalman_engine.py

import numpy as np
import pandas as pd
from typing import Dict


def univariate_kalman_smoother(
    series: pd.Series,
    process_var: float = 1.0,
    meas_var: float = 4.0,
) -> Dict[str, pd.Series]:
    """
    Minimal univariate Kalman filter + RTS smoother.
    Safely handles non-numeric values by coercing to numeric and dropping invalid entries.
    """

    # -------------------------------------------------------------------
    # STEP 1: CLEAN SERIES — COERCE TO NUMERIC AND DROP INVALID VALUES
    # -------------------------------------------------------------------
    numeric_series = pd.to_numeric(series, errors="coerce").dropna()

    n = len(numeric_series)
    if n == 0:
        raise ValueError("Empty or non-numeric series for Kalman.")

    # Convert clean values to float array
    y = numeric_series.values.astype(float)

    # -------------------------------------------------------------------
    # STEP 2: INITIALISE FILTER MEMORY
    # -------------------------------------------------------------------
    x_est = np.zeros(n)
    P = np.zeros(n)

    # Initial state estimate
    x_est[0] = y[0]
    P[0] = 1.0

    # Define model parameters
    Q = process_var  # process noise variance
    R = meas_var     # measurement noise variance
    A = 1.0          # state transition
    H = 1.0          # observation model

    # -------------------------------------------------------------------
    # STEP 3: KALMAN FORWARD FILTER
    # -------------------------------------------------------------------
    for t in range(1, n):
        # Prediction step
        x_pred = A * x_est[t - 1]
        P_pred = A * P[t - 1] * A + Q

        # Kalman Gain
        K = P_pred * H / (H * P_pred * H + R)

        # Update step
        x_est[t] = x_pred + K * (y[t] - H * x_pred)
        P[t] = (1 - K * H) * P_pred

    # -------------------------------------------------------------------
    # STEP 4: RTS SMOOTHER (BACKWARD PASS)
    # -------------------------------------------------------------------
    x_smooth = np.zeros(n)
    x_smooth[-1] = x_est[-1]

    for t in range(n - 2, -1, -1):
        C = P[t] * A / (A**2 * P[t] + Q)
        x_smooth[t] = x_est[t] + C * (x_smooth[t + 1] - A * x_est[t])

    # -------------------------------------------------------------------
    # STEP 5: RETURN CLEAN, INDEX-ALIGNED SERIES
    # -------------------------------------------------------------------
    smooth_series = pd.Series(
        x_smooth,
        index=numeric_series.index,
        name=f"{series.name}_smooth",
    )

    filtered_series = pd.Series(
        x_est,
        index=numeric_series.index,
        name=f"{series.name}_filtered",
    )

    return {
        "filtered": filtered_series,
        "smoothed": smooth_series,
    }
