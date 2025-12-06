# mpc_engine.py

from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd


def simple_mpc_trajectory(
    current_value: float,
    target_value: float,
    horizon_steps: int = 10,
    max_delta_per_step: float | None = None,
) -> pd.Series:
    """
    A very simplified "MPC-style" trajectory:
    - linearly approaches target over horizon
    - optionally enforces max change per step
    This is a placeholder; you can replace with cvxpy-based QP later.
    """
    if horizon_steps <= 0:
        raise ValueError("Horizon steps must be positive")

    values = [current_value]
    remaining = target_value - current_value

    for t in range(1, horizon_steps + 1):
        # naive equal step
        ideal_step = remaining / (horizon_steps - (t - 1)) if horizon_steps - (t - 1) > 0 else 0.0
        if max_delta_per_step is not None:
            step = np.clip(ideal_step, -max_delta_per_step, max_delta_per_step)
        else:
            step = ideal_step
        new_val = values[-1] + step
        values.append(new_val)
        remaining = target_value - new_val

    idx = range(len(values))
    return pd.Series(values, index=idx, name="mpc_trajectory")
