# engine_orchestrator.py

from __future__ import annotations
from typing import Dict, Any
import pandas as pd

from queue_engine import mmc_queue, queue_summary_dict
from kalman_engine import univariate_kalman_smoother
from wavelet_engine import wavelet_decompose
from mpc_engine import simple_mpc_trajectory
from routing_engine import shortest_routes


def run_queue_block(arrival_rate: float, service_rate: float, servers: int) -> Dict[str, Any]:
    res = mmc_queue(arrival_rate, service_rate, servers)
    return queue_summary_dict(res)


def run_kalman_block(series: pd.Series, process_var: float, meas_var: float) -> Dict[str, pd.Series]:
    return univariate_kalman_smoother(series, process_var, meas_var)


def run_wavelet_block(series: pd.Series) -> Dict[str, pd.Series]:
    return wavelet_decompose(series)


def run_mpc_block(
    current_value: float,
    target_value: float,
    horizon_steps: int,
    max_delta: float | None,
) -> pd.Series:
    return simple_mpc_trajectory(current_value, target_value, horizon_steps, max_delta)


def run_routing_block(loc_df: pd.DataFrame, depot_id: str):
    return shortest_routes(loc_df, depot_id)
