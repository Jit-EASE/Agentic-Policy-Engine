# engines/queue_engine.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict


@dataclass
class QueueResults:
    rho: float
    stable: bool
    Lq: float
    Wq: float
    L: float
    W: float


def mmc_queue(arrival_rate: float, service_rate: float, servers: int) -> QueueResults:
    """
    Simple M/M/c queue approx using Erlang-C formula.
    arrival_rate (lambda), service_rate (mu), servers (c).
    Units must be consistent.
    """
    if arrival_rate <= 0 or service_rate <= 0 or servers <= 0:
        raise ValueError("Rates and servers must be positive.")

    rho = arrival_rate / (servers * service_rate)

    # handle overload
    if rho >= 1:
        # system unstable, approximate infinite queue metrics
        return QueueResults(rho=rho, stable=False, Lq=float("inf"), Wq=float("inf"), L=float("inf"), W=float("inf"))

    import math

    a = arrival_rate / service_rate
    # P0 (probability of zero in system)
    sum_terms = sum((a**n) / math.factorial(n) for n in range(servers))
    last_term = (a**servers) / (math.factorial(servers) * (1 - rho))
    P0 = 1.0 / (sum_terms + last_term)

    # Erlang C
    Pc = (a**servers * P0) / (math.factorial(servers) * (1 - rho))

    Lq = (Pc * rho) / (1 - rho)
    Wq = Lq / arrival_rate
    L = Lq + a
    W = L / arrival_rate

    return QueueResults(rho=rho, stable=True, Lq=Lq, Wq=Wq, L=L, W=W)


def queue_summary_dict(results: QueueResults) -> Dict[str, float | bool]:
    return {
        "utilisation_rho": results.rho,
        "stable": results.stable,
        "Lq_expected_queue_length": results.Lq,
        "Wq_expected_wait_time": results.Wq,
        "L_expected_system_size": results.L,
        "W_expected_system_time": results.W,
    }
