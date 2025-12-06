# routing_engine.py

from __future__ import annotations
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import networkx as nx


def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0  # km
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dlat = p2 - p1
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def build_graph(loc_df: pd.DataFrame) -> nx.Graph:
    """
    loc_df: columns [id, lat, lon]
    fully connected graph with haversine distances.
    """
    G = nx.Graph()
    for _, row in loc_df.iterrows():
        G.add_node(row["id"], lat=row["lat"], lon=row["lon"])

    ids = list(loc_df["id"])
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a = loc_df.loc[loc_df["id"] == ids[i]].iloc[0]
            b = loc_df.loc[loc_df["id"] == ids[j]].iloc[0]
            d = _haversine(a["lat"], a["lon"], b["lat"], b["lon"])
            G.add_edge(ids[i], ids[j], distance=d)
    return G


def shortest_routes(
    loc_df: pd.DataFrame, depot_id: str
) -> List[Tuple[str, List[str], float]]:
    """
    Returns shortest path from depot to each other node.
    """
    if "id" not in loc_df or "lat" not in loc_df or "lon" not in loc_df:
        raise ValueError("Location df must have columns: id, lat, lon")

    G = build_graph(loc_df)
    routes = []
    for dest in loc_df["id"]:
        if dest == depot_id:
            continue
        path = nx.shortest_path(G, depot_id, dest, weight="distance")
        dist = nx.path_weight(G, path, weight="distance")
        routes.append((dest, path, dist))
    return routes
