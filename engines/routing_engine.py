# engines/routing_engine.py

from __future__ import annotations
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import networkx as nx

# Optional: internet-based geocoding for place names
try:
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter
    _GEOCODER_AVAILABLE = True
except ImportError:
    _GEOCODER_AVAILABLE = False


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
    loc_df: pd.DataFrame,
    depot_id: str,
) -> List[Tuple[str, List[str], float]]:
    """
    Returns shortest path from depot to each other node.
    Requires columns: id, lat, lon.
    """
    if "id" not in loc_df or "lat" not in loc_df or "lon" not in loc_df:
        raise ValueError("Location df must have columns: id, lat, lon")

    G = build_graph(loc_df)
    routes: List[Tuple[str, List[str], float]] = []
    for dest in loc_df["id"]:
        if dest == depot_id:
            continue
        path = nx.shortest_path(G, depot_id, dest, weight="distance")
        dist = nx.path_weight(G, path, weight="distance")
        routes.append((dest, path, dist))
    return routes


def geocode_locations_from_names(
    df: pd.DataFrame,
    place_col: str,
    id_col: str | None = None,
    user_agent: str = "agentic-policy-engine",
    min_delay_seconds: float = 1.0,
) -> pd.DataFrame:
    """
    AUTO-GEOCODING MODE

    Takes a dataframe with a column of place names (e.g. county, town, co-op)
    and returns a new dataframe with columns [id, lat, lon].

    - If id_col is None, uses the place_col values as ids.
    - Uses OpenStreetMap Nominatim via geopy (internet required).
    - Geocodes UNIQUE place names and re-joins results back.

    This will raise a ValueError if geopy is not installed or internet is blocked.
    """
    if place_col not in df.columns:
        raise ValueError(f"Column '{place_col}' not found in dataframe.")

    if not _GEOCODER_AVAILABLE:
        raise ImportError(
            "geopy is not installed. Add 'geopy' to requirements.txt to enable auto-geocoding."
        )

    # Prepare ids and place names
    if id_col is not None and id_col in df.columns:
        ids = df[id_col].astype(str)
    else:
        # Use place names as ids if no explicit ID
        ids = df[place_col].astype(str)

    places = df[place_col].astype(str)

    # Unique mapping
    unique_places = places.unique().tolist()

    geolocator = Nominatim(user_agent=user_agent, timeout=10)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=min_delay_seconds)

    place_to_coord: Dict[str, tuple[float, float]] = {}

    for p in unique_places:
        try:
            location = geocode(p)
        except Exception:
            location = None

        if location is not None:
            place_to_coord[p] = (location.latitude, location.longitude)
        else:
            # If we cannot geocode, leave it missing and filter later
            place_to_coord[p] = (np.nan, np.nan)

    # Build loc_df
    lat_list = []
    lon_list = []
    for p in places:
        lat, lon = place_to_coord.get(p, (np.nan, np.nan))
        lat_list.append(lat)
        lon_list.append(lon)

    loc_df = pd.DataFrame(
        {
            "id": ids.values,
            "lat": lat_list,
            "lon": lon_list,
        }
    )

    # Drop any rows that still don't have coordinates
    loc_df = loc_df.dropna(subset=["lat", "lon"]).drop_duplicates(subset=["id"]).reset_index(drop=True)

    if loc_df.empty:
        raise ValueError(
            "Auto-geocoding failed for all entries. Check place names, internet access, or API rate limits."
        )

    return loc_df
