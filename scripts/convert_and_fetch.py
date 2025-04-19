#!/usr/bin/env python3
"""
Script to merge elevation data with all trail JSON files in `dataset/raw`, generate a graph where
nodes represent locations with altitude metadata and edges with distance & altitude_diff,
fetch OSM location importance via reverse geocoding, and output two CSVs:
- `nodes.csv`: name, latitude, longitude, altitude, importance
- `edges.csv`: source, target, distance, altitude_diff
"""

import os
import math
import time
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from scipy.spatial import cKDTree
import networkx as nx

# User agent for Nominatim API per policy
HEADERS = {"User-Agent": "LocationGraphScript/1.0 (your_email@example.com)"}
NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"


def haversine(coord1, coord2):
    """
    Calculate the great-circle distance between two points on the Earth surface.
    :param coord1: (lat, lon)
    :param coord2: (lat, lon)
    :return: distance in meters
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def get_importance(lat, lon, cache):
    """
    Query Nominatim reverse geocoding for importance. Cache results to minimize calls.
    """
    key = (lat, lon)
    if key in cache:
        return cache[key]
    params = {"format": "json", "lat": lat, "lon": lon, "zoom": 18, "addressdetails": 0}
    resp = requests.get(NOMINATIM_URL, params=params, headers=HEADERS)
    if resp.status_code == 200:
        data = resp.json()
        importance = data.get("importance", None)
    else:
        importance = None
    # respect rate limit
    time.sleep(1)
    cache[key] = importance
    return importance


def main():
    # Load elevation CSV
    df = pd.read_csv("dataset/final/final_elevation_filtered.csv")
    df = df.dropna(subset=["location_name", "location_lat", "location_lon", "altitude"])

    # Unique nodes
    nodes = df[["location_name", "location_lat", "location_lon", "altitude"]]
    nodes = nodes.drop_duplicates().reset_index(drop=True)
    coords = list(zip(nodes["location_lat"], nodes["location_lon"]))
    node_names = nodes["location_name"].tolist()

    # Spatial index for nearest-neighbor lookup
    tree = cKDTree(coords)

    # Build graph
    G = nx.Graph()
    for _, row in nodes.iterrows():
        G.add_node(
            row["location_name"],
            latitude=row["location_lat"],
            longitude=row["location_lon"],
            altitude=row["altitude"],
        )

    raw_folder = "dataset/raw"
    for fname in os.listdir(raw_folder):
        if not fname.lower().endswith(".json"):
            continue
        path = os.path.join(raw_folder, fname)
        try:
            trails = gpd.read_file(path)
        except Exception:
            continue
        for _, feature in trails.iterrows():
            geom = feature.geometry
            if geom is None:
                continue
            if isinstance(geom, LineString):
                lines = [geom]
            elif isinstance(geom, MultiLineString):
                lines = list(geom.geoms)
            else:
                lines = []
            for line in lines:
                pts = list(line.coords)
                for i in range(len(pts) - 1):
                    lon1, lat1 = pts[i]
                    lon2, lat2 = pts[i + 1]
                    _, i1 = tree.query((lat1, lon1))
                    _, i2 = tree.query((lat2, lon2))
                    n1 = node_names[i1]
                    n2 = node_names[i2]
                    if n1 == n2:
                        continue
                    alt1 = G.nodes[n1]["altitude"]
                    alt2 = G.nodes[n2]["altitude"]
                    alt_diff = abs(alt1 - alt2)
                    dist = haversine((lat1, lon1), (lat2, lon2))
                    if G.has_edge(n1, n2):
                        existing = G[n1][n2]
                        if dist < existing.get("distance", float("inf")):
                            G[n1][n2].update(
                                {"distance": dist, "altitude_diff": alt_diff}
                            )
                    else:
                        G.add_edge(n1, n2, distance=dist, altitude_diff=alt_diff)

    # Prepare node CSV
    cache = {}
    nodes_out = []
    for n, data in G.nodes(data=True):
        lat = data["latitude"]
        lon = data["longitude"]
        imp = get_importance(lat, lon, cache)
        nodes_out.append(
            {
                "name": n,
                "latitude": lat,
                "longitude": lon,
                "altitude": data["altitude"],
                "importance": imp,
            }
        )
    pd.DataFrame(nodes_out).to_csv("nodes.csv", index=False)
    print("Saved nodes.csv")

    # Prepare edge CSV
    edges_out = []
    for u, v, ed in G.edges(data=True):
        edges_out.append(
            {
                "source": u,
                "target": v,
                "distance": ed["distance"],
                "altitude_diff": ed["altitude_diff"],
            }
        )
    pd.DataFrame(edges_out).to_csv("edges.csv", index=False)
    print("Saved edges.csv")


if __name__ == "__main__":
    main()
