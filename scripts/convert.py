#!/usr/bin/env python3
"""
Script to merge elevation data with all trail JSON files in `dataset/raw` and generate a graph where
nodes represent locations (by name) with altitude metadata, and edges represent
trail segments weighted by altitude difference and geographic distance.
"""

import os
import json
import math
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from scipy.spatial import cKDTree
import networkx as nx


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


def main():
    # Load elevation CSV
    df = pd.read_csv("dataset/final/final_elevation_filtered.csv")
    df = df.dropna(subset=["location_name", "location_lat", "location_lon", "altitude"])

    # Build unique node list
    nodes = (
        df[["location_name", "location_lat", "location_lon", "altitude"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    coords = list(zip(nodes["location_lat"], nodes["location_lon"]))
    node_names = nodes["location_name"].tolist()

    # Spatial index for nearest-neighbor lookup
    tree = cKDTree(coords)

    # Initialize graph
    G = nx.Graph()
    for _, row in nodes.iterrows():
        G.add_node(
            row["location_name"],
            latitude=row["location_lat"],
            longitude=row["location_lon"],
            altitude=row["altitude"],
        )

    # Process each JSON (GeoJSON) in dataset/raw
    raw_folder = "dataset/raw"
    for fname in os.listdir(raw_folder):
        if not fname.lower().endswith(".json"):
            continue
        path = os.path.join(raw_folder, fname)
        try:
            trails = gpd.read_file(path)
        except Exception as e:
            print(f"Skipping {fname}: cannot read as GeoJSON ({e})")
            continue

        for _, feature in trails.iterrows():
            geom = feature.geometry
            if geom is None:
                continue
            # Handle single and multi-line geometries correctly
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
                    # Snap to nearest nodes
                    _, idx1 = tree.query((lat1, lon1))
                    _, idx2 = tree.query((lat2, lon2))
                    n1 = node_names[idx1]
                    n2 = node_names[idx2]

                    # Skip self-loops
                    if n1 == n2:
                        continue

                    alt1 = G.nodes[n1]["altitude"]
                    alt2 = G.nodes[n2]["altitude"]
                    alt_diff = abs(alt1 - alt2)
                    dist = haversine((lat1, lon1), (lat2, lon2))

                    # Add or update edge with minimum distance
                    if G.has_edge(n1, n2):
                        existing = G[n1][n2]
                        if dist < existing.get("distance", float("inf")):
                            G[n1][n2].update(
                                {"distance": dist, "altitude_diff": alt_diff}
                            )
                    else:
                        G.add_edge(n1, n2, distance=dist, altitude_diff=alt_diff)

    # Export to JSON
    output = {
        "nodes": [{"name": n, **G.nodes[n]} for n in G.nodes],
        "edges": [
            {"source": u, "target": v, **edata} for u, v, edata in G.edges(data=True)
        ],
    }
    with open("nodes_edges.json", "w") as f_out:
        json.dump(output, f_out, indent=2)
    print("Merged graph saved to nodes_edges.json")


if __name__ == "__main__":
    main()
