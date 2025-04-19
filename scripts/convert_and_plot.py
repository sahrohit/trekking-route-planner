#!/usr/bin/env python3
"""
Script to merge elevation data with all trail JSON files in `dataset/raw`, generate a graph where
nodes represent locations with altitude metadata and edges with distance & altitude_diff,
and plot the result as an interactive Folium map saved to `map.html`.
"""

import os
import json
import math
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from scipy.spatial import cKDTree
import networkx as nx
import folium


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


def plot_map(G, map_path="map.html"):
    """
    Plot nodes and edges of graph G on an interactive Folium map.
    Saves to HTML file at map_path.
    """
    # Compute center of map
    latitudes = [data["latitude"] for _, data in G.nodes(data=True)]
    longitudes = [data["longitude"] for _, data in G.nodes(data=True)]
    center = (sum(latitudes) / len(latitudes), sum(longitudes) / len(longitudes))

    m = folium.Map(location=center, zoom_start=12)

    # Add nodes
    for name, data in G.nodes(data=True):
        folium.CircleMarker(
            location=(data["latitude"], data["longitude"]),
            radius=5,
            popup=f"{name}<br>Altitude: {data['altitude']} m",
            fill=True,
            color="red",
            fill_opacity=0.7,
        ).add_to(m)

    # Add edges
    for u, v, edata in G.edges(data=True):
        lat1, lon1 = G.nodes[u]["latitude"], G.nodes[u]["longitude"]
        lat2, lon2 = G.nodes[v]["latitude"], G.nodes[v]["longitude"]
        folium.PolyLine(
            locations=[(lat1, lon1), (lat2, lon2)],
            weight=2,
            opacity=0.6,
            popup=f"Distance: {edata['distance']:.1f} m<br>Alt diff: {edata['altitude_diff']} m",
        ).add_to(m)

    m.save(map_path)
    print(f"Interactive map saved to {map_path}")


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
                    _, idx1 = tree.query((lat1, lon1))
                    _, idx2 = tree.query((lat2, lon2))
                    n1 = node_names[idx1]
                    n2 = node_names[idx2]
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

    # Plot interactive map
    plot_map(G, map_path="map.html")


if __name__ == "__main__":
    main()
