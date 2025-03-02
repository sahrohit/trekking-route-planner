import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from collections import defaultdict
import requests
import time
import os


def load_data(file_path=None, data_str=None):
    """
    Load data either from a file or from a string containing CSV data
    """
    if file_path and os.path.exists(file_path):
        return pd.read_csv(file_path)
    elif data_str:
        # Clean up CSV string if needed (remove asterisks from column names)
        clean_lines = []
        lines = data_str.strip().split("\n")
        header = lines[0].replace("*", "")
        clean_lines.append(header)
        clean_lines.extend(lines[1:])

        clean_csv = "\n".join(clean_lines)
        return pd.read_csv(pd.StringIO(clean_csv))
    else:
        raise ValueError("Either file_path or data_str must be provided")


def preprocess_data(df):
    """
    Preprocess the data by cleaning columns and preparing for clustering
    """
    # Clean column names by removing asterisks if present
    df.columns = [col.replace("*", "") for col in df.columns]

    # Ensure longitude and latitude are numeric
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")

    # Drop rows with missing coordinates
    df = df.dropna(subset=["longitude", "latitude"])

    # Add a unique identifier if not present
    if "id" not in df.columns:
        df["id"] = range(len(df))

    return df


def cluster_locations(df, eps_km=0.5, min_samples=1):
    """
    Cluster locations based on their coordinates using DBSCAN

    Parameters:
    - df: DataFrame with 'longitude' and 'latitude' columns
    - eps_km: Maximum distance between two points (in km) to be considered in the same cluster
    - min_samples: Minimum number of points to form a dense region

    Returns:
    - DataFrame with an additional 'cluster' column
    """
    # Extract coordinates for clustering
    coords = df[["latitude", "longitude"]].values

    # Use DBSCAN for clustering
    kms_per_radian = 6371.0  # Earth's radius in km
    epsilon = eps_km / kms_per_radian  # Convert km to radians

    db = DBSCAN(
        eps=epsilon, min_samples=min_samples, algorithm="ball_tree", metric="haversine"
    )
    df["cluster"] = db.fit_predict(np.radians(coords))

    return df


def merge_location_data(df):
    """
    Merge location data based on clusters

    Parameters:
    - df: DataFrame with 'cluster' column from DBSCAN

    Returns:
    - DataFrame with merged location data
    """
    merged_locations = []

    # Group by cluster
    for cluster_id, group in df.groupby("cluster"):
        if cluster_id == -1:  # Noise points (not assigned to any cluster)
            # Keep each noise point as is
            for _, row in group.iterrows():
                merged_locations.append(row.to_dict())
        else:
            # For actual clusters, merge the data
            merged_location = {}

            # Calculate the centroid
            merged_location["latitude"] = group["latitude"].mean()
            merged_location["longitude"] = group["longitude"].mean()

            # Use the most common location name or the first one if all are different
            if "location_name" in group.columns:
                location_names = group["location_name"].dropna().unique()
                if len(location_names) > 0:
                    merged_location["location_name"] = location_names[0]

            # Similarly for road name
            if "road" in group.columns:
                road_names = group["road"].dropna().unique()
                if len(road_names) > 0:
                    merged_location["road"] = road_names[0]

            # For address and other fields, use the most common value
            for col in df.columns:
                if (
                    col not in ["latitude", "longitude", "id", "cluster"]
                    and col not in merged_location
                ):
                    values = group[col].dropna().unique()
                    if len(values) > 0:
                        # Try to find the most frequent value
                        value_counts = group[col].value_counts()
                        if not value_counts.empty:
                            merged_location[col] = value_counts.index[0]

            # Count how many points were merged
            merged_location["merged_point_count"] = len(group)

            # Store original IDs that were merged
            merged_location["original_ids"] = list(group["id"])

            merged_locations.append(merged_location)

    return pd.DataFrame(merged_locations)


def fetch_osm_details(lat, lon):
    """
    Fetch location details from OpenStreetMap's Nominatim API

    Parameters:
    - lat: Latitude
    - lon: Longitude

    Returns:
    - Dictionary of location details or None if API call fails
    """
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
        headers = {
            "User-Agent": "LocationDataMerger/1.0"  # Required by OSM Nominatim API
        }

        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching OSM data: {response.status_code}")
            return None
    except Exception as e:
        print(f"Exception in API call: {e}")
        return None


def enrich_with_osm_data(df, sample_size=None):
    """
    Enrich the dataset with additional information from OSM

    Parameters:
    - df: DataFrame with 'latitude' and 'longitude' columns
    - sample_size: Optional, number of locations to enrich (to avoid API rate limits)

    Returns:
    - Enriched DataFrame
    """
    if sample_size and sample_size < len(df):
        df_to_enrich = df.sample(sample_size).copy()
    else:
        df_to_enrich = df.copy()

    for idx, row in df_to_enrich.iterrows():
        osm_data = fetch_osm_details(row["latitude"], row["longitude"])
        if osm_data:
            # Extract and add relevant OSM data
            if "address" in osm_data:
                for key, value in osm_data["address"].items():
                    df_to_enrich.at[idx, f"osm_{key}"] = value

            if "display_name" in osm_data:
                df_to_enrich.at[idx, "osm_display_name"] = osm_data["display_name"]

        # Be nice to the OSM API
        time.sleep(1)

    return df_to_enrich


def main(
    input_file=None,
    input_data=None,
    output_file="merged_locations.csv",
    eps_km=0.5,
    min_samples=1,
    enrich_with_osm=False,
    osm_sample_size=10,
):
    """
    Main function to load, cluster, merge, and optionally enrich location data

    Parameters:
    - input_file: Path to the input CSV file
    - input_data: String containing CSV data (alternative to input_file)
    - output_file: Path to save the output CSV
    - eps_km: Clustering parameter - maximum distance in km to be considered same location
    - min_samples: Clustering parameter - minimum points to form a cluster
    - enrich_with_osm: Whether to enrich the data with OSM API
    - osm_sample_size: Number of locations to enrich if enrich_with_osm is True
    """
    # Load and preprocess the data
    df = load_data(file_path=input_file, data_str=input_data)
    df = preprocess_data(df)

    print(f"Loaded {len(df)} locations")

    # Cluster the locations
    df = cluster_locations(df, eps_km=eps_km, min_samples=min_samples)

    # Count the number of clusters
    n_clusters = len(df["cluster"].unique())
    if -1 in df["cluster"].unique():  # -1 is for noise points
        n_clusters -= 1

    print(f"Found {n_clusters} clusters and {sum(df['cluster'] == -1)} noise points")

    # Merge the location data
    merged_df = merge_location_data(df)

    print(f"Merged into {len(merged_df)} locations")

    # Optionally enrich with OSM data
    if enrich_with_osm:
        print(f"Enriching data with OSM (sample size: {osm_sample_size})")
        merged_df = enrich_with_osm_data(merged_df, sample_size=osm_sample_size)

    # Save the merged data
    merged_df.to_csv(output_file, index=False)
    print(f"Saved merged locations to {output_file}")

    return merged_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge location data based on proximity and other factors"
    )
    parser.add_argument("--input", help="Path to the input CSV file")
    parser.add_argument(
        "--output", default="merged_locations.csv", help="Path to save the output CSV"
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.5,
        help="Maximum distance (km) to be considered same location",
    )
    parser.add_argument(
        "--min-samples", type=int, default=1, help="Minimum points to form a cluster"
    )
    parser.add_argument("--enrich", action="store_true", help="Enrich with OSM data")
    parser.add_argument(
        "--osm-sample", type=int, default=10, help="Sample size for OSM enrichment"
    )

    args = parser.parse_args()

    main(
        input_file=args.input,
        output_file=args.output,
        eps_km=args.eps,
        min_samples=args.min_samples,
        enrich_with_osm=args.enrich,
        osm_sample_size=args.osm_sample,
    )
