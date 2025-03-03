import pandas as pd
import requests
import time
from tqdm import tqdm  # for progress bar


def get_elevation(lat, lon, api="open-elevation"):
    """
    Get elevation data for a given latitude and longitude.

    Parameters:
    lat (float): Latitude
    lon (float): Longitude
    api (str): API to use - "open-elevation" or "opentopodata"

    Returns:
    float: Elevation in meters
    """
    if api == "open-elevation":
        # Open-Elevation API
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data["results"][0]["elevation"]
        except Exception as e:
            print(f"Error with Open-Elevation API: {e}")
            # Fall back to opentopodata
            api = "opentopodata"

    if api == "opentopodata":
        # OpenTopoData API (alternative)
        url = f"https://api.opentopodata.org/v1/srtm30m?locations={lat},{lon}"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data["results"][0]["elevation"]
        except Exception as e:
            print(f"Error with OpenTopoData API: {e}")

    # Return None if both APIs fail
    return None


def add_elevation_data(input_file, output_file, batch_size=5):
    """
    Add elevation data to a CSV file containing latitude and longitude columns.

    Parameters:
    input_file (str): Path to input CSV file
    output_file (str): Path to output CSV file
    batch_size (int): Number of coordinates to process in one batch (to avoid API rate limits)
    """
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Check if altitude column already exists
    if "altitude" not in df.columns:
        # Create a new altitude column
        df["altitude"] = None

    # Process in batches to avoid rate limits
    for i in tqdm(range(0, len(df))):
        if pd.isna(df.at[i, "altitude"]):  # Only process rows without altitude
            lat = df.at[i, "latitude"]
            lon = df.at[i, "longitude"]

            # Get elevation data
            elevation = get_elevation(lat, lon)

            # Update DataFrame
            df.at[i, "altitude"] = elevation

            # Save intermediate results periodically
            if i % 10 == 0:
                df.to_csv(output_file, index=False)

            # Add a small delay to avoid overwhelming the API
            time.sleep(1)

    # Save the final result
    df.to_csv(output_file, index=False)
    print(f"Elevation data added and saved to {output_file}")


if __name__ == "__main__":
    input_file = "./dataset/final/final.csv"  # Change this to your input file path
    output_file = "./dataset/final/final_elevation.csv"  # Change this to your desired output file path

    add_elevation_data(input_file, output_file)
