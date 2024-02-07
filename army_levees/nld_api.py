#%%
import requests
import json
from get_3dep import process_segment
import numpy as np
import cartopy.feature as cfeatures
import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from cartopy.io.img_tiles import GoogleTiles
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.patheffects as pe
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from shapely.geometry import shape
from utils import json_to_geodataframe, plot_segment

from shapely.geometry import LineString, MultiLineString
import geopandas as gpd

import json
import sys
import topojson as tp


def get_request(url):
    headers = {
        'accept': 'application/json'
    }
    try:
        response = requests.get(url, headers=headers)
        #print(f"Full response: {response.text}")  # Print the full response
        response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code
        try:
            return response.json()  # This should return a dictionary if the response is JSON
        except json.JSONDecodeError:
            # If response is not JSON, print the response and return None
            print(f"Failed to decode JSON from response: {response.text}")
            return None
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")  # Print the HTTP error
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred: {conn_err}")  # Print the connection error
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error occurred: {timeout_err}")  # Print the timeout error
    except requests.exceptions.RequestException as req_err:
        print(f"Request exception occurred: {req_err}")  # Print any other request exception
    return None


def plot_profiles(profile_gdf, elevation_data_full):
    # Ensure data is sorted by 'distance_along_track' to represent downstream direction
    profile_gdf_sorted = profile_gdf.sort_values(by='distance_along_track')
    elevation_data_sorted = elevation_data_full.sort_values(by='distance_along_track')
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot profile_gdf
    plt.plot(profile_gdf_sorted['distance_along_track'], profile_gdf_sorted['elevation'], label='NLD Profile', color='blue', marker='o', linestyle='-', markersize=5)
    
    # Plot elevation_data_full
    plt.plot(elevation_data_sorted['distance_along_track'], elevation_data_sorted['elevation'], label='3DEP Profile', color='red', marker='x', linestyle='--', markersize=5)
    
    plt.title('Elevation Profiles Comparison')
    plt.xlabel('Distance Along Track (m)')
    plt.ylabel('Elevation (m)')
    plt.legend()
    plt.grid(True)
    plt.show()

get_url = 'https://levees.sec.usace.army.mil:443/api-local/system-categories/usace-nonusace'

# Perform a GET request
get_response = get_request(get_url)

# Check if 'USACE' key exists and if its value is a string
if 'USACE' in get_response and isinstance(get_response['USACE'], str):
    # Convert the string representation of the list into an actual list
    usace_system_ids = json.loads(get_response['USACE'])
else:
    print("The 'USACE' key is not present or not a string.")

# Create lists to store valid and invalid system IDs
valid_system_ids = []
invalid_system_ids = []

for system_id in usace_system_ids[2:5]:
    print(f"Processing system ID: {system_id}")
    
    # Get Measured Profile from NLD first
    try:
        print(f"Attempting to get profile data for system ID: {system_id}")  # Debug print
        profile_data = requests.get(f'https://levees.sec.usace.army.mil:443/api-local/system/{system_id}/route')
        
        print(f"Profile data received: {profile_data}")  # Debug print
        if profile_data is None:
            print(f"No profile data found for system ID: {system_id}")
            invalid_system_ids.append(system_id)
            continue  # Skip to the next iteration of the loop
        
        json_response = profile_data.json()
        if json_response is not None:
            profile_gdf = json_to_geodataframe(json_response)
            profile_gdf = profile_gdf.to_crs("EPSG:4269")
            # Check if all elevation values are 0
            if (profile_gdf['elevation'].astype(float) == 0).all():
                print(f"All elevation values are 0 for system ID: {system_id}")
                invalid_system_ids.append(system_id)
                continue
        else:
            print(f"No valid geometry found for system ID: {system_id}")
            invalid_system_ids.append(system_id)
            continue 
        
        # If the profile is valid, proceed to get the 3DEP data
        print("Attempting to get 3DEP data")  # Debug print
        geojson_download_url = f'https://levees.sec.usace.army.mil:443/api-local/geometries/query?type=centerline&systemId={system_id}&format=geo&props=true&coll=true'
        threedep = get_request(geojson_download_url)
        threedep_gdf = gpd.GeoDataFrame.from_features(threedep)
    except Exception as e:
        print(f"Failed to get profile data for system ID: {system_id}: {e}")
        invalid_system_ids.append(system_id)

    elevation_data = []
    for i, g in threedep_gdf.groupby('segmentId'):
        for i, row in g.iterrows():
            data = process_segment(row, crs="EPSG:4269")
            elevation_data.append(data)

    elevation_data_full = pd.concat(elevation_data)
    try:
        elevation_data_full = elevation_data_full.drop(['name'], axis=1)
    except:
        print('No name column')
    elevation_data_full = gpd.GeoDataFrame(elevation_data_full, geometry='geometry', crs="EPSG:4269")

    profile_gdf.elevation = profile_gdf.elevation.astype(float)
    profile_gdf.distance_along_track = profile_gdf.distance_along_track.astype(float)
    elevation_data_full.elevation = elevation_data_full.elevation.astype(float)
    elevation_data_full.rename(columns={'distance': 'distance_along_track'}, inplace=True)
    elevation_data_full.distance_along_track = elevation_data_full.distance_along_track.astype(float)
    # Save the profile data
    profile_gdf.to_file(f'/Users/jakegearon/CursorProjects/army_levees/data/NLD_profile_{system_id}.geojson', index=False)
    elevation_data_full.to_file(f'/Users/jakegearon/CursorProjects/army_levees/data/3DEP_{system_id}.geojson', index=False)
    # If the system ID is valid, add it to the valid_system_ids list
    valid_system_ids.append(system_id)
    plot_profiles(profile_gdf, elevation_data_full)



# Call the function with the GeoDataFrames
        

# Save valid and invalid system IDs to files
with open('valid_system_ids.txt', 'w') as f:
    for system_id in valid_system_ids:
        f.write(f"{system_id}\n")

with open('invalid_system_ids.txt', 'w') as f:
    for system_id in invalid_system_ids:
        f.write(f"{system_id}\n")


# %%
