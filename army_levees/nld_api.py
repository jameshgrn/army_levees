#%%
import requests
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pyproj import Transformer
import py3dep
from shapely.geometry import LineString, MultiLineString
from utils import json_to_geodataframe, get_request, plot_profiles
# Set CRS variable
CRS = "EPSG:26914"

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

for system_id in usace_system_ids[15:30]:
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
            profile_gdf = profile_gdf.to_crs(CRS)
            nld_spacing = np.round(profile_gdf.distance_along_track.diff().mean(), 0)

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
    
    zipped_coords = zip(profile_gdf.geometry.x, profile_gdf.geometry.y)
    # Convert zipped coordinates to a list of tuples
    coords_list = list(zipped_coords)
    elevations = py3dep.elevation_bycoords(coords_list, crs=CRS)
    # Create a DataFrame from coords_list
    coords_df = pd.DataFrame(coords_list, columns=['x', 'y'])

    # Add the elevations list as a new column to this DataFrame
    coords_df['elevation'] = elevations

    # Convert the DataFrame to a GeoDataFrame
    elevation_data_full = gpd.GeoDataFrame(coords_df, geometry=gpd.points_from_xy(coords_df['x'], coords_df['y']), crs=CRS)

    # Set geometry as index for both GeoDataFrames
    profile_gdf.set_index('geometry', inplace=True)
    elevation_data_full.set_index('geometry', inplace=True)

    # Join 'distance_along_track' from profile_gdf to elevation_data_full
    # Since the indexes are geometries, this operation aligns matching geometries
    elevation_data_full = elevation_data_full.join(profile_gdf['distance_along_track'], how='left')

    # Reset index if needed to make 'geometry' a column again
    elevation_data_full.reset_index(inplace=True)
    profile_gdf.reset_index(inplace=True)
    try:
        elevation_data_full = elevation_data_full.drop(['name'], axis=1)
    except:
        print('No name column')
    elevation_data_full = gpd.GeoDataFrame(elevation_data_full, geometry='geometry', crs=CRS)
    profile_gdf.elevation = profile_gdf.elevation.astype(float)
    profile_gdf.distance_along_track = profile_gdf.distance_along_track.astype(float)
    elevation_data_full.elevation = elevation_data_full.elevation.astype(float)
    elevation_data_full.rename(columns={'distance': 'distance_along_track'}, inplace=True)
    elevation_data_full.distance_along_track = elevation_data_full.distance_along_track.astype(float)
    
    # Check if elevation values between profiles are significantly different
    max_profile_elevation = profile_gdf['elevation'].max()
    max_elevation_data_full = elevation_data_full['elevation'].max()
    if abs(max_profile_elevation - max_elevation_data_full) > 20:  # Arbitrary threshold to check significant difference
        # Assuming the larger value is in feet, convert it to meters
        if max_profile_elevation > max_elevation_data_full:
            profile_gdf['elevation'] = profile_gdf['elevation'] * 0.3048  # Convert feet to meters
    
    # elevation_data_full['distance_along_track'] = elevation_data_full['distance_along_track'] * .3048
    # profile_gdf['distance_along_track'] = profile_gdf['distance_along_track'] * .3048
    valid_system_ids.append(system_id)
    plt.plot(elevation_data_full['distance_along_track'], elevation_data_full['elevation'], label='3DEP Profile', color='red', marker='x', linestyle='--', markersize=5)
    plt.plot(profile_gdf['distance_along_track'], profile_gdf['elevation'], label='NLD Profile', color='blue', marker='o', linestyle='-', markersize=5)
    plt.title('Elevation Profiles Comparison')
    plt.xlabel('Distance Along Track (m)')
    plt.ylabel('Elevation (m)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Save valid and invalid system IDs to files
with open('valid_system_ids.txt', 'w') as f:
    for system_id in valid_system_ids:
        f.write(f"{system_id}\n")

with open('invalid_system_ids.txt', 'w') as f:
    for system_id in invalid_system_ids:
        f.write(f"{system_id}\n")

# %%
source_crs = "EPSG:4979"
target_crs = "EPSG:5498"
transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

def transform_elevation(row):
    geom = row['geometry']
    elevation = row['elevation']
    # Use the centroid for MultiLineString or the geometry itself if it's a Point
    representative_point = geom.centroid if isinstance(geom, MultiLineString) else geom
    # Transform the elevation value using the transformer
    new_elevation = transformer.transform(representative_point.x, representative_point.y, elevation)
    return new_elevation[2]  # The third element is the transformed elevation
    
# Apply the function to each row in the DataFrame
elevation_data_full['transformed_elevation'] = elevation_data_full.apply(transform_elevation, axis=1)


# %%
