import requests, json, random
import numpy as np
import pandas as pd
import rasterio
import os
import geopandas as gpd
import matplotlib.pyplot as plt
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from shapely.geometry import LineString, MultiLineString
from utils import json_to_geodataframe, plot_profiles
from tqdm import tqdm
import py3dep
from requests.exceptions import RetryError
import time

CRS = "EPSG:4269"
get_url = 'https://levees.sec.usace.army.mil:443/api-local/system-categories/usace-nonusace'

# Define a circuit breaker function
def circuit_breaker(max_failures, reset_time):
    def decorator(func):
        failures = 0

        def wrapper(*args, **kwargs):
            nonlocal failures
            try:
                response = func(*args, **kwargs)
                failures = 0  # Reset on success
                return response
            except RetryError as e:
                failures += 1
                if failures >= max_failures:
                    print(f"Max failures reached: {failures}. Entering cooldown period.")
                    time.sleep(reset_time)
                    failures = 0  # Reset after cooldown
                raise e
        return wrapper
    return decorator

# Apply the circuit breaker to the request function
@circuit_breaker(max_failures=3, reset_time=60)
def get_dem_vrt_with_retry(*args, **kwargs):
    return py3dep.get_dem_vrt(*args, **kwargs)

def requests_retry_session(retries=5, backoff_factor=1, status_forcelist=(500, 502, 504), session=None):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def get_usace_system_ids(url):
    try:
        response = requests_retry_session().get(url)
        if response.status_code == 200 and 'USACE' in response.json() and isinstance(response.json()['USACE'], str):
            return json.loads(response.json()['USACE'])
        else:
            print("Error: 'USACE' key issue or request failed with status code:", response.status_code)
            return []
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)
        return []

def process_system_ids(system_ids, show_plot=False, n_system_ids=None, try_system_ids=None):
    valid_system_ids, invalid_system_ids, contains_zero_ids, large_offset_ids = [], [], [], []
    n_system_ids = n_system_ids or len(system_ids)
    try_system_ids = try_system_ids or system_ids  # Use supplied list or default to all system_ids
    elevation_data_list = []
    for system_id in tqdm(try_system_ids[:n_system_ids], desc="Processing system IDs"):
        if system_id not in system_ids:
            print(f"System ID {system_id} not in the provided system_ids list.")
            continue
        profile_data, profile_gdf, skip_reason = get_profile_data(system_id)
        if skip_reason == 'contains_zero':
            contains_zero_ids.append(system_id)
            continue
        elif profile_data:
            profile_gdf['source'] = 'nld'
            profile_gdf['elevation'] = profile_gdf['elevation'] * .3048
            elevation_data_full = get_elevation_data(profile_gdf, system_id)  # Pass system_id here
            if elevation_data_full:
                elevation_data_full['source'] = 'tep'
                elevation_data_full = clip_to_nld_extent(elevation_data_full, profile_gdf)
                mean_elevation_3dep = elevation_data_full['elevation'].mean()
                mean_elevation_nld = profile_gdf['elevation'].mean()
                offset = abs(mean_elevation_3dep - mean_elevation_nld)
                if offset > 10:
                    large_offset_ids.append(system_id)
                    continue
                if show_plot:
                    plot_elevation_data(elevation_data_full, profile_gdf)
                valid_system_ids.append(system_id)
                elevation_data_list.append(elevation_data_full)
                elevation_data_list.append(profile_gdf)
            else:
                invalid_system_ids.append(system_id)
    save_skipped_ids(contains_zero_ids, 'contains_0.txt')
    save_skipped_ids(large_offset_ids, 'large_offset.txt')
    return elevation_data_list

def get_profile_data(system_id):
    try:
        url = f'https://levees.sec.usace.army.mil:443/api-local/system/{system_id}/route'
        response = requests_retry_session().get(url)
        if response.status_code == 200:
            profile_data = response.json()
            if profile_data is None or not profile_data:  # Check if profile_data is None or empty
                return None, None, 'empty'
            profile_gdf = json_to_geodataframe(profile_data).to_crs(CRS)
            if profile_gdf.empty:
                return None, None, 'empty'
            if (profile_gdf['elevation'].astype(float) == 0).any():
                return None, None, 'contains_zero'
            profile_gdf['system_id'] = system_id  # Add system_id as a column
            return profile_data, profile_gdf, None
    except Exception as e:
        print(f"Failed to get profile data for system ID: {system_id}: {e}")
    return None, None, None

def clip_to_nld_extent(elevation_data_3dep, profile_gdf_nld):
    # Get the extent of the NLD profile data
    nld_bounds = profile_gdf_nld.total_bounds
    
    # Clip the 3DEP data to the NLD extent
    elevation_data_3dep_clipped = elevation_data_3dep.cx[nld_bounds[0]:nld_bounds[2], nld_bounds[1]:nld_bounds[3]]
    
    return elevation_data_3dep_clipped

def get_elevation_data(profile_gdf, system_id):
    coords_list = list(zip(profile_gdf.geometry.x, profile_gdf.geometry.y))
    bounds = profile_gdf.total_bounds
    vrt_path = f"temp_{system_id}.vrt"  # Temporary VRT file name
    tiff_dir = "cache"  # Directory to save the TIFF file, ensure this directory exists
    
    # Download DEM data as a VRT file
    try:
        get_dem_vrt_with_retry(bounds, resolution=1, vrt_path=vrt_path, tiff_dir=tiff_dir, crs=CRS)
    except RetryError as e:
        print(f"An error occurred: {e}")
        return None
    
    # Sample elevation points from the VRT file
    with rasterio.open(vrt_path) as src:
        elevations = [next(src.sample([(x, y)]))[0] for x, y in coords_list]  # Correctly extract elevation values
    
    # Clean up: remove the VRT file to free up memory
    os.remove(vrt_path)
    
    # Prepare the elevation data as a GeoDataFrame
    coords_df = pd.DataFrame(coords_list, columns=['x', 'y']).assign(elevation=elevations)
    elevation_data_full = gpd.GeoDataFrame(coords_df, geometry=gpd.points_from_xy(coords_df['x'], coords_df['y']), crs=CRS)
    elevation_data_full = elevation_data_full.join(profile_gdf.set_index('geometry')['distance_along_track'], on='geometry').reset_index(drop=True)
    elevation_data_full['system_id'] = system_id  # Add system_id as a column
    return elevation_data_full

def plot_elevation_data(elevation_data_full, profile_gdf):
    plt.plot(elevation_data_full['distance_along_track'], elevation_data_full['elevation'], 'rx--', label='3DEP Profile', markersize=5)
    plt.plot(profile_gdf['distance_along_track'], profile_gdf['elevation'], 'bo-', label='NLD Profile', markersize=5)
    plt.title('Elevation Profiles Comparison')
    plt.xlabel('Distance Along Track (m)')
    plt.ylabel('Elevation (m)')
    plt.legend()
    plt.grid(True)
    plt.show()

def save_system_ids(valid_system_ids, invalid_system_ids):
    with open('valid_system_ids.txt', 'w') as f:
        f.writelines(f"{id}\n" for id in valid_system_ids)
    with open('invalid_system_ids.txt', 'w') as f:
        f.writelines(f"{id}\n" for id in invalid_system_ids)

def save_skipped_ids(skipped_ids, filename):
    with open(filename, 'w') as f:
        f.writelines(f"{id}\n" for id in skipped_ids)

if __name__ == "__main__":
    system_id_to_try = [2105000003,
                        2205000002,
                        2205000004,
                        2205000012]
    usace_system_ids = get_usace_system_ids(get_url)
    df_list = process_system_ids(usace_system_ids, show_plot=False, try_system_ids=system_id_to_try)
    df = pd.concat(df_list)
    df = gpd.GeoDataFrame(df, geometry='geometry', crs=CRS)

    # Assuming system_id is numeric without leading zeros; adjust as necessary
    df = df.astype({
        'x': 'float64',
        'y': 'float64',
        'elevation': 'float64',
        'distance_along_track': 'float64',
        'system_id': 'int64', 
        'source': 'category',
    })
    print(df.system_id.unique())
    df.to_parquet('elevation_data.parquet')

# %%
