
import requests, json, random
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from shapely.geometry import LineString, MultiLineString
from utils import json_to_geodataframe, plot_profiles
from tqdm import tqdm
import py3dep

CRS = "EPSG:4269"
get_url = 'https://levees.sec.usace.army.mil:443/api-local/system-categories/usace-nonusace'

def requests_retry_session(retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504), session=None):
    session = session or requests.Session()
    retry = Retry(total=retries, read=retries, connect=retries, 
                  backoff_factor=backoff_factor, status_forcelist=status_forcelist)
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

def process_system_ids(system_ids, show_plot=False, n_system_ids=None):
    valid_system_ids, invalid_system_ids, contains_zero_ids, large_offset_ids = [], [], [], []
    n_system_ids = n_system_ids or len(system_ids)
    elevation_data_list = []
    for system_id in tqdm(system_ids[:n_system_ids], desc="Processing system IDs"):
        profile_data, profile_gdf, skip_reason = get_profile_data(system_id)
        if skip_reason == 'contains_zero':
            contains_zero_ids.append(system_id)
        elif profile_data:
            profile_gdf['source'] = 'nld'
            profile_gdf['elevation'] = profile_gdf['elevation'] * .3048
            elevation_data_full = get_elevation_data(profile_gdf, system_id)  # Pass system_id here
            elevation_data_full['source'] = 'tep'
            mean_elevation_3dep = elevation_data_full['elevation'].mean()
            mean_elevation_nld = profile_gdf['elevation'].mean()
            offset = abs(mean_elevation_3dep - mean_elevation_nld)
            if offset > 10:
                large_offset_ids.append(system_id)
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

def get_elevation_data(profile_gdf, system_id):
    coords_list = list(zip(profile_gdf.geometry.x, profile_gdf.geometry.y))
    elevations = py3dep.elevation_bycoords(coords_list, crs=CRS, source='tep')
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
    usace_system_ids = get_usace_system_ids(get_url)
    df_list = process_system_ids(usace_system_ids, show_plot=False)
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
    df.to_parquet('elevation_data_full.parquet')

# %%
