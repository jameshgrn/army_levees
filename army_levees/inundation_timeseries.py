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
from shapely.geometry import Polygon
import eemont
import py3dep
from requests.exceptions import RetryError
import time
import ee
import geemap
# Initialize the Earth Engine library.
ee.Authenticate()
ee.Initialize()

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
    
def download_usace_system_ids(url, filepath='usace_system_ids.json'):
    # Check if the file already exists
    if not os.path.exists(filepath):
        try:
            # Make a request to download the data
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad responses
            # Save the data to a file
            with open(filepath, 'w') as file:
                json.dump(response.json()['USACE'], file)
            print(f"Downloaded USACE system IDs to {filepath}.")
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return None
    else:
        print(f"Using existing file for USACE system IDs: {filepath}.")

def load_usace_system_ids(filepath='usace_system_ids.json'):
    # Load the USACE system IDs from the file
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
            # Ensure 'USACE' key exists and contains a list
            if 'USACE' in data and isinstance(data['USACE'], list):
                return data['USACE']
            elif 'USACE' in data and isinstance(data['USACE'], str):
                # Attempt to parse the string as JSON
                try:
                    usace_data = json.loads(data['USACE'])
                    if isinstance(usace_data, list):
                        return usace_data
                    else:
                        print("Error: 'USACE' key does not contain a list.")
                        return None
                except json.JSONDecodeError:
                    print("Error decoding 'USACE' string as JSON.")
                    return None
            else:
                print("Error: 'USACE' key missing or not a list.")
                return None
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {filepath}")
        return None
    
def get_leveed_area(system_id):
    try:
        url = f'https://levees.sec.usace.army.mil:443/api-local/leveed-areas?system_id={system_id}&embed=geometry&format=geo'
        response = requests_retry_session().get(url)
        print(response)
        # if response.status_code == 200:
        #     profile_data = response.json()
        #     if profile_data is None or not profile_data:  # Check if profile_data is None or empty
        #         return None, None, 'empty'
        #     profile_gdf = json_to_geodataframe(profile_data).to_crs(CRS)
        #     if profile_gdf.empty:
        #         return None, None, 'empty'
        #     if (profile_gdf['elevation'].astype(float) == 0).any():
        #         return None, None, 'contains_zero'
        #     profile_gdf['system_id'] = system_id  # Add system_id as a column
        #     return profile_data, profile_gdf, None
    except Exception as e:
        print(f"Failed to get profile data for system ID: {system_id}: {e}")
    return None, None, None

if __name__ == '__main__':
    system_ids = load_usace_system_ids()
    usace_ids_url = 'https://levees.sec.usace.army.mil:443/api-local/system-categories/usace-nonusace'
    download_usace_system_ids(usace_ids_url)
    usace_system_ids = load_usace_system_ids()
    system_id = usace_system_ids[0]
    # print(usace_system_ids)
    url = f'https://levees.sec.usace.army.mil:443/api-local/leveed-areas?system_id={system_id}&embed=geometry&format=geo'
    response = requests_retry_session().get(url).json()
    # Assuming 'response' is your JSON response and contains geometry data
    coords = response[0]['geometry']['coordinates'][0][0]  # Assuming the first set of coordinates represents the polygon

    # # Create a Polygon geometry from the coordinates
    # polygon = Polygon(coords)

    # # Create the GeoDataFrame with this single Polygon geometry
    # gdf = gpd.GeoDataFrame(crs=CRS, geometry=[polygon])

    # # Now you can proceed to add other fields as before
    # for field, value in response[0].items():
    #     if field != 'geometry':  # Skip the geometry field
    #         # Ensure the value is treated as a single value for the entire column
    #         gdf[field] = [value] * len(gdf)

    # print(gdf)
    # gdf.to_crs(crs=CRS, inplace=True)
    # # Convert the 'stewardOrgIds' column's lists to strings
    # gdf['stewardOrgIds'] = gdf['stewardOrgIds'].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x)
    # # Convert GeoDataFrame to a GeoJSON string
    # geojson_str = gdf.to_json()

    # Convert GeoJSON to an Earth Engine Geometry
    # Remove the third element (elevation) from each coordinate pair
    coords_2d = [[lon, lat] for lon, lat, _ in coords]

    # Use the adjusted coordinates for the Earth Engine Polygon
    ee_geometry = ee.Geometry.Polygon([coords_2d])
    landsat8Sr = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')


    def mask_l8sr(image):
        # Bit 0 - Fill
        # Bit 1 - Dilated Cloud
        # Bit 2 - Cirrus
        # Bit 3 - Cloud
        # Bit 4 - Cloud Shadow
        qa_mask = image.select('QA_PIXEL').bitwiseAnd(int('11111', 2)).eq(0)
        saturation_mask = image.select('QA_RADSAT').eq(0)

        # Apply the scaling factors to the appropriate bands.
        optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
        thermal_bands = image.select('ST_B.*').multiply(0.00341802).add(149.0)

        # Replace the original bands with the scaled ones and apply the masks.
        return image.addBands(optical_bands, None, True) \
                    .addBands(thermal_bands, None, True) \
                    .updateMask(qa_mask) \
                    .updateMask(saturation_mask)
    def add_variables(image):
    # Compute time in fractional years since the epoch.
        date = image.date()
        years = date.difference(ee.Date('1970-01-01'), 'year')
        # Return the image with the added bands.
        return image \
            .addBands(image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')) \
            .addBands(ee.Image(years).rename('t').float()) \
            .addBands(ee.Image.constant(1))

    # Assuming landsat8Sr is your Landsat 8 Surface Reflectance ImageCollection,
    # roi is your region of interest as an ee.Geometry,
    # and mask_l8sr is the cloud masking function defined previously.
    roi = ee_geometry
    # Filter the Landsat 8 ImageCollection to the area of interest and date range,
    # apply cloud masking, and then add the NDVI, time, and constant bands.
    filtered_landsat = landsat8Sr \
        .filterBounds(roi) \
        .filterDate('2013-01-01', '2022-12-31') \
        .map(mask_l8sr) \
        .map(add_variables)
    
    ts = filtered_landsat.getTimeSeriesByRegion(reducer = [ee.Reducer.median()],
                             geometry = roi,
                             bands = ['NDVI'],
                             scale = 30)
    ts_df = geemap.ee_to_df(ts, columns = ['date', 'NDVI'])
    
    ts_df['date'] = pd.to_datetime(ts_df['date'])
    ts_df = ts_df.sort_values('date')
    #replace values below -1 with NaN
    ts_df['NDVI'] = ts_df['NDVI'].apply(lambda x: x if x > -1 else np.nan)
    ts_df_clean = ts_df.dropna()
    #plot timesries
    fig, ax = plt.subplots()
    ts_df_clean['NDVI'] = ts_df_clean['NDVI'].astype(float)
    ts_df_clean.plot(x='date', y='NDVI', ax=ax)
    plt.show()


    # for system_id in usace_system_ids:
    #     get_leveed_area(system_id)
    # if usace_system_ids is not None:
    #     # Proceed with processing the USACE system IDs
    #     print("USACE system IDs loaded successfully.")
    # else:
    #     print("Failed to load USACE system IDs.")

