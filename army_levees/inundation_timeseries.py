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