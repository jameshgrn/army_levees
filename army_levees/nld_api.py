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
import utm

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
    
def get_elevation_data_gee(profile_gdf, system_id):
    try:
        # Convert the GeoDataFrame to an Earth Engine Geometry.
        profile_geometry = geemap.geopandas_to_ee(profile_gdf, geodesic=False)
        
        # Filter the ImageCollection to the first image intersecting the profile geometry.
        dataset = ee.ImageCollection('USGS/3DEP/1m').mosaic()
        
        # Attempt to sample the elevation values along the profile geometry.
        elevations = dataset.sampleRegions(collection=profile_geometry, scale=1, geometries=True, projection=CRS)
        # Check if the query returned too many elements
        if len(elevations.getInfo()['features']) == 0:
            return None
        else:   
            # Convert the sampled elevations to a GeoDataFrame.
            elevation_gdf = geemap.ee_to_gdf(elevations)
            elevation_gdf['system_id'] = system_id  # Add system_id as a column
            elevation_gdf.to_crs(CRS, inplace=True)
            return elevation_gdf
        
    except ee.ee_exception.EEException as e:
        print(f"Error querying Earth Engine: {e}")
        # Handle the exception, e.g., by splitting the query into smaller parts or returning None
        return None
    
import time
from functools import wraps
from requests.exceptions import ChunkedEncodingError

def retry_on_chunked_encoding_error(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except ChunkedEncodingError as e:
                    retries += 1
                    if retries == max_retries:
                        raise
                    print(f"ChunkedEncodingError occurred. Retrying in {delay} seconds... (Attempt {retries}/{max_retries})")
                    time.sleep(delay)
            return func(*args, **kwargs)
        return wrapper
    return decorator

def long2utm(longitude):
    """Calculate UTM zone number from longitude."""
    return int((longitude + 180) / 6) % 60 + 1

def get_hemisphere(latitude):
    """Determine the UTM hemisphere (N or S) from latitude."""
    return 'N' if latitude >= 0 else 'S'

def get_profile_data(system_id):
    try:
        url = f'https://levees.sec.usace.army.mil:443/api-local/system/{system_id}/route'
        response = requests_retry_session().get(url)
        if response.status_code == 200:
            profile_data = response.json()
            if profile_data is None or not profile_data:  # Check if profile_data is None or empty
                return None, None, 'empty'
            profile_gdf = json_to_geodataframe(profile_data) #gets transformed into 3857 in this function
            if profile_gdf.empty or profile_gdf.geometry.is_empty.any():
                print(f"Profile data for system ID {system_id} is empty or has invalid geometries.")
                return None, None, 'invalid_geometry'
            profile_gdf_utm_finding = profile_gdf.to_crs('EPSG:4326')
            SFlong = profile_gdf_utm_finding.geometry.x[0]
            SFlat = profile_gdf_utm_finding.geometry.y[0]
            zone_number = long2utm(SFlong)
            hemisphere = get_hemisphere(SFlat)

            # print(f"UTM Zone Number: {zone_number}, Hemisphere: {hemisphere}")
            epsg_code = 26900 + zone_number
            profile_gdf = profile_gdf.to_crs(f'EPSG:4269') #back to 4269
            profile_gdf['x'] = profile_gdf.geometry.x #these are now in 4269
            profile_gdf['y'] = profile_gdf.geometry.y #these are now in 4269
            #print(profile_gdf.crs)
            #find UTM zone here
            profile_gdf['system_id'] = system_id  # Add system_id as a column
            #check if all values are zero
            if profile_gdf['elevation'].eq(0).all():
                return None, None, 'contains_zero'
            return profile_data, profile_gdf, epsg_code
    except Exception as e:
        print(f"Failed to get profile data for system ID: {system_id}: {e}")
    return None, None, None

def clip_to_nld_extent(elevation_data_3dep, profile_gdf_nld):
    # Create a bounding box around the NLD profile
    nld_bounds = profile_gdf_nld.unary_union.envelope
    
    # Clip the 3DEP data to the bounding box of the NLD profile
    elevation_data_3dep_clipped = elevation_data_3dep[elevation_data_3dep.geometry.within(nld_bounds)]
    
    return elevation_data_3dep_clipped

@retry_on_chunked_encoding_error(max_retries=3, delay=2)
def get_elevation_data(profile_gdf, system_id, epsg_code):
        coords_list = list(zip(profile_gdf.geometry.x, profile_gdf.geometry.y)) #this is in 4269 btw
        bounds = profile_gdf.total_bounds #also in 4269
        vrt_path = f"temp_{system_id}.vrt"  # Temporary VRT file name
        tiff_dir = "cache"  # Directory to save the TIFF file, ensure this directory exists
        
        # Download DEM data as a VRT file
        try:
            get_dem_vrt_with_retry(bounds, resolution=1, vrt_path=vrt_path, tiff_dir=tiff_dir, crs=profile_gdf.crs) #this is in 4269
        except RetryError as e:
            print(f"An error occurred: {e}")
            return None
        
        # Sample elevation points from the VRT file
        with rasterio.open(vrt_path) as src:
            elevations = [next(src.sample([(x, y)]))[0] for x, y in coords_list]  # Correctly extract elevation values from the VRT file with 4269 crs
        # Clean up: remove the VRT file to free up memory
        os.remove(vrt_path)
        # os.remove("/Users/jakegearon/CursorProjects/army_levees/cache") #remove the cache folder
        # Prepare the elevation data as a GeoDataFrame
        coords_df = pd.DataFrame(coords_list, columns=['x', 'y']).assign(elevation=elevations) #this is in 4269
        elevation_data_full = gpd.GeoDataFrame(coords_df, geometry=gpd.points_from_xy(coords_df['x'], coords_df['y']), crs="EPSG:4269")
        # print(elevation_data_full.head())
        elevation_data_full = elevation_data_full.to_crs(f'EPSG:{epsg_code}') #convert to UTM
        # print(elevation_data_full.head())
        profile_gdf = profile_gdf.to_crs(f'EPSG:{epsg_code}') #convert to UTM
        
        #Print CRS for debugging
        # print("CRS for profile_gdf:", profile_gdf.crs)
        # print("CRS for elevation_data_full:", elevation_data_full.crs)

        # # Check if geometries are exactly the same
        # print("Checking geometry equality:")
        # print(profile_gdf.geometry.equals(elevation_data_full.geometry))

        # Perform the join operation
        elevation_data_full = elevation_data_full.join(profile_gdf.set_index('geometry')['distance_along_track'], on='geometry', how='left')
        # print(elevation_data_full.head())
        # Add system_id as a column
        elevation_data_full['system_id'] = system_id

        # Display the head of the dataframe to check results
        # print(elevation_data_full.head())
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

def save_elevation_data(elevation_data, system_id, epsg_code, source, directory='data'):
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    filename = os.path.join(directory, f"elevation_data_{system_id}_{source}_epsg{epsg_code}.parquet")
    elevation_data.to_parquet(filename)
    print(f"Saved elevation data for system ID {system_id} ({source}) to {filename}")

if __name__ == "__main__":
    usace_ids_url = 'https://levees.sec.usace.army.mil:443/api-local/system-categories/usace-nonusace'
    download_usace_system_ids(usace_ids_url)
    usace_system_ids = load_usace_system_ids()
    
    target_sample_count = 250
    successful_samples = 0
    max_attempts = 1000  # Limit the number of attempts to avoid infinite loops
    attempts = 0

    valid_ids = set()
    invalid_ids = set()
    
    # Load previously processed IDs if files exist
    try:
        with open('valid_system_ids.txt', 'r') as f:
            valid_ids = set(f.read().splitlines())
        with open('invalid_system_ids.txt', 'r') as f:
            invalid_ids = set(f.read().splitlines())
    except FileNotFoundError:
        pass

    with tqdm(total=target_sample_count, desc="Processing system IDs") as pbar:
        while successful_samples < target_sample_count and attempts < max_attempts:
            random_system_ids = np.random.choice(usace_system_ids, target_sample_count - successful_samples)
            for system_id in random_system_ids:
                if system_id in valid_ids or system_id in invalid_ids:
                    continue  # Skip already processed IDs
                
                attempts += 1
                if system_id not in usace_system_ids:
                    tqdm.write(f"System ID {system_id} not in the provided system_ids list.")
                    continue
                profile_data, profile_gdf, epsg_code = get_profile_data(system_id)
                if profile_data:
                    profile_gdf['source'] = 'nld'
                    profile_gdf['elevation'] = profile_gdf['elevation'] * .3048
                    try:
                        elevation_data_full = get_elevation_data(profile_gdf, system_id, epsg_code)
                        if elevation_data_full is None:
                            tqdm.write(f"Skipping system ID {system_id} due to missing elevation data.")
                            continue
                        
                        profile_gdf = profile_gdf.to_crs(f'EPSG:{epsg_code}')
                        
                        if not elevation_data_full.empty:
                            elevation_data_full['source'] = 'tep'
                            elevation_data_full = clip_to_nld_extent(elevation_data_full, profile_gdf)
                            mean_elevation_3dep = elevation_data_full['elevation'].mean()
                            mean_elevation_nld = profile_gdf['elevation'].mean()
                            tqdm.write(f"Mean elevation (3DEP): {mean_elevation_3dep:.2f} m")
                            tqdm.write(f"Mean elevation (NLD): {mean_elevation_nld:.2f} m")
                            
                            # Check for NaN values
                            if elevation_data_full['elevation'].isna().any() or profile_gdf['elevation'].isna().any():
                                tqdm.write(f"Skipping system ID {system_id} due to NaN values in elevation data.")
                                continue
                            
                            # Save the elevation data for this system
                            save_elevation_data(elevation_data_full, system_id, epsg_code, source='tep')
                            save_elevation_data(profile_gdf, system_id, epsg_code, source='nld')
                            
                            successful_samples += 1
                            pbar.update(1)
                            if successful_samples >= target_sample_count:
                                break
                        
                    except Exception as e:
                        tqdm.write(f"Error processing system ID {system_id}: {str(e)}")
                    valid_ids.add(system_id)
                else:
                    invalid_ids.add(system_id)

                # Periodically save processed IDs
                if attempts % 100 == 0:
                    save_system_ids(valid_ids, invalid_ids)

    # Save final set of processed IDs
    save_system_ids(valid_ids, invalid_ids)

    if successful_samples < target_sample_count:
        tqdm.write(f"Only {successful_samples} samples were successfully processed out of the requested {target_sample_count}.")

# %%
