import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, MultiLineString, LineString
from pyproj import Transformer
from py3dep import elevation_bycoords
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
from tqdm import tqdm
from shapely.geometry import Polygon
import eemont
import py3dep
from requests.exceptions import RetryError
import time
import ee
import geemap
ee.Authenticate()
ee.Initialize()

CRS = "EPSG:4269"

coefficients = {
    'itcps': ee.Image([0.0003, 0.0088, 0.0061, 0.0412, 0.0254, 0.0172]).multiply(10000),
    'slopes': ee.Image([0.8474, 0.8483, 0.9047, 0.8462, 0.8937, 0.9071])
}

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

# Function to extract coordinates from a Polygon's exterior
def extract_polygon_coords(polygon):
    if polygon.is_empty:
        return []
    else:
        # Extracting exterior coordinates of the polygon and formatting them as a list of lists
        return [list(polygon.exterior.coords)]



def get_leveed_area(system_id):
    try:
        url = f'https://levees.sec.usace.army.mil:443/api-local/leveed-areas?system_id={system_id}&embed=geometry&format=geo'
        response = requests_retry_session().get(url).json()
        coords = response[0]['geometry']['coordinates'][0][0]
        coords_2d = [[lon, lat] for lon, lat, _ in coords]
        return coords_2d
    except Exception as e:
        print(f"Failed to get profile data for system ID: {system_id}: {e}")
        return None

def get_profile_data(system_id):
    try:
        url = f'https://levees.sec.usace.army.mil:443/api-local/system/{system_id}/route'
        response = requests_retry_session().get(url)
        if response.status_code == 200:
            profile_data = response.json()
            profile_gdf = json_to_geodataframe(profile_data).to_crs(CRS)
            profile_gdf['system_id'] = system_id  # Add system_id as a column
            return profile_data, profile_gdf
    except Exception as e:
        print(f"Failed to get profile data for system ID: {system_id}: {e}")
        return None, None

def get_state_info(system_id):
    try:
        url = f'https://levees.sec.usace.army.mil:443/api-local/system/{system_id}/detail'
        response = requests_retry_session().get(url).json()
        states = response['states']
        print(states)
        return states
    except Exception as e:
        print(f"Failed to get state info for system ID: {system_id}: {e}")
    return None


def json_to_geodataframe(json_response):
    """
    Converts JSON response containing geometry arcs into a GeoDataFrame.
    
    Parameters:
    - json_response: dict, JSON object containing geometry information.
    
    Returns:
    - gdf: GeoDataFrame, contains columns for elevation, distance along track, and geometry points.
    """
    # Extract coordinates (longitude, latitude) and attributes (elevation, distance) from the first arc in the JSON response
    coords = [(arc[0], arc[1]) for arc in json_response['geometry']['arcs'][0]]
    elevations = [arc[2] for arc in json_response['geometry']['arcs'][0]]
    distances = [arc[3] for arc in json_response['geometry']['arcs'][0]]

    # Create a DataFrame from the extracted data
    df = pd.DataFrame({
        'elevation': elevations,  # Elevation data
        'distance_along_track': distances,  # Distance data
        'geometry': gpd.points_from_xy(*zip(*coords))  # Geometry data created from coordinates
    })

    # Convert the DataFrame to a GeoDataFrame, specifying the 'geometry' column and setting the coordinate reference system (CRS) to EPSG:3857
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:3857")
    return gdf
        
def read_and_parse_elevation_data(filepath, system_ids=None):
    import geopandas as gpd
    """
    Reads elevation data for specified system IDs from a Parquet file without loading the entire DataFrame into memory.

    Parameters:
    - filepath: str, path to the Parquet file containing elevation data.
    - system_ids: list of str, system IDs to filter the data. If None, all data is loaded.

    Returns:
    - DataFrame with parsed elevation data for the specified system IDs.
    """
    try:
        # If system_ids is provided, prepare a filter
        if system_ids:
            filters = [('system_id', 'in', system_ids)]
            # Use the 'columns' parameter if you want to load specific columns only, e.g., ['system_id', 'elevation']
            df = gpd.read_parquet(filepath, filters=filters)
        else:
            df = gpd.read_parquet(filepath)
    except Exception as e:
        print(f"Failed to read the Parquet file: {e}")
        return None

    # Further processing can be done here if needed
    return df

def plot_profiles(profile_gdf, elevation_data_full):
    # Sort data by 'distance_along_track'
    system_id = profile_gdf['system_id'].iloc[0]
    profile_gdf_sorted = profile_gdf.sort_values(by='distance_along_track')
    elevation_data_sorted = elevation_data_full.sort_values(by='distance_along_track')
    print(system_id)
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(profile_gdf_sorted['distance_along_track'], profile_gdf_sorted['elevation'], label='NLD Profile', color='blue', marker='o', linestyle='-', markersize=1)
    plt.plot(elevation_data_sorted['distance_along_track'], elevation_data_sorted['elevation'], label='3DEP Profile', color='red', marker='x', linestyle='--', markersize=1)
    plt.title(f'Elevation Profiles Comparison {system_id}')
    plt.xlabel('Distance Along Track (m)')
    plt.ylabel('Elevation (m)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to get and rename bands of interest from OLI.
def rename_oli(img):
    img = img.select(
        ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'pixel_qa'])
    return img.rename(
        ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'pixel_qa'])

# Function to get and rename bands of interest from ETM+.
def rename_etm(img):
    img = img.select(
        ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'pixel_qa'])
    return img.rename(
        ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'pixel_qa'])

def etm_to_oli(img):
    return img.select(['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2'])\
        .multiply(coefficients['slopes'])\
        .add(coefficients['itcps'])\
        .round()\
        .toShort()\
        .addBands(img.select('pixel_qa'))

def calc_ndvi(img):
    return img.normalizedDifference(['NIR', 'Red']).rename('NDVI')

def calc_mndwi(img):
    return img.normalizedDifference(['Green', 'SWIR1']).rename('MNDWI')

def calc_ndwi_ns(img):
    return img.expression(
        '(Green - a * NIR)/(Green + NIR)',
        {
            'Green': img.select('Green'),
            'NIR': img.select('NIR'),
            'a': 1.5
        }).rename('NDWI_NS')

def calc_ndsi_nw(img):
    return img.expression(
        '(NIR - SWIR1 - b)/(NIR + SWIR1)',
        {
            'SWIR1': img.select('SWIR1'),
            'NIR': img.select('NIR'),
            'b': .05
        }).rename('NDSI_NW')

#actual ANDWI
def calc_andwi(img):
        return img.expression(
        '(Blue + Green + Red - NIR - SWIR1 - SWIR2)/(Blue + Green + Red + NIR + SWIR1 + SWIR2)',
        {
            'SWIR1': img.select('SWIR1'),
            'SWIR2': img.select('SWIR2'),
            'NIR': img.select('NIR'),
            'Green': img.select('Green'),
            'Blue': img.select('Blue'),
            'Red': img.select('Red')
        }).rename('ANDWI')
        
def calc_embi(img):
    return img.expression(
        '((((SWIR1 - SWIR2 - NIR)/(SWIR1 + SWIR2 + NIR)) + 0.5) - ((Green - SWIR1)/(Green + SWIR1)) - 0.5)/((((SWIR1 - SWIR2 - NIR)/(SWIR1 + SWIR2 + NIR)) + 0.5) + ((Green - SWIR1)/(Green + SWIR1)) + 1.5)',
        {
            'SWIR1': img.select('SWIR1'),
            'SWIR2': img.select('SWIR2'),
            'NIR': img.select('NIR'),
            'Green': img.select('Green')
        }).rename('EMBI')
    

    
def fmask(img):
    cloud_shadow_bit_mask = 1 << 3
    clouds_bit_mask = 1 << 5
    qa = img.select('pixel_qa')
    mask = qa.bitwiseAnd(cloud_shadow_bit_mask)\
               .eq(0)\
               .And(qa.bitwiseAnd(clouds_bit_mask).eq(0))
    return img.updateMask(mask)

# Define function to prepare OLI images.
def prep_oli(img):
    orig = img
    img = rename_oli(img)
    img = fmask(img)
    #img = calc_ndvi(img)
    #img = calc_mndwi(img)
    img = calc_ndwi_ns(img)
    #img = calc_andwi(img)
    #img = calc_embi(img)
    return ee.Image(img.copyProperties(orig, orig.propertyNames()))

# Define function to prepare ETM+ images.
def prep_etm(img):
    orig = img
    img = rename_etm(img)
    img = fmask(img)
    img = etm_to_oli(img)
    #img = calc_ndvi(img)
    #img = calc_mndwi(img)
    img = calc_ndwi_ns(img)
    #img = calc_andwi(img)
    #img = calc_embi(img)
    return ee.Image(img.copyProperties(orig, orig.propertyNames()))

def prep_tm(img):
    orig = img
    img = rename_etm(img)
    img = fmask(img)
    img = etm_to_oli(img)
    #img = calc_ndvi(img)
    #img = calc_mndwi(img)
    img = calc_ndwi_ns(img)
    #img = calc_andwi(img)
    #img = calc_embi(img)
    return ee.Image(img.copyProperties(orig, orig.propertyNames()))

# Return the DN that maximizes interclass variance in B5 (in the region).
def otsu(histogram):
    counts = ee.Array(ee.Dictionary(histogram).get("histogram"))
    means = ee.Array(ee.Dictionary(histogram).get("bucketMeans"))
    size = means.length().get([0])
    total = counts.reduce(ee.Reducer.sum(), [0]).get([0])
    sum = means.multiply(counts).reduce(ee.Reducer.sum(), [0]).get([0])
    mean = sum.divide(total)

    indices = ee.List.sequence(1, size)

    # Compute between sum of squares, where each mean partitions the data.

    def func_xxx(i):
        aCounts = counts.slice(0, 0, i)
        aCount = aCounts.reduce(ee.Reducer.sum(), [0]).get([0])
        aMeans = means.slice(0, 0, i)
        aMean = (
            aMeans.multiply(aCounts)
            .reduce(ee.Reducer.sum(), [0])
            .get([0])
            .divide(aCount)
        )
        bCount = total.subtract(aCount)
        bMean = sum.subtract(aCount.multiply(aMean)).divide(bCount)
        return aCount.multiply(aMean.subtract(mean).pow(2)).add(
            bCount.multiply(bMean.subtract(mean).pow(2))
        )

    bss = indices.map(func_xxx)

    # Return the mean value corresponding to the maximum BSS.
    return means.sort(bss).get([-1])

def extract_water(image, geometry):
    histogram = image.select("N").reduceRegion(
        reducer=ee.Reducer.histogram(255, 2),
        geometry=geometry,
        scale=10,
        bestEffort=True,
    )
    threshold = otsu(histogram.get("N"))
    water = image.select("N").lt(threshold).selfMask()
    return water.set({"threshold": threshold})

def plot_water_area_time_series(collection, geometry_first, geometry_second, title_first, title_second):
    def calculate_area(image, geometry):
        water_image = extract_water(image, geometry)
        area = water_image.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geometry,
            scale=30,
            maxPixels=1e9
        )
        return image.set('water_area', area.get('N'))

    # Use a lambda function to pass both the image and the geometry to calculate_area
    water_areas_first = collection.map(lambda image: calculate_area(image, geometry_first))
    water_areas_second = collection.map(lambda image: calculate_area(image, geometry_second))

    # Convert to DataFrame
    water_area_list_first = water_areas_first.aggregate_array('water_area').getInfo()
    water_area_list_second = water_areas_second.aggregate_array('water_area').getInfo()
    dates = water_areas_first.aggregate_array('system:time_start').getInfo()
    dates = pd.to_datetime(dates, unit='ms')

    df_first = pd.DataFrame({'Date': dates, 'Water_Area': water_area_list_first})
    df_second = pd.DataFrame({'Date': dates, 'Water_Area': water_area_list_second})

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(df_first['Date'], df_first['Water_Area'], label=title_first, color='blue')
    plt.plot(df_second['Date'], df_second['Water_Area'], label=title_second, color='red')
    plt.xlabel('Date')
    plt.ylabel('Water Area (square meters)')
    plt.title('Water Area Time Series')
    plt.legend()
    plt.show()

