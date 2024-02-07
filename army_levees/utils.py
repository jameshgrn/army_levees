import cartopy.feature as cfeatures
import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from cartopy.io.img_tiles import GoogleTiles
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.patheffects as pe
import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import py3dep
from pyproj import Transformer
import geopandas as gpd
from shapely.geometry import MultiLineString, LineString
import numpy as np

def calculate_cumulative_distance(points):
    distances = [0]
    for i in range(1, len(points)):
        distances.append(points[i].distance(points[i-1]) + distances[-1])
    return distances

def json_to_geodataframe(json_response):
    # Initialize lists to store the parsed data
    points = []
    elevations = []
    distances = []
    
    # Iterate over the arcs to extract the data
    for arc in json_response['geometry']['arcs'][0]:
        x, y, elevation, distance = arc
        points.append(Point(x, y))
        elevations.append(elevation)
        distances.append(distance)
    
    # Create a DataFrame with the elevation and distance along track
    df = pd.DataFrame({
        'elevation': elevations,
        'distance_along_track': distances
    })
    
    # Convert the DataFrame to a GeoDataFrame, setting the geometry
    gdf = gpd.GeoDataFrame(df, geometry=points, crs="EPSG:26914")
    
    return gdf

def process_segment(row, crs):
    try:
        # Create a DataFrame to hold the elevation data
        elevation_data_list = []
        if isinstance(row.geometry, MultiLineString):
            for line in row.geometry.geoms:
                elevation = py3dep.elevation_profile(line, crs=crs, spacing=10, dem_res=10)
                elevation_df = elevation.to_dataframe(name='elevation').reset_index()        
                elevation_data_list.append(elevation_df)
        else:
            line = row.geometry
            elevation = py3dep.elevation_profile(line, crs=crs, spacing=10, dem_res=10)
            elevation_df = elevation.to_dataframe(name='elevation').reset_index()        
            elevation_data_list.append(elevation_df)
        
        # Concatenate all elevation data
        elevation_data_full = pd.concat(elevation_data_list)
        elevation_gdf = gpd.GeoDataFrame(elevation_data_full, geometry=gpd.points_from_xy(elevation_data_full.x, elevation_data_full.y))
        
        # Add the 'name' and other row information to the GeoDataFrame
        for column in row.index:
            elevation_gdf[column] = row[column]
        
        return elevation_gdf
    except ValueError as e:
        print(f"Caught an error: {e}")
        

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



def reproject_dataframe_geometries(df, source_crs="EPSG:4326", target_crs="EPSG:5048"):
    """
    Transforms the elevation values in a GeoDataFrame from one vertical datum to another.

    Parameters:
    - df: GeoDataFrame with point geometries including 'longitude', 'latitude', and 'elevation' columns.
    - source_crs: The current CRS of the GeoDataFrame. Defaults to EPSG:4269.
    - target_crs: The target CRS to which the geometries will be reprojected. Defaults to EPSG:4326.

    Returns:
    - A new GeoDataFrame with transformed elevation values.
    """
    # Initialize a transformer object with the source and target CRS
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
    df['transformed_elevation'] = df.apply(transform_elevation, axis=1)
    return df

def plot_profiles(profile_gdf, elevation_data_full):
    # Ensure data is sorted by 'distance_along_track' to represent downstream direction
    profile_gdf_sorted = profile_gdf.sort_values(by='distance_along_track')
    elevation_data_sorted = elevation_data_full.sort_values(by='distance_along_track')
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot profile_gdf
    plt.plot(profile_gdf_sorted['distance_along_track'], profile_gdf_sorted['transformed_elevation'], label='NLD Profile', color='blue', marker='o', linestyle='-', markersize=5)
    
    # Plot elevation_data_full
    plt.plot(elevation_data_sorted['distance_along_track'], elevation_data_sorted['elevation'], label='3DEP Profile', color='red', marker='x', linestyle='--', markersize=5)
    
    plt.title('Elevation Profiles Comparison')
    plt.xlabel('Distance Along Track (m)')
    plt.ylabel('Elevation (m)')
    plt.legend()
    plt.grid(True)
    plt.show()