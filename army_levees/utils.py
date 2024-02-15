import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, MultiLineString, LineString
from pyproj import Transformer
from py3dep import elevation_bycoords

def json_to_geodataframe(json_response):
    # Extract coordinates and attributes
    coords = [(arc[0], arc[1]) for arc in json_response['geometry']['arcs'][0]]
    elevations = [arc[2] for arc in json_response['geometry']['arcs'][0]]
    distances = [arc[3] for arc in json_response['geometry']['arcs'][0]]

    # Create DataFrame
    df = pd.DataFrame({
        'elevation': elevations,
        'distance_along_track': distances,
        'geometry': gpd.points_from_xy(*zip(*coords))
    })

    # Convert DataFrame to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:3857")
    return gdf
        

def plot_profiles(profile_gdf, elevation_data_full):
    # Sort data by 'distance_along_track'
    profile_gdf_sorted = profile_gdf.sort_values(by='distance_along_track')
    elevation_data_sorted = elevation_data_full.sort_values(by='distance_along_track')
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(profile_gdf_sorted['distance_along_track'], profile_gdf_sorted['elevation'], label='NLD Profile', color='blue', marker='o', linestyle='-', markersize=1)
    plt.plot(elevation_data_sorted['distance_along_track'], elevation_data_sorted['elevation'], label='3DEP Profile', color='red', marker='x', linestyle='--', markersize=1)
    plt.title('Elevation Profiles Comparison')
    plt.xlabel('Distance Along Track (m)')
    plt.ylabel('Elevation (m)')
    plt.legend()
    plt.grid(True)
    plt.show()
