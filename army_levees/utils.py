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
from shapely.geometry import MultiLineString, LineString
import numpy as np

def calculate_cumulative_distance(points):
    distances = [0]
    for i in range(1, len(points)):
        distances.append(points[i].distance(points[i-1]) + distances[-1])
    return distances

def plot_segment(segment_id, column='elevation'):
    el_df = None  # Initialize el_df to ensure it's in the function's scope
    try:
        el_df = elevation_data_full.query(f'segmentId == {segment_id}')
        el_df = gpd.GeoDataFrame(el_df, geometry=gpd.points_from_xy(el_df['x'], el_df['y']), crs=5070)
        # Convert to a UTM CRS for accurate distance measurements
        el_df = el_df.to_crs(epsg=4979)  # Replace with the correct UTM zone for your area
        if len(el_df) > 1:
            # Sort by latitude (y) since the river flows south
            el_df.sort_values(by='y', ascending=True, inplace=True)
            max_height_df = el_df.groupby('distance')[column].max().reset_index()
            # Apply a rolling median filter
            max_height_df[column] = max_height_df[column].rolling(window=6, center=True).median().fillna(method='bfill').fillna(method='ffill')
            # Convert distance and height to feet
            max_height_df['distance'] = max_height_df['distance'] * 3.281
            max_height_df[column] = max_height_df[column] * 3.281
            # Plot the maximum elevation for each distance
            plt.plot(max_height_df['distance'], max_height_df[column], 'o-', alpha=0.5, markersize=3)
            # Calculate the standard deviation and mean residuals
            std_dev = np.std(max_height_df[column])
            residuals = max_height_df[column] - np.poly1d(np.polyfit(max_height_df['distance'], max_height_df[column], 1))(max_height_df['distance'])
            mean_residuals = np.mean(residuals)
            # Add plot details
            system_name = el_df['systemName'].iloc[0]
            plt.title(f'Segment ID: {segment_id} - System Name: {system_name}')
            plt.text(0.05, 0.95, f'Standard Deviation: {std_dev:.2f}', transform=plt.gca().transAxes)
            plt.text(0.05, 0.90, f'Mean Residuals: {mean_residuals:.2f}', transform=plt.gca().transAxes)
            plt.xlabel('Distance along river (ft)')
            plt.ylabel(f'Maximum {column} (ft)')
            plt.show()
            plt.close()
        return el_df
    except Exception as e:
        print(f"Error processing segment: {e}")
        return el_df  # Return el_df even if it's None to handle the exception gracefully

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
    gdf = gpd.GeoDataFrame(df, geometry=points, crs="EPSG:3857")
    
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