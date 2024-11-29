'''
Script to plot 3 by 3 examples for the presentation.    
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import os
from shapely.geometry import Point
from scipy.stats import zscore
import random
import math
import contextily as ctx
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import box
from shapely.geometry import LineString
from fastkml import kml, styles

# Directory where the elevation data files are saved
data_dir = '/Users/liyuan/projects/army_levees/data_no_floodwall'
output_dir = '/Users/liyuan/projects/army_levees/plots'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# System ID to plot (change this value to plot different profiles)
system_id_example = 6005000685  # You can change this to any system ID you want to examine

def load_and_process_data():
    gdfs = [gpd.read_parquet(os.path.join(data_dir, f)).to_crs(epsg=4326) 
            for f in os.listdir(data_dir) if f.endswith('.parquet')]
    if not gdfs:
        return gpd.GeoDataFrame()  # return empty GeoDataFrame if no files found
    return pd.concat(gdfs, ignore_index=True)


def calculate_mean_differences(plot_df):
    differences = []
    filtered_data = []
    for system_id in plot_df['system_id'].unique():
        df_nld = plot_df[(plot_df['system_id'] == system_id) & (plot_df['source'] == 'nld')]
        df_3dep = plot_df[(plot_df['system_id'] == system_id) & (plot_df['source'] == 'tep')]
        
        if not df_nld.empty and not df_3dep.empty:
            # Remove segments where NLD elevation values are consistently zero
            df_nld = df_nld[df_nld['elevation'] != 0]
            
            # Merge the two dataframes on 'system_id' and 'distance_along_track' to align the measurements
            merged = pd.merge(df_nld, df_3dep, on=['system_id', 'distance_along_track'], suffixes=('_nld', '_tep'))
            
            # Remove rows where either elevation is NaN, None, or empty string
            merged = merged.dropna(subset=['elevation_nld', 'elevation_tep'])
            merged = merged[(merged['elevation_nld'] != '') & (merged['elevation_tep'] != '')]
            
            # Ensure elevations are numeric
            merged['elevation_nld'] = pd.to_numeric(merged['elevation_nld'], errors='coerce')
            merged['elevation_tep'] = pd.to_numeric(merged['elevation_tep'], errors='coerce')
            
            # Filter out rows with NaN values in elevation columns
            merged = merged.dropna(subset=['elevation_nld', 'elevation_tep'])
            
            # Calculate z-scores for elevation columns
            merged['zscore_nld'] = zscore(merged['elevation_nld'])
            merged['zscore_tep'] = zscore(merged['elevation_tep'])
            
            # Remove rows with z-scores beyond a threshold (e.g., |z| > 3)
            merged_filtered = merged[(merged['zscore_nld'].abs() <= 3) & (merged['zscore_tep'].abs() <= 3)]
            
            if len(merged_filtered) > 0:
                # Calculate the difference only for valid points
                elevation_diff = merged_filtered['elevation_tep'] - merged_filtered['elevation_nld']  # TEP - NLD
                elevation_diff = elevation_diff.dropna()
                
                if len(elevation_diff) > 0:
                    mean_diff = elevation_diff.mean()
                    differences.append({
                        'system_id': system_id, 
                        'mean_elevation_diff': mean_diff,
                        'mean_diff_significant': 1 if abs(mean_diff) > 0.1 else 0,
                        'std_elevation_diff': elevation_diff.std(),
                        'max_elevation_diff': elevation_diff.max(),
                        'min_elevation_diff': elevation_diff.min(),
                        'range_elevation_diff': elevation_diff.max() - elevation_diff.min(),
                        'percentile_25_elevation_diff': elevation_diff.quantile(0.25),
                        'percentile_75_elevation_diff': elevation_diff.quantile(0.75),
                        'points_compared': len(elevation_diff)
                    })
                    
                    # Add filtered data for both NLD and TEP
                    filtered_data.append(merged_filtered[['system_id', 'distance_along_track', 'elevation_nld', 'source_nld', 'x_nld', 'y_nld']]
                                         .rename(columns={'elevation_nld': 'elevation', 'source_nld': 'source', 'x_nld': 'x', 'y_nld': 'y'}))
                    filtered_data.append(merged_filtered[['system_id', 'distance_along_track', 'elevation_tep', 'source_tep', 'x_tep', 'y_tep']]
                                         .rename(columns={'elevation_tep': 'elevation', 'source_tep': 'source', 'x_tep': 'x', 'y_tep': 'y'}))
    
    diff_df = pd.DataFrame(differences)
    
    # Filter out profiles with more than 15m of mean difference
    filtered_df = diff_df[diff_df['mean_elevation_diff'].abs() <= 15]
    
    print(f"Filtered out {len(diff_df) - len(filtered_df)} profiles with >15m mean difference")
    print(f"Total points compared: {filtered_df['points_compared'].sum()}")
    
    # Combine all filtered data
    filtered_plot_df = pd.concat(filtered_data, ignore_index=True)
    
    # Merge differences back into the filtered DataFrame
    filtered_plot_df = filtered_plot_df.merge(filtered_df, on='system_id', how='left')
    
    return filtered_plot_df, filtered_df


def plot_system_track_map(plot_df, system_id):
    # Filter the data for the specified system_id
    system_data = plot_df[plot_df['system_id'] == system_id]
    
    if system_data.empty:
        print(f"No data found for system_id {system_id}")
        return
    
    # Sort the data by distance_along_track to ensure correct line order
    system_data = system_data.sort_values('distance_along_track')
    
    # Print the original x and y coordinates of the first point
    first_point = system_data.iloc[0]
    print(f"First point coordinates (lat, lon): ({first_point.y}, {first_point.x})")
    
    # Calculate the maximum distance along the track
    max_distance = system_data['distance_along_track'].max()
    
    # Find the points closest to 1/4, 1/2, and 3/4 of the maximum distance
    div_points = []
    for fraction in [0.25, 0.5, 0.75]:
        target_distance = max_distance * fraction
        closest_point = system_data.iloc[(system_data['distance_along_track'] - target_distance).abs().argsort().iloc[0]]
        div_points.append(closest_point)
    
    # Create a GeoDataFrame for the division points
    div_gdf = gpd.GeoDataFrame(div_points, geometry=gpd.points_from_xy([p.x for p in div_points], [p.y for p in div_points]), crs="EPSG:4326")
    
    # Function to create and style the subplots
    def create_subplots(basemap_provider):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        line = LineString(zip(system_data.x, system_data.y))
        start_point = Point(first_point.x, first_point.y)
        
        line_gdf = gpd.GeoDataFrame(geometry=[line], crs="EPSG:4326")
        point_gdf = gpd.GeoDataFrame(geometry=[start_point], crs="EPSG:4326")
        
        line_gdf = line_gdf.to_crs(epsg=3857)
        point_gdf = point_gdf.to_crs(epsg=3857)
        div_gdf_3857 = div_gdf.to_crs(epsg=3857)
        
        line_gdf.plot(ax=ax1, color='red', linewidth=2)
        point_gdf.plot(ax=ax1, color='green', marker='*', markersize=200, zorder=10)
        div_gdf_3857.plot(ax=ax1, color='yellow', marker='*', markersize=150, edgecolor='red', zorder=11)
        
        minx, miny, maxx, maxy = line_gdf.total_bounds
        padding = 200
        ax1.set_xlim(minx - padding, maxx + padding)
        ax1.set_ylim(miny - padding, maxy + padding)
        
        # ctx.add_basemap(ax1, source=basemap_provider, crs=line_gdf.crs, 
        #                 zoom='auto', attribution_size=6)
        # Use OpenTopography as the basemap
        ctx.add_basemap(ax1, 
                       source=ctx.providers.Esri.WorldImagery,
                       crs=line_gdf.crs, 
                       zoom='auto', 
                       attribution_size=6)
        
        ax1.set_xlabel('Easting (m)')
        ax1.set_ylabel('Northing (m)')
        ax1.set_title(f'Track of System ID: {system_id}')
        
        ax1.plot([], [], color='red', label='Track', linewidth=2)
        ax1.plot([], [], color='green', marker='*', markersize=15, linestyle='None', label='Start Point')
        ax1.plot([], [], color='yellow', marker='*', markersize=15, markeredgecolor='red', linestyle='None', label='Division Points')
        ax1.legend()
        
        # Plot the elevation profiles on the second subplot
        nld_data = system_data[system_data['source'] == 'nld']
        tep_data = system_data[system_data['source'] == 'tep']
        
        ax2.plot(nld_data['distance_along_track'], nld_data['elevation'], 'b-', label='NLD')
        ax2.plot(tep_data['distance_along_track'], tep_data['elevation'], 'r-', label='TEP')
        
        # Plot vertical lines without text labels
        for point in div_points:
            ax2.axvline(x=point['distance_along_track'], color='gray', linestyle='--', alpha=0.7)
        
        ax2.set_xlabel('Distance Along Track (m)')
        ax2.set_ylabel('Elevation (m)')
        ax2.set_title(f'Elevation Profiles for System ID: {system_id}')
        ax2.legend()
        
        plt.tight_layout()
        return fig
    
    # Create and show the plot with OpenTopography basemap
    fig_topo = create_subplots(ctx.providers.OpenTopoMap)
    plt.show()


def export_to_shapefile(plot_df, system_id, output_dir):
    """
    Export the levee track coordinates to a shapefile.
    Args:
        plot_df: DataFrame containing the levee track data
        system_id: ID of the system to export
        output_dir: Directory to save the shapefile
    """
    # Filter the data for the specified system_id
    system_data = plot_df[plot_df['system_id'] == system_id]
    
    if system_data.empty:
        print(f"No data found for system_id {system_id}")
        return
    
    # Sort the data by distance_along_track
    system_data = system_data.sort_values('distance_along_track')
    
    # Create line geometry from points
    line_coords = [(row.x, row.y) for _, row in system_data.drop_duplicates(['x', 'y']).iterrows()]
    line = LineString(line_coords)
    
    # Create GeoDataFrame with the line
    track_gdf = gpd.GeoDataFrame(
        {'system_id': [system_id], 'geometry': [line]}, 
        crs="EPSG:4326"
    )
    
    # Create GeoDataFrame with points
    points = []
    point_types = []
    
    # Add start point
    first_point = system_data.iloc[0]
    points.append(Point(first_point.x, first_point.y))
    point_types.append('start')
    
    # Add division points
    max_distance = system_data['distance_along_track'].max()
    for fraction in [0.25, 0.5, 0.75]:
        target_distance = max_distance * fraction
        closest_point = system_data.iloc[(system_data['distance_along_track'] - target_distance).abs().argsort().iloc[0]]
        points.append(Point(closest_point.x, closest_point.y))
        point_types.append(f'division_{int(fraction*100)}')
    
    points_gdf = gpd.GeoDataFrame(
        {'system_id': [system_id] * len(points),
         'point_type': point_types,
         'geometry': points}, 
        crs="EPSG:4326"
    )
    
    # Save to shapefiles
    track_path = os.path.join(output_dir, f'levee_track_{system_id}.shp')
    points_path = os.path.join(output_dir, f'levee_points_{system_id}.shp')
    
    track_gdf.to_file(track_path)
    points_gdf.to_file(points_path)
    
    print(f"Track shapefile saved to: {track_path}")
    print(f"Points shapefile saved to: {points_path}")

if __name__ == "__main__":
    plot_df = load_and_process_data()
    print(f"Combined DataFrame shape: {plot_df.shape}")
    print(f"Combined DataFrame columns: {plot_df.columns}")
    
    if plot_df.empty:
        print("WARNING: No data loaded. Check your data directory and file formats.")
        exit()
    
    filtered_plot_df, elevation_statistics = calculate_mean_differences(plot_df)
    
    '''plot only one case'''
    plot_system_track_map(filtered_plot_df, system_id_example)
    
    '''export to shapefile'''
    export_to_shapefile(filtered_plot_df, system_id_example, output_dir)
