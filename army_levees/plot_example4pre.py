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

# Directory where the elevation data files are saved
data_dir = '/Users/liyuan/projects/army_levees/data'
output_dir = '/Users/liyuan/projects/army_levees/plots'
# Import the csv data mean_differences.csv to get the potential example sites
csv_path = '/Users/liyuan/projects/army_levees/mean_differences.csv'
# Read the csv file
df_100 = pd.read_csv(csv_path)
# Get the system_id column
system_id_100 = df_100['system_id']
# # Print the first 5 rows of the dataframe
# print(system_id_100.head())

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

def load_and_process_data():
    gdfs = [gpd.read_parquet(os.path.join(data_dir, f)).to_crs(epsg=4326) 
            for f in os.listdir(data_dir) if f.endswith('.parquet')]
    if not gdfs:
        return gpd.GeoDataFrame()  # return empty GeoDataFrame if no files found
    return pd.concat(gdfs, ignore_index=True)

def create_subset_data(plot_df, system_id_100):
    system_id_set = set(system_id_100)
    subset_df = plot_df[plot_df['system_id'].isin(system_id_set)]
    print(f"Number of unique system_ids in the subset: {subset_df['system_id'].nunique()}")
    print(f"Shape of the subset DataFrame: {subset_df.shape}")
    return subset_df

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

def create_significant_subsets(subset_df):
    # Create significant subset
    subset_df_sig = subset_df[subset_df['mean_diff_significant'] == 1]
    
    # Create non-significant subset
    subset_df_nonsig = subset_df[subset_df['mean_diff_significant'] == 0]
    
    # print(f"Number of unique system_ids in significant subset: {subset_df_sig['system_id'].nunique()}")
    # print(f"Shape of significant subset: {subset_df_sig.shape}")
    # print(f"Number of unique system_ids in non-significant subset: {subset_df_nonsig['system_id'].nunique()}")
    # print(f"Shape of non-significant subset: {subset_df_nonsig.shape}")
    
    return subset_df_sig, subset_df_nonsig

def plot_elevation_profiles(df, num_plots=9, title_suffix=""):
    # Get unique system_ids and randomly select 9
    unique_systems = df['system_id'].unique()
    selected_systems = random.sample(list(unique_systems), min(num_plots, len(unique_systems)))
    
    # Create a 3x3 grid of subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f'Elevation Profiles: NLD vs TEP {title_suffix}', fontsize=16)
    
    for idx, system_id in enumerate(selected_systems):
        ax = axes[idx // 3, idx % 3]
        system_data = df[df['system_id'] == system_id]
        
        nld_data = system_data[system_data['source'] == 'nld']
        tep_data = system_data[system_data['source'] == 'tep']
        
        ax.plot(nld_data['distance_along_track'], nld_data['elevation'], 'b-o', markersize=2, linewidth=1, label='NLD')
        ax.plot(tep_data['distance_along_track'], tep_data['elevation'], 'r-o', markersize=2, linewidth=1, label='TEP')
        
        mean_diff = system_data['mean_elevation_diff'].iloc[0]
        ax.set_title(f'System ID: {system_id}\nMean Diff: {mean_diff:.2f}m', fontsize=10)
        ax.set_xlabel('Distance Along Track (m)', fontsize=8)
        ax.set_ylabel('Elevation (m)', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.legend(fontsize=6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'a_9example_{title_suffix.replace(" ", "_")}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_all_profiles(df, title, filename_prefix, profiles_per_page=50):
    unique_systems = df['system_id'].unique()
    num_systems = len(unique_systems)
    pages = math.ceil(num_systems / profiles_per_page)
    
    for page in range(pages):
        start_idx = page * profiles_per_page
        end_idx = min((page + 1) * profiles_per_page, num_systems)
        systems_to_plot = unique_systems[start_idx:end_idx]
        
        rows = math.ceil(len(systems_to_plot) / 10)  # 10 columns per row
        fig, axes = plt.subplots(rows, 10, figsize=(50, 5*rows))
        fig.suptitle(f'{title} (Page {page+1}/{pages})', fontsize=24)
        
        for idx, system_id in enumerate(systems_to_plot):
            ax = axes[idx // 10, idx % 10] if rows > 1 else axes[idx]
            system_data = df[df['system_id'] == system_id]
            
            nld_data = system_data[system_data['source'] == 'nld']
            tep_data = system_data[system_data['source'] == 'tep']
            
            ax.plot(nld_data['distance_along_track'], nld_data['elevation'], 'b-', linewidth=1, label='NLD')
            ax.plot(tep_data['distance_along_track'], tep_data['elevation'], 'r-', linewidth=1, label='TEP')
            
            mean_diff = system_data['mean_elevation_diff'].iloc[0]
            ax.set_title(f'ID: {system_id}\nDiff: {mean_diff:.2f}m', fontsize=8)
            ax.set_xlabel('Distance (m)', fontsize=6)
            ax.set_ylabel('Elevation (m)', fontsize=6)
            ax.tick_params(axis='both', which='major', labelsize=5)
            
            if idx == 0:  # Only show legend for the first subplot
                ax.legend(fontsize=6, loc='upper right')
        
        # Remove any unused subplots
        for idx in range(len(systems_to_plot), rows*10):
            fig.delaxes(axes[idx // 10, idx % 10] if rows > 1 else axes[idx])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{filename_prefix}_page{page+1}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

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
        
        ctx.add_basemap(ax1, source=basemap_provider, crs=line_gdf.crs, 
                        zoom='auto', attribution_size=6)
        
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
    
    # Create and show the plot with USGS topo basemap
    # fig_topo = create_subplots(ctx.providers.USGS.USTopo)
    # plt.show()
    
    # Create and show the plot with satellite imagery basemap
    fig_satellite = create_subplots(ctx.providers.Esri.WorldImagery)
    plt.show()

if __name__ == "__main__":
    plot_df = load_and_process_data()
    print(f"Combined DataFrame shape: {plot_df.shape}")
    print(f"Combined DataFrame columns: {plot_df.columns}")
    
    if plot_df.empty:
        print("WARNING: No data loaded. Check your data directory and file formats.")
        exit()
    
    filtered_plot_df, elevation_statistics = calculate_mean_differences(plot_df)
    
    # Create the subset using system_id_100
    subset_df = create_subset_data(filtered_plot_df, system_id_100)
    
    # Create significant and non-significant subsets
    subset_df_sig, subset_df_nonsig = create_significant_subsets(subset_df)
    
    # # Plot elevation profiles for 9 random system_ids from each subset
    # plot_elevation_profiles(subset_df, title_suffix="All")
    # plot_elevation_profiles(subset_df_sig, title_suffix="Significant")
    # plot_elevation_profiles(subset_df_nonsig, title_suffix="Non-significant")
    
    # # Plot all significant profiles
    # plot_all_profiles(subset_df_sig, "Significant Elevation Profiles: NLD vs TEP", "all_significant_profiles")
    
    # # Plot all non-significant profiles
    # plot_all_profiles(subset_df_nonsig, "Non-significant Elevation Profiles: NLD vs TEP", "all_non_significant_profiles")
    
    # Plot the map for system_id 3905550001
    plot_system_track_map(filtered_plot_df, 6005000230)

    # print("All elevation profile plots have been saved in the 'plots' folder.")

    # # Export the first 200 rows of plot_df to a CSV file
    # csv_output_path = os.path.join(data_dir, 'allData_filtered.csv')
    # plot_df.head(200).to_csv(csv_output_path, index=False)
    # print(f"First 200 rows of plot_df have been saved to {csv_output_path}")