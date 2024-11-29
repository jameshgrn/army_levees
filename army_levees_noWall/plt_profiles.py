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
data_dir = '/Users/liyuan/projects/army_levees/data_no_floodwall'
output_dir = '/Users/liyuan/projects/army_levees/army_levees_noWall/plots'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

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

def plot_significant_profiles(filtered_significant, output_dir, profiles_per_plot=56):  # 7x8=56 plots per figure
    """
    Plot elevation profiles for significant differences in a 7x8 grid
    """
    # Create plots directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique system IDs
    unique_systems = filtered_significant['system_id'].unique()
    num_systems = len(unique_systems)
    
    # Calculate number of figures needed
    num_figures = math.ceil(num_systems / profiles_per_plot)
    
    for fig_num in range(num_figures):
        # Create figure with 7x8 subplots
        fig, axes = plt.subplots(7, 8, figsize=(32, 28))  # Adjusted figure size for 7x8 grid
        fig.suptitle('Significant Elevation Profiles (NLD vs TEP)', fontsize=16)
        
        # Flatten axes array for easier indexing
        axes_flat = axes.flatten()
        
        # Get the subset of systems for this figure
        start_idx = fig_num * profiles_per_plot
        end_idx = min((fig_num + 1) * profiles_per_plot, num_systems)
        systems_subset = unique_systems[start_idx:end_idx]
        
        # Plot each system's profile
        for i, system_id in enumerate(systems_subset):
            ax = axes_flat[i]
            
            # Get data for this system
            system_data = filtered_significant[filtered_significant['system_id'] == system_id]
            nld_data = system_data[system_data['source'] == 'nld'].sort_values('distance_along_track')
            tep_data = system_data[system_data['source'] == 'tep'].sort_values('distance_along_track')
            
            # Plot profiles with both lines and dots
            # Plot NLD data with blue line and dots
            ax.plot(nld_data['distance_along_track'], nld_data['elevation'], 'b-', linewidth=1, label='NLD')
            ax.plot(nld_data['distance_along_track'], nld_data['elevation'], 'bo', markersize=3)
            
            # Plot TEP data with red line and dots
            ax.plot(tep_data['distance_along_track'], tep_data['elevation'], 'r-', linewidth=1, label='TEP')
            ax.plot(tep_data['distance_along_track'], tep_data['elevation'], 'ro', markersize=3)
            
            # Add title and legend
            ax.set_title(f'System ID: {system_id}', fontsize=8)
            ax.legend(fontsize=6)
            ax.tick_params(labelsize=6)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # Turn off any unused subplots
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)
        
        # Adjust layout and save figure
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)  # Make room for suptitle
        
        # Save figure
        filename = f'significant_profiles_part{fig_num + 1}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {filename}")

def plot_nonsignificant_profiles(filtered_nonsignificant, output_dir, profiles_per_plot=56):  # 7x8=56 plots per figure
    """
    Plot elevation profiles for non-significant differences in a 7x8 grid
    """
    # Create plots directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique system IDs
    unique_systems = filtered_nonsignificant['system_id'].unique()
    num_systems = len(unique_systems)
    
    # Calculate number of figures needed
    num_figures = math.ceil(num_systems / profiles_per_plot)
    
    for fig_num in range(num_figures):
        # Create figure with 7x8 subplots
        fig, axes = plt.subplots(7, 8, figsize=(32, 28))  # Adjusted figure size for 7x8 grid
        fig.suptitle('Non-Significant Elevation Profiles (NLD vs TEP)', fontsize=16)
        
        # Flatten axes array for easier indexing
        axes_flat = axes.flatten()
        
        # Get the subset of systems for this figure
        start_idx = fig_num * profiles_per_plot
        end_idx = min((fig_num + 1) * profiles_per_plot, num_systems)
        systems_subset = unique_systems[start_idx:end_idx]
        
        # Plot each system's profile
        for i, system_id in enumerate(systems_subset):
            ax = axes_flat[i]
            
            # Get data for this system
            system_data = filtered_nonsignificant[filtered_nonsignificant['system_id'] == system_id]
            nld_data = system_data[system_data['source'] == 'nld'].sort_values('distance_along_track')
            tep_data = system_data[system_data['source'] == 'tep'].sort_values('distance_along_track')
            
            # Plot profiles with both lines and dots
            # Plot NLD data with blue line and dots
            ax.plot(nld_data['distance_along_track'], nld_data['elevation'], 'b-', linewidth=1, label='NLD')
            ax.plot(nld_data['distance_along_track'], nld_data['elevation'], 'bo', markersize=3)
            
            # Plot TEP data with red line and dots
            ax.plot(tep_data['distance_along_track'], tep_data['elevation'], 'r-', linewidth=1, label='TEP')
            ax.plot(tep_data['distance_along_track'], tep_data['elevation'], 'ro', markersize=3)
            
            # Add title and legend
            ax.set_title(f'System ID: {system_id}', fontsize=8)
            ax.legend(fontsize=6)
            ax.tick_params(labelsize=6)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # Turn off any unused subplots
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)
        
        # Adjust layout and save figure
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)  # Make room for suptitle
        
        # Save figure
        filename = f'nonsignificant_profiles_part{fig_num + 1}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {filename}")

if __name__ == "__main__":
    plot_df = load_and_process_data()
    print(f"Combined DataFrame shape: {plot_df.shape}")
    print(f"Combined DataFrame columns: {plot_df.columns}")
    
    if plot_df.empty:
        print("WARNING: No data loaded. Check your data directory and file formats.")
        exit()
    
    # Print total number of unique system IDs (profile sets)
    total_profiles = plot_df['system_id'].nunique()
    print(f"Total number of profile sets in plot_df: {total_profiles}")
    
    filtered_plot_df, elevation_statistics = calculate_mean_differences(plot_df)
    
    # Create significant and non-significant subsets
    filtered_significant = filtered_plot_df[filtered_plot_df['mean_diff_significant'] == 1]
    filtered_nonsignificant = filtered_plot_df[filtered_plot_df['mean_diff_significant'] == 0]
    
    # Print information about the subsets
    print("\nSubset Information:")
    print(f"Number of significant profiles: {filtered_significant['system_id'].nunique()}")
    print(f"Number of non-significant profiles: {filtered_nonsignificant['system_id'].nunique()}")
    
    # Plot both significant and non-significant profiles
    print("\nPlotting significant profiles...")
    plot_significant_profiles(filtered_significant, os.path.join(output_dir, 'significant'))
    
    print("\nPlotting non-significant profiles...")
    plot_nonsignificant_profiles(filtered_nonsignificant, os.path.join(output_dir, 'nonsignificant'))

