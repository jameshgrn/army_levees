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
from scipy.signal import correlate
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw  

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

def diff_normalize(profile_nld, profile_3dep):
    """
    Normalize both elevation profiles to have zero mean and unit variance.
    
    Args:
        profile_nld (array-like): NLD elevation profile
        profile_3dep (array-like): 3DEP elevation profile
    
    Returns:
        tuple: Normalized NLD and 3DEP profiles
    """
    # Convert to numpy arrays if they aren't already
    profile_nld = np.array(profile_nld)
    profile_3dep = np.array(profile_3dep)
    
    # Normalize profiles
    nld_norm = (profile_nld - np.mean(profile_nld)) / np.std(profile_nld)
    dep_norm = (profile_3dep - np.mean(profile_3dep)) / np.std(profile_3dep)
    
    return nld_norm, dep_norm

def cal_diff_normalized(profile_nld, profile_3dep):
    """
    Calculate the difference between normalized profiles.
    
    Args:
        profile_nld (array-like): NLD elevation profile
        profile_3dep (array-like): 3DEP elevation profile
    
    Returns:
        array: Difference between normalized profiles
    """
    # Get normalized profiles
    nld_norm, dep_norm = diff_normalize(profile_nld, profile_3dep)
    # Calculate difference
    return nld_norm - dep_norm

def plot_diff_distributions(plot_df, filtered_significant):
    """
    Plot distributions of differences between normalized profiles for all significant cases.
    Creates two figures with 7x8 subplots each.
    """
    # Get unique system IDs from filtered_significant
    significant_ids = filtered_significant['system_id'].unique()
    n_plots = len(significant_ids)
    plots_per_fig = 56  # 7 rows * 8 columns
    n_figs = (n_plots + plots_per_fig - 1) // plots_per_fig  # Ceiling division
    
    for fig_num in range(n_figs):
        # Create figure with 7x8 subplots
        fig, axes = plt.subplots(7, 8, figsize=(32, 28))
        fig.suptitle(f'Normalized Difference Distributions (Page {fig_num + 1})', fontsize=16)
        
        # Flatten axes for easier indexing
        axes_flat = axes.flatten()
        
        # Get the subset of system_ids for this figure
        start_idx = fig_num * plots_per_fig
        end_idx = min((fig_num + 1) * plots_per_fig, n_plots)
        current_ids = significant_ids[start_idx:end_idx]
        
        for idx, system_id in enumerate(current_ids):
            ax = axes_flat[idx]
            
            # Get data for this system
            df_nld = plot_df[(plot_df['system_id'] == system_id) & (plot_df['source'] == 'nld')]
            df_3dep = plot_df[(plot_df['system_id'] == system_id) & (plot_df['source'] == 'tep')]
            
            if not df_nld.empty and not df_3dep.empty:
                # Merge the dataframes
                merged = pd.merge(df_nld, df_3dep, on=['system_id', 'distance_along_track'], 
                                suffixes=('_nld', '_tep'))
                
                # Calculate normalized differences
                norm_diff = cal_diff_normalized(merged['elevation_nld'], merged['elevation_tep'])
                
                # Plot distribution
                ax.hist(norm_diff, bins=30, alpha=0.7, color='purple')
                ax.set_title(f'ID: {system_id}', fontsize=10)
                ax.set_xlim(-8, 8)
                ax.grid(True)
                
                # Only show y-axis labels for leftmost plots
                if idx % 8 != 0:
                    ax.set_yticklabels([])
                
                # Only show x-axis labels for bottom plots
                if idx < (end_idx - 8):
                    ax.set_xticklabels([])
            
        # Turn off any unused subplots
        for idx in range(len(current_ids), plots_per_fig):
            axes_flat[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'normalized_diff_distributions_page{fig_num+1}.png'))
        plt.close()

def plot_normalized_profiles(plot_df, system_id_examples):
    """
    Plot original and normalized profiles and their distributions for example system IDs.
    """
    for system_id in system_id_examples:
        # Get data for this system
        df_nld = plot_df[(plot_df['system_id'] == system_id) & (plot_df['source'] == 'nld')]
        df_3dep = plot_df[(plot_df['system_id'] == system_id) & (plot_df['source'] == 'tep')]
        
        if df_nld.empty or df_3dep.empty:
            print(f"Skipping system_id {system_id}: insufficient data")
            continue
            
        # Merge the dataframes
        merged = pd.merge(df_nld, df_3dep, on=['system_id', 'distance_along_track'], suffixes=('_nld', '_tep'))
        
        # Normalize profiles
        nld_norm, dep_norm = diff_normalize(merged['elevation_nld'], merged['elevation_tep'])
        
        # Calculate normalized differences
        norm_diff = cal_diff_normalized(merged['elevation_nld'], merged['elevation_tep'])
        
        # Create figure with 5 subplots
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(12, 20))
        
        # Plot original profiles
        ax1.plot(merged['distance_along_track'], merged['elevation_nld'], label='NLD', color='blue')
        ax1.plot(merged['distance_along_track'], merged['elevation_tep'], label='3DEP', color='red')
        ax1.set_title(f'Original Elevation Profiles - System ID: {system_id}')
        ax1.set_xlabel('Distance Along Track')
        ax1.set_ylabel('Elevation (m)')
        ax1.legend()
        ax1.grid(True)
        
        # Plot normalized profiles
        ax2.plot(merged['distance_along_track'], nld_norm, label='NLD (normalized)', color='blue')
        ax2.plot(merged['distance_along_track'], dep_norm, label='3DEP (normalized)', color='red')
        ax2.set_title('Normalized Elevation Profiles')
        ax2.set_xlabel('Distance Along Track')
        ax2.set_ylabel('Normalized Elevation')
        ax2.legend()
        ax2.grid(True)
        
        # Plot distributions
        ax3.hist(nld_norm, bins=30, alpha=0.5, label='NLD', color='blue')
        ax3.hist(dep_norm, bins=30, alpha=0.5, label='3DEP', color='red')
        ax3.set_title('Distribution of Normalized Elevations')
        ax3.set_xlabel('Normalized Elevation')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True)
        ax3.set_xlim(-8, 8)
        
        # Plot distribution of differences
        ax4.hist(norm_diff, bins=30, alpha=0.7, color='purple', label='NLD - 3DEP')
        ax4.set_title('Distribution of Differences Between Normalized Profiles')
        ax4.set_xlabel('Difference in Normalized Elevation')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True)
        ax4.set_xlim(-8, 8)
        
        # Plot along-track differences
        ax5.plot(merged['distance_along_track'], norm_diff, color='purple', label='Normalized Difference')
        ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax5.set_title('Along-Track Normalized Differences (NLD - 3DEP)')
        ax5.set_xlabel('Distance Along Track')
        ax5.set_ylabel('Normalized Difference')
        ax5.legend()
        ax5.grid(True)
        ax5.set_ylim(-8, 8)  # Match the x-limits of the difference distribution
        
        # Add statistics to the along-track difference plot
        stats_text = f'Mean: {np.mean(norm_diff):.3f}\nStd: {np.std(norm_diff):.3f}'
        ax5.text(0.02, 0.98, stats_text,
                transform=ax5.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'normalized_profiles_{system_id}.png'))
        plt.close()

def plot_normalized_scatter(plot_df, filtered_significant):
    """
    Plot scatter plots of normalized profiles for all significant cases.
    Creates figures with 7x8 subplots each.
    """
    # Get unique system IDs from filtered_significant
    significant_ids = filtered_significant['system_id'].unique()
    n_plots = len(significant_ids)
    plots_per_fig = 56  # 7 rows * 8 columns
    n_figs = (n_plots + plots_per_fig - 1) // plots_per_fig  # Ceiling division
    
    for fig_num in range(n_figs):
        # Create figure with 7x8 subplots
        fig, axes = plt.subplots(7, 8, figsize=(32, 28))
        fig.suptitle(f'Normalized Profile Scatter Plots (Page {fig_num + 1})', fontsize=16)
        
        # Flatten axes for easier indexing
        axes_flat = axes.flatten()
        
        # Get the subset of system_ids for this figure
        start_idx = fig_num * plots_per_fig
        end_idx = min((fig_num + 1) * plots_per_fig, n_plots)
        current_ids = significant_ids[start_idx:end_idx]
        
        for idx, system_id in enumerate(current_ids):
            ax = axes_flat[idx]
            
            # Get data for this system
            df_nld = plot_df[(plot_df['system_id'] == system_id) & (plot_df['source'] == 'nld')]
            df_3dep = plot_df[(plot_df['system_id'] == system_id) & (plot_df['source'] == 'tep')]
            
            if not df_nld.empty and not df_3dep.empty:
                # Merge the dataframes
                merged = pd.merge(df_nld, df_3dep, on=['system_id', 'distance_along_track'], 
                                suffixes=('_nld', '_tep'))
                
                # Get normalized profiles
                nld_norm, dep_norm = diff_normalize(merged['elevation_nld'], merged['elevation_tep'])
                
                # Calculate correlation coefficient
                corr_coef = np.corrcoef(nld_norm, dep_norm)[0,1]
                
                # Create scatter plot
                ax.scatter(nld_norm, dep_norm, alpha=0.5, c='blue', s=10)
                
                # Add diagonal line
                lims = [
                    min(ax.get_xlim()[0], ax.get_ylim()[0]),
                    max(ax.get_xlim()[1], ax.get_ylim()[1]),
                ]
                ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
                
                # Set equal limits
                ax.set_xlim(-4, 4)
                ax.set_ylim(-4, 4)
                
                # Add title with correlation coefficient
                ax.set_title(f'ID: {system_id}\nρ: {corr_coef:.3f}', fontsize=8)
                ax.grid(True)
                
                # Only show y-axis labels for leftmost plots
                if idx % 8 != 0:
                    ax.set_yticklabels([])
                else:
                    ax.set_ylabel('3DEP Normalized', fontsize=8)
                
                # Only show x-axis labels for bottom plots
                if idx < (end_idx - 8):
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel('NLD Normalized', fontsize=8)
                
                # Set equal aspect ratio
                ax.set_aspect('equal')
            
        # Turn off any unused subplots
        for idx in range(len(current_ids), plots_per_fig):
            axes_flat[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'normalized_scatter_plots_page{fig_num+1}.png'))
        plt.close()

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
    
    # Get all significant system IDs
    significant_ids = filtered_significant['system_id'].unique()
    
    print("\nPlotting normalized profiles for all significant system IDs...")
    # Create a subdirectory for the normalized profile plots
    normalized_plots_dir = os.path.join(output_dir, 'normalized_profiles')
    os.makedirs(normalized_plots_dir, exist_ok=True)
    
    # Plot normalized profiles for all significant IDs
    for i, system_id in enumerate(significant_ids):
        print(f"\rProcessing profile {i+1}/{len(significant_ids)}: {system_id}", end='')
        plot_normalized_profiles(filtered_plot_df, [system_id])  # Pass as single-item list
    print("\nCompleted plotting all normalized profiles.")
    
    print("\nPlotting normalized difference distributions...")
    plot_diff_distributions(filtered_plot_df, filtered_significant)
    
    print("\nPlotting normalized scatter plots...")
    plot_normalized_scatter(filtered_plot_df, filtered_significant)
    
    

