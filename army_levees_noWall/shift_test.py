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
from scipy import odr

# Directory where the elevation data files are saved
data_dir = '/Users/liyuan/projects/army_levees/data_no_floodwall'
output_dir = '/Users/liyuan/projects/army_levees/army_levees_noWall/plots'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# System IDs to plot (change this value to plot different profiles)
system_id_examples = [2205000054, 5705000036, 5005000043, 6005000545, 6005000026,
                     280005000773, 5705000034, 4705000101, 5505000014, 5305000008]

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

def cross_correlation_compare(old_profile, new_profile):
    """
    Align two profiles using cross-correlation
    """
    # Calculate the cross-correlation
    correlation = correlate(new_profile, old_profile, mode='full')
    shift_index = np.argmax(correlation) - len(old_profile) + 1
    
    # Shift the new profile by the best alignment found
    shifted_new_profile = np.roll(new_profile, shift_index)
    
    return shifted_new_profile, shift_index

def find_optimal_vertical_shift(nld_elev, tep_elev, shift_range=(-10, 10), step=0.05):
    """
    Find the optimal vertical shift that minimizes the RMSE between TEP and NLD elevations
    """
    shifts = np.arange(shift_range[0], shift_range[1], step)
    rmse_values = []
    
    # Calculate RMSE for each shift
    for shift in shifts:
        shifted_tep = tep_elev + shift
        rmse = np.sqrt(np.mean((shifted_tep - nld_elev)**2))
        rmse_values.append(rmse)
    
    # Find optimal shift
    optimal_shift = shifts[np.argmin(rmse_values)]
    min_rmse = np.min(rmse_values)
    
    return optimal_shift, min_rmse

def plot_scatter_profile(filtered_significant, system_ids):
    """
    Create a combined figure with scatter plots (top 2 rows) and profiles (bottom 2 rows)
    """
    # Create a 4x5 subplot layout with adjusted figure size
    fig, axes = plt.subplots(4, 5, figsize=(35, 16))
    
    for idx, system_id in enumerate(system_ids):
        if system_id not in filtered_significant['system_id'].unique():
            print(f"System ID {system_id} not found in significant differences dataset.")
            continue
        
        # Calculate row and column indices
        row_idx = idx // 5
        col_idx = idx % 5
        
        # Get data for this system
        system_data = filtered_significant[filtered_significant['system_id'] == system_id]
        
        # Merge NLD and TEP data
        nld_data = system_data[system_data['source'] == 'nld'].sort_values('distance_along_track')
        tep_data = system_data[system_data['source'] == 'tep'].sort_values('distance_along_track')
        merged_data = pd.merge(
            nld_data[['distance_along_track', 'elevation']], 
            tep_data[['distance_along_track', 'elevation']], 
            on='distance_along_track', 
            suffixes=('_nld', '_tep')
        )
        
        # Calculate correlation coefficient
        corr_coef = np.corrcoef(merged_data['elevation_nld'], merged_data['elevation_tep'])[0,1]
        
        # Find optimal vertical shift
        optimal_shift, min_rmse = find_optimal_vertical_shift(
            merged_data['elevation_nld'].values,
            merged_data['elevation_tep'].values
        )
        
        # Create scatter plot (top two rows)
        ax_scatter = axes[row_idx, col_idx]
        
        # Plot original data
        ax_scatter.scatter(merged_data['elevation_nld'], merged_data['elevation_tep'], 
                         alpha=0.5, c='blue', s=20, label='Original')
        
        # Plot shifted data
        shifted_tep = merged_data['elevation_tep'] + optimal_shift
        ax_scatter.scatter(merged_data['elevation_nld'], shifted_tep,
                         alpha=0.5, c='green', s=20, label='Shifted')
        
        # Add 1:1 line for scatter plot
        min_elev = min(merged_data['elevation_nld'].min(), merged_data['elevation_tep'].min())
        max_elev = max(merged_data['elevation_nld'].max(), merged_data['elevation_tep'].max())
        ax_scatter.plot([min_elev, max_elev], [min_elev, max_elev], 'r--', label='1:1 line')
        
        # Format scatter plot
        ax_scatter.set_xlabel('NLD Elevation (m)', fontsize=8)
        ax_scatter.set_ylabel('TEP Elevation (m)', fontsize=8)
        ax_scatter.set_title(f'System ID: {system_id}\nCorr: {corr_coef:.4f}, Shift: {optimal_shift:.2f}m', 
                           fontsize=9)
        ax_scatter.grid(True, linestyle='--', alpha=0.7)
        ax_scatter.legend(fontsize=8)
        ax_scatter.set_aspect('equal')
        
        # Create profile plot (bottom two rows)
        row_idx_profile = idx // 5 + 2
        col_idx_profile = idx % 5
        ax_profile = axes[row_idx_profile, col_idx_profile]
        
        # Plot profiles
        ax_profile.plot(nld_data['distance_along_track'], nld_data['elevation'], 'b-', 
                       linewidth=1, label='NLD')
        ax_profile.plot(nld_data['distance_along_track'], nld_data['elevation'], 'bo', 
                       markersize=4)
        # Plot both original and shifted TEP profiles
        ax_profile.plot(tep_data['distance_along_track'], tep_data['elevation'], 'r-', 
                       linewidth=1, label='TEP')
        ax_profile.plot(tep_data['distance_along_track'], tep_data['elevation'], 'ro', 
                       markersize=4)
        ax_profile.plot(tep_data['distance_along_track'], tep_data['elevation'] + optimal_shift, 'g-', 
                       linewidth=1, label='TEP Shifted')
        ax_profile.plot(tep_data['distance_along_track'], tep_data['elevation'] + optimal_shift, 'go', 
                       markersize=4)
        
        # Format profile plot
        ax_profile.set_xlabel('Distance Along Track (m)', fontsize=8)
        ax_profile.set_ylabel('Elevation (m)', fontsize=8)
        ax_profile.set_title(f'System ID: {system_id}\nCorr: {corr_coef:.4f}, Shift: {optimal_shift:.2f}m', 
                           fontsize=9)
        ax_profile.grid(True, linestyle='--', alpha=0.7)
        ax_profile.legend(fontsize=8)
    
    plt.tight_layout()
    plt.show()

def plot_elevation_differences(filtered_significant, system_ids):
    """
    Create a figure showing elevation differences after shifting and their distributions
    """
    # Create a 4x5 subplot layout (2 rows for differences, 2 rows for histograms)
    fig, axes = plt.subplots(4, 5, figsize=(35, 16))
    
    for idx, system_id in enumerate(system_ids):
        if system_id not in filtered_significant['system_id'].unique():
            print(f"System ID {system_id} not found in significant differences dataset.")
            continue
        
        # Get data for this system
        system_data = filtered_significant[filtered_significant['system_id'] == system_id]
        
        # Merge NLD and TEP data
        nld_data = system_data[system_data['source'] == 'nld'].sort_values('distance_along_track')
        tep_data = system_data[system_data['source'] == 'tep'].sort_values('distance_along_track')
        merged_data = pd.merge(
            nld_data[['distance_along_track', 'elevation']], 
            tep_data[['distance_along_track', 'elevation']], 
            on='distance_along_track', 
            suffixes=('_nld', '_tep')
        )
        
        # Find optimal vertical shift
        optimal_shift, min_rmse = find_optimal_vertical_shift(
            merged_data['elevation_nld'].values,
            merged_data['elevation_tep'].values
        )
        
        # Calculate differences
        original_diff = merged_data['elevation_nld'] - merged_data['elevation_tep']
        shifted_diff = merged_data['elevation_nld'] - (merged_data['elevation_tep'] + optimal_shift)
        
        # Calculate statistics
        original_std = original_diff.std()
        shifted_std = shifted_diff.std()
        original_mean = original_diff.mean()
        shifted_mean = shifted_diff.mean()
        
        # Plot differences (top two rows)
        ax_diff = axes[idx // 5, idx % 5]
        ax_diff.plot(merged_data['distance_along_track'], original_diff, 'b-', 
                    label=f'Original (std: {original_std:.2f}m)', alpha=0.5)
        ax_diff.plot(merged_data['distance_along_track'], shifted_diff, 'r-', 
                    label=f'After Shift (std: {shifted_std:.2f}m)', alpha=0.5)
        
        # Add horizontal line at zero
        ax_diff.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Set y-axis limits
        ax_diff.set_ylim([-6, 6])
        
        # Format difference plot
        ax_diff.set_xlabel('Distance Along Track (m)', fontsize=8)
        ax_diff.set_ylabel('Elevation Difference (m)\n(NLD - TEP)', fontsize=8)
        ax_diff.set_title(f'System ID: {system_id}\nShift: {optimal_shift:.2f}m', fontsize=9)
        ax_diff.grid(True, linestyle='--', alpha=0.7)
        ax_diff.legend(fontsize=8)
        
        # Plot histograms (bottom two rows)
        ax_hist = axes[idx // 5 + 2, idx % 5]
        
        # Set fixed bins between -6 and 6
        bins = np.linspace(-6, 6, 30)
        
        ax_hist.hist(original_diff, bins=bins, alpha=0.3, color='blue', 
                    density=True, label=f'Original\nμ={original_mean:.2f}m')
        ax_hist.hist(shifted_diff, bins=bins, alpha=0.3, color='red', 
                    density=True, label=f'After Shift\nμ={shifted_mean:.2f}m')
        
        # Add vertical lines for means
        ax_hist.axvline(x=original_mean, color='blue', linestyle='--', alpha=0.5)
        ax_hist.axvline(x=shifted_mean, color='red', linestyle='--', alpha=0.5)
        ax_hist.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        
        # Set x-axis limits
        ax_hist.set_xlim([-6, 6])
        
        # Format histogram plot
        ax_hist.set_xlabel('Elevation Difference (m)', fontsize=8)
        ax_hist.set_ylabel('Density', fontsize=8)
        ax_hist.set_title('Difference Distribution', fontsize=9)
        ax_hist.grid(True, linestyle='--', alpha=0.7)
        ax_hist.legend(fontsize=8)
    
    plt.tight_layout()
    plt.show()

def analyze_specific_systems(filtered_significant, system_ids=[5505000014, 6005000545]):
    """
    Detailed analysis of differences after shift for specific systems
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for idx, system_id in enumerate(system_ids):
        if system_id not in filtered_significant['system_id'].unique():
            print(f"System ID {system_id} not found in significant differences dataset.")
            continue
        
        # Get data for this system
        system_data = filtered_significant[filtered_significant['system_id'] == system_id]
        
        # Merge NLD and TEP data
        nld_data = system_data[system_data['source'] == 'nld'].sort_values('distance_along_track')
        tep_data = system_data[system_data['source'] == 'tep'].sort_values('distance_along_track')
        merged_data = pd.merge(
            nld_data[['distance_along_track', 'elevation']], 
            tep_data[['distance_along_track', 'elevation']], 
            on='distance_along_track', 
            suffixes=('_nld', '_tep')
        )
        
        # Find optimal vertical shift
        optimal_shift, min_rmse = find_optimal_vertical_shift(
            merged_data['elevation_nld'].values,
            merged_data['elevation_tep'].values
        )
        
        # Calculate differences after shift
        shifted_diff = merged_data['elevation_nld'] - (merged_data['elevation_tep'] + optimal_shift)
        
        # Calculate statistics
        mean_diff = shifted_diff.mean()
        std_diff = shifted_diff.std()
        median_diff = shifted_diff.median()
        q1_diff = shifted_diff.quantile(0.25)
        q3_diff = shifted_diff.quantile(0.75)
        
        # Plot difference along track
        ax_diff = axes[idx, 0]
        ax_diff.plot(merged_data['distance_along_track'], shifted_diff, 'b-', 
                    label=f'Difference after shift', alpha=0.7)
        ax_diff.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax_diff.set_ylim([-6, 6])
        
        # Format difference plot
        ax_diff.set_xlabel('Distance Along Track (m)')
        ax_diff.set_ylabel('Elevation Difference (m)\n(NLD - TEP)')
        ax_diff.set_title(f'System ID: {system_id}\nShift: {optimal_shift:.2f}m')
        ax_diff.grid(True)
        
        # Add statistics to plot
        stats_text = f'Statistics:\n' \
                    f'Mean: {mean_diff:.2f}m\n' \
                    f'Median: {median_diff:.2f}m\n' \
                    f'Std Dev: {std_diff:.2f}m\n' \
                    f'Q1: {q1_diff:.2f}m\n' \
                    f'Q3: {q3_diff:.2f}m'
        ax_diff.text(0.02, 0.98, stats_text,
                    transform=ax_diff.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot histogram
        ax_hist = axes[idx, 1]
        bins = np.linspace(-6, 6, 30)
        ax_hist.hist(shifted_diff, bins=bins, density=True, alpha=0.7,
                    label=f'μ={mean_diff:.2f}m\nσ={std_diff:.2f}m')
        
        # Add vertical lines for mean and median
        ax_hist.axvline(x=mean_diff, color='r', linestyle='--', 
                       label='Mean', alpha=0.7)
        ax_hist.axvline(x=median_diff, color='g', linestyle='--', 
                       label='Median', alpha=0.7)
        ax_hist.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        
        # Format histogram
        ax_hist.set_xlim([-6, 6])
        ax_hist.set_xlabel('Elevation Difference (m)')
        ax_hist.set_ylabel('Density')
        ax_hist.set_title('Difference Distribution')
        ax_hist.grid(True)
        ax_hist.legend()
    
    plt.tight_layout()
    plt.show()

def analyze_difference_patterns(filtered_significant, system_ids=[2205000054, 4705000101, 6005000545, 5005000043]):
    """
    Analyze and visualize different patterns in the elevation differences using peak detection
    """
    # Create a 4x2 subplot layout
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    
    for idx, system_id in enumerate(system_ids):
        # Get data for this system
        system_data = filtered_significant[filtered_significant['system_id'] == system_id]
        
        # Merge NLD and TEP data
        nld_data = system_data[system_data['source'] == 'nld'].sort_values('distance_along_track')
        tep_data = system_data[system_data['source'] == 'tep'].sort_values('distance_along_track')
        merged_data = pd.merge(
            nld_data[['distance_along_track', 'elevation']], 
            tep_data[['distance_along_track', 'elevation']], 
            on='distance_along_track', 
            suffixes=('_nld', '_tep')
        )
        
        # Find optimal vertical shift
        optimal_shift, min_rmse = find_optimal_vertical_shift(
            merged_data['elevation_nld'].values,
            merged_data['elevation_tep'].values
        )
        
        # Calculate differences after shift
        diff_data = merged_data['elevation_nld'] - (merged_data['elevation_tep'] + optimal_shift)
        distance = merged_data['distance_along_track']
        
        # Calculate rolling statistics
        window = 20  # adjust window size as needed
        rolling_mean = pd.Series(diff_data).rolling(window=window, center=True).mean()
        rolling_std = pd.Series(diff_data).rolling(window=window, center=True).std()
        
        # Define thresholds for peak detection (e.g., 3 standard deviations)
        threshold = 3
        upper_bound = rolling_mean + threshold * rolling_std
        lower_bound = rolling_mean - threshold * rolling_std
        
        # Identify significant peaks (signals)
        peaks = (diff_data > upper_bound) | (diff_data < lower_bound)
        
        # Plot time domain analysis
        ax_time = axes[idx, 0]
        # Plot original differences
        ax_time.plot(distance, diff_data, 'b-', alpha=0.7, label='Difference')
        # Plot rolling mean
        ax_time.plot(distance, rolling_mean, 'g-', alpha=0.7, label='Rolling Mean')
        # Plot threshold bounds
        ax_time.plot(distance, upper_bound, 'r--', alpha=0.5, label=f'{threshold}σ Bounds')
        ax_time.plot(distance, lower_bound, 'r--', alpha=0.5)
        # Highlight peaks
        ax_time.scatter(distance[peaks], diff_data[peaks], color='red', s=50, 
                       alpha=0.5, label='Significant Peaks')
        
        # Format time domain plot
        ax_time.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax_time.set_xlabel('Distance Along Track (m)')
        ax_time.set_ylabel('Elevation Difference (m)')
        ax_time.set_title(f'System ID: {system_id}\nPeak Detection Analysis')
        ax_time.grid(True)
        ax_time.legend()
        ax_time.set_ylim([-3, 3])
        
        # Plot histogram with thresholds
        ax_hist = axes[idx, 1]
        bins = np.linspace(-3, 3, 50)
        ax_hist.hist(diff_data, bins=bins, density=True, alpha=0.7, color='blue')
        
        # Add vertical lines for thresholds
        mean_diff = np.mean(diff_data)
        std_diff = np.std(diff_data)
        ax_hist.axvline(x=mean_diff, color='g', linestyle='-', 
                       label='Mean', alpha=0.7)
        ax_hist.axvline(x=mean_diff + threshold*std_diff, color='r', linestyle='--', 
                       label=f'{threshold}σ Bounds', alpha=0.7)
        ax_hist.axvline(x=mean_diff - threshold*std_diff, color='r', linestyle='--', 
                       alpha=0.7)
        
        # Format histogram plot
        ax_hist.set_xlabel('Elevation Difference (m)')
        ax_hist.set_ylabel('Density')
        ax_hist.set_title('Difference Distribution with Thresholds')
        ax_hist.grid(True)
        ax_hist.legend()
        
        # Add statistics
        num_peaks = np.sum(peaks)
        peak_percentage = (num_peaks / len(diff_data)) * 100
        stats_text = f'Statistics:\n' \
                    f'Window Size: {window}m\n' \
                    f'Threshold: {threshold}σ\n' \
                    f'Peaks Found: {num_peaks}\n' \
                    f'Peak %: {peak_percentage:.1f}%\n' \
                    f'Mean: {mean_diff:.3f}m\n' \
                    f'Std: {std_diff:.3f}m'
        ax_time.text(0.02, 0.98, stats_text,
                    transform=ax_time.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def plot_all_significant_scatter(filtered_significant):
    """
    Create scatter plots for all significant profiles in a 7x8 grid
    """
    # Get unique system IDs
    system_ids = filtered_significant['system_id'].unique()
    n_plots = len(system_ids)
    plots_per_fig = 56  # 7 rows * 8 columns
    n_figs = (n_plots + plots_per_fig - 1) // plots_per_fig  # Ceiling division
    
    for fig_num in range(n_figs):
        # Create figure with 7x8 subplots
        fig, axes = plt.subplots(7, 8, figsize=(32, 28))
        fig.suptitle(f'Elevation Scatter Plots (Page {fig_num + 1})', fontsize=16)
        
        # Flatten axes for easier indexing
        axes_flat = axes.flatten()
        
        # Get the subset of system_ids for this figure
        start_idx = fig_num * plots_per_fig
        end_idx = min((fig_num + 1) * plots_per_fig, n_plots)
        current_ids = system_ids[start_idx:end_idx]
        
        for idx, system_id in enumerate(current_ids):
            ax = axes_flat[idx]
            
            # Get data for this system
            system_data = filtered_significant[filtered_significant['system_id'] == system_id]
            
            # Merge NLD and TEP data
            nld_data = system_data[system_data['source'] == 'nld'].sort_values('distance_along_track')
            tep_data = system_data[system_data['source'] == 'tep'].sort_values('distance_along_track')
            merged_data = pd.merge(
                nld_data[['distance_along_track', 'elevation']], 
                tep_data[['distance_along_track', 'elevation']], 
                on='distance_along_track', 
                suffixes=('_nld', '_tep')
            )
            
            # Calculate correlation coefficient
            corr_coef = np.corrcoef(merged_data['elevation_nld'], merged_data['elevation_tep'])[0,1]
            
            # Create scatter plot
            ax.scatter(merged_data['elevation_nld'], merged_data['elevation_tep'], 
                      alpha=0.5, c='blue', s=10)
            
            # Add 1:1 line
            min_elev = min(merged_data['elevation_nld'].min(), merged_data['elevation_tep'].min())
            max_elev = max(merged_data['elevation_nld'].max(), merged_data['elevation_tep'].max())
            ax.plot([min_elev, max_elev], [min_elev, max_elev], 'r--', linewidth=1)
            
            # Format plot
            ax.set_title(f'ID: {system_id}\nρ: {corr_coef:.3f}', fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Only show y-axis labels for leftmost plots
            if idx % 8 != 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel('TEP Elevation (m)', fontsize=8)
            
            # Only show x-axis labels for bottom plots
            if idx < (end_idx - 8):
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('NLD Elevation (m)', fontsize=8)
            
            # Set equal aspect ratio
            ax.set_aspect('equal')
        
        # Turn off any unused subplots
        for idx in range(len(current_ids), plots_per_fig):
            axes_flat[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'significant_scatter_plots_page{fig_num+1}.png'))
        plt.close()

def analyze_profile_correlation(filtered_significant):
    """
    Analyze profile correlations using Orthogonal Distance Regression.
    Returns profiles with significant deviation from 1:1 relationship.
    """
    def linear_func(p, x):
        """Linear function y = mx + b"""
        m, b = p
        return m*x + b
    
    problematic_profiles = []
    
    # Get unique system IDs
    system_ids = filtered_significant['system_id'].unique()
    
    for system_id in system_ids:
        # Get data for this system
        df_nld = filtered_significant[
            (filtered_significant['system_id'] == system_id) & 
            (filtered_significant['source'] == 'nld')]
        df_3dep = filtered_significant[
            (filtered_significant['system_id'] == system_id) & 
            (filtered_significant['source'] == 'tep')]
        
        if not df_nld.empty and not df_3dep.empty:
            # Merge the dataframes
            merged = pd.merge(df_nld, df_3dep, on=['system_id', 'distance_along_track'], 
                            suffixes=('_nld', '_tep'))
            
            # Perform ODR
            linear = odr.Model(linear_func)
            data = odr.RealData(merged['elevation_nld'], merged['elevation_tep'])
            odr_obj = odr.ODR(data, linear, beta0=[1., 0.])
            results = odr_obj.run()
            
            # Get slope and R-squared
            slope = results.beta[0]
            intercept = results.beta[1]
            
            # Calculate R-squared
            y_pred = linear_func(results.beta, merged['elevation_nld'])
            r_squared = np.corrcoef(merged['elevation_tep'], y_pred)[0,1]**2
            
            # Check if profile deviates significantly from 1:1
            if (abs(slope - 1) > 0.1) or (r_squared < 0.95):  # Adjust thresholds as needed
                problematic_profiles.append({
                    'system_id': system_id,
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_squared
                })
    
    return pd.DataFrame(problematic_profiles)

def plot_poor_correlation_scatter(filtered_significant, problematic_profiles):
    """
    Create scatter plots for profiles with poor correlation in a 7x8 grid
    """
    # Get system IDs from problematic profiles
    system_ids = problematic_profiles['system_id'].values
    n_plots = len(system_ids)
    plots_per_fig = 56  # 7 rows * 8 columns
    n_figs = (n_plots + plots_per_fig - 1) // plots_per_fig  # Ceiling division
    
    for fig_num in range(n_figs):
        # Create figure with 7x8 subplots
        fig, axes = plt.subplots(7, 8, figsize=(32, 28))
        fig.suptitle(f'Poor Correlation Scatter Plots (Page {fig_num + 1})', fontsize=16)
        
        # Flatten axes for easier indexing
        axes_flat = axes.flatten()
        
        # Get the subset of system_ids for this figure
        start_idx = fig_num * plots_per_fig
        end_idx = min((fig_num + 1) * plots_per_fig, n_plots)
        current_ids = system_ids[start_idx:end_idx]
        
        for idx, system_id in enumerate(current_ids):
            ax = axes_flat[idx]
            
            # Get data for this system
            df_nld = filtered_significant[
                (filtered_significant['system_id'] == system_id) & 
                (filtered_significant['source'] == 'nld')]
            df_3dep = filtered_significant[
                (filtered_significant['system_id'] == system_id) & 
                (filtered_significant['source'] == 'tep')]
            
            if not df_nld.empty and not df_3dep.empty:
                # Merge the dataframes
                merged = pd.merge(df_nld, df_3dep, on=['system_id', 'distance_along_track'], 
                                suffixes=('_nld', '_tep'))
                
                # Get ODR results for this profile
                profile_stats = problematic_profiles[problematic_profiles['system_id'] == system_id].iloc[0]
                slope = profile_stats['slope']
                intercept = profile_stats['intercept']
                r_squared = profile_stats['r_squared']
                
                # Create scatter plot
                ax.scatter(merged['elevation_nld'], merged['elevation_tep'], 
                          alpha=0.5, c='blue', s=10)
                
                # Add 1:1 line
                min_elev = min(merged['elevation_nld'].min(), merged['elevation_tep'].min())
                max_elev = max(merged['elevation_nld'].max(), merged['elevation_tep'].max())
                ax.plot([min_elev, max_elev], [min_elev, max_elev], 'r--', 
                       linewidth=1, label='1:1 Line')
                
                # Add regression line
                x_range = np.array([min_elev, max_elev])
                y_range = slope * x_range + intercept
                ax.plot(x_range, y_range, 'g-', linewidth=1, 
                       label=f'Slope={slope:.2f}')
                
                # Format plot
                ax.set_title(f'ID: {system_id}\nR²: {r_squared:.3f}', fontsize=8)
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Only show y-axis labels for leftmost plots
                if idx % 8 != 0:
                    ax.set_yticklabels([])
                else:
                    ax.set_ylabel('TEP Elevation (m)', fontsize=8)
                
                # Only show x-axis labels for bottom plots
                if idx < (end_idx - 8):
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel('NLD Elevation (m)', fontsize=8)
                
                # Set equal aspect ratio
                ax.set_aspect('equal')
                
                # Add legend to first plot only
                if idx == 0:
                    ax.legend(fontsize=6)
        
        # Turn off any unused subplots
        for idx in range(len(current_ids), plots_per_fig):
            axes_flat[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'poor_correlation_scatter_plots_page{fig_num+1}.png'))
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
    
    # Plot the original scatter and profile plots
    plot_scatter_profile(filtered_significant, system_id_examples)
    
    # Plot the elevation differences
    plot_elevation_differences(filtered_significant, system_id_examples)
    
    # Add the new analysis
    analyze_specific_systems(filtered_significant)
    analyze_difference_patterns(filtered_significant)
    
    # After creating filtered_significant
    print("\nPlotting scatter plots for all significant profiles...")
    plot_all_significant_scatter(filtered_significant)
    
    print("\nAnalyzing profile correlations...")
    problematic_profiles = analyze_profile_correlation(filtered_significant)
    print(f"\nFound {len(problematic_profiles)} profiles with poor 1:1 correlation:")
    print(problematic_profiles.sort_values('r_squared'))
    
    print("\nPlotting scatter plots for poorly correlated profiles...")
    plot_poor_correlation_scatter(filtered_significant, problematic_profiles)

