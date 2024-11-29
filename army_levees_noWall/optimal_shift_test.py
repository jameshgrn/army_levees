import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import os
from shapely.geometry import Point
from shapely.geometry import LineString
from scipy.stats import zscore, linregress
import scipy.stats as stats

# Todo: maybe add outlier detection will help


# Directory where the elevation data files are saved
data_dir = '/Users/liyuan/projects/army_levees/data_no_floodwall'
output_dir = '/Users/liyuan/projects/army_levees/army_levees_noWall/plots'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

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

def diff_optimal_shift(profile_nld, profile_3dep):
    """
    Shift the 3DEP profile by finding the optimal vertical shift that minimizes RMSE.
    """
    # Convert to numpy arrays
    profile_nld = np.array(profile_nld)
    profile_3dep = np.array(profile_3dep)
    
    # Find optimal shift
    optimal_shift, min_rmse = find_optimal_vertical_shift(profile_nld, profile_3dep)
    profile_3dep_shifted = profile_3dep + optimal_shift
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(profile_nld, profile_3dep_shifted)
    
    return profile_nld, profile_3dep_shifted, r_value, p_value, std_err, optimal_shift, min_rmse

def plot_difference_profiles(plot_df, filtered_significant, output_dir):
    """
    Plot elevation difference profiles with rolling statistics and peak detection.
    Returns set of system IDs with peaks detected.
    """
    significant_ids = filtered_significant['system_id'].unique()
    n_plots = len(significant_ids)
    plots_per_fig = 56
    n_figs = (n_plots + plots_per_fig - 1) // plots_per_fig
    window = 20  # rolling window size
    
    # Dictionary to store systems with peaks and their peak counts
    systems_with_peaks = {}
    
    for fig_num in range(n_figs):
        fig, axes = plt.subplots(7, 8, figsize=(32, 28))
        axes_flat = axes.flatten()
        
        start_idx = fig_num * plots_per_fig
        end_idx = min((fig_num + 1) * plots_per_fig, n_plots)
        current_ids = significant_ids[start_idx:end_idx]
        
        for idx, system_id in enumerate(current_ids):
            ax = axes_flat[idx]
            
            df_nld = plot_df[(plot_df['system_id'] == system_id) & (plot_df['source'] == 'nld')]
            df_3dep = plot_df[(plot_df['system_id'] == system_id) & (plot_df['source'] == 'tep')]
            
            if not df_nld.empty and not df_3dep.empty:
                merged = pd.merge(df_nld, df_3dep, on=['system_id', 'distance_along_track'], 
                                suffixes=('_nld', '_tep'))
                
                nld_orig, dep_shifted, r_value, p_value, std_err, opt_shift, min_rmse = diff_optimal_shift(
                    merged['elevation_nld'], merged['elevation_tep'])
                
                slope, intercept, _, _, _ = linregress(nld_orig, dep_shifted)
                r_squared = r_value**2
                
                # Check if this would be a black point (good alignment)
                if slope >= 0.5 and slope <= 1.5 and r_squared >= 0.5:
                    # Sort by distance along track
                    merged = merged.sort_values('distance_along_track')
                    
                    # Calculate difference
                    difference = dep_shifted - nld_orig
                    
                    # Calculate rolling statistics
                    rolling_mean = pd.Series(difference).rolling(window=window, center=True).mean()
                    rolling_std = pd.Series(difference).rolling(window=window, center=True).std()
                    
                    # Define thresholds for peak detection
                    threshold = 3
                    upper_bound = rolling_mean + threshold * rolling_std
                    lower_bound = rolling_mean - threshold * rolling_std
                    
                    # Identify significant peaks
                    peaks = (difference > upper_bound) | (difference < lower_bound)
                    
                    # If peaks are detected, store system info
                    if peaks.any():
                        n_peaks = peaks.sum()
                        systems_with_peaks[system_id] = {
                            'num_peaks': n_peaks,
                            'rmse': min_rmse,
                            'r_squared': r_squared,
                            'slope': slope
                        }
                    
                    # Plot base difference profile
                    ax.plot(merged['distance_along_track'], difference, 'k-', linewidth=1)
                    
                    # Plot bounds
                    ax.plot(merged['distance_along_track'], upper_bound, 'r--', alpha=0.5, linewidth=0.5)
                    ax.plot(merged['distance_along_track'], lower_bound, 'r--', alpha=0.5, linewidth=0.5)
                    
                    # Plot zero line
                    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                    
                    # Plot peaks as red dots
                    peak_distances = merged['distance_along_track'][peaks]
                    peak_differences = difference[peaks]
                    ax.scatter(peak_distances, peak_differences, c='red', s=10, alpha=0.7)
                    
                    # Update title to include peak count if peaks exist
                    title = f'ID: {system_id}\nshift: {opt_shift:.3f}m, RMSE: {min_rmse:.3f}m'
                    if peaks.any():
                        title += f'\nPeaks: {n_peaks}'
                    ax.set_title(title, fontsize=8)
                    
                    ax.grid(True)
                    ax.set_ylabel('Elevation Difference (m)', fontsize=8)
                    ax.set_xlabel('Distance Along Track (m)', fontsize=8)
                    ax.tick_params(axis='both', labelsize=6)
                else:
                    # Turn off display for non-well-aligned systems
                    ax.set_visible(False)
        
        # Turn off any remaining unused subplots
        for idx in range(len(current_ids), plots_per_fig):
            axes_flat[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'optimal_shifted_diff_profiles_page{fig_num+1}.png'))
        plt.close()
    
    # Print summary of systems with peaks
    if systems_with_peaks:
        print("\nSystems with detected peaks:")
        print(f"Total number of systems with peaks: {len(systems_with_peaks)}")
        print("\nDetailed information:")
        print(f"{'System ID':<15} {'Num Peaks':<10} {'RMSE':<10} {'R²':<10} {'Slope':<10}")
        print("-" * 55)
        for system_id, info in sorted(systems_with_peaks.items(), key=lambda x: x[1]['num_peaks'], reverse=True):
            print(f"{system_id:<15} {info['num_peaks']:<10} {info['rmse']:<10.3f} {info['r_squared']:<10.3f} {info['slope']:<10.3f}")
    else:
        print("\nNo systems with peaks detected.")
    
    return set(systems_with_peaks.keys())  # Return the set of system IDs with peaks

def plot_optimal_shifted_scatter(plot_df, filtered_significant, output_dir, peak_system_ids=None):
    """
    Plot scatter plots with regression lines and highlight systems with peaks.
    Args:
        plot_df: DataFrame with plot data
        filtered_significant: DataFrame with significant systems
        output_dir: Directory for output files
        peak_system_ids: Set of system IDs that have peaks detected
    """
    significant_ids = filtered_significant['system_id'].unique()
    n_plots = len(significant_ids)
    plots_per_fig = 56
    n_figs = (n_plots + plots_per_fig - 1) // plots_per_fig
    
    peak_system_ids = peak_system_ids or set()
    
    for fig_num in range(n_figs):
        fig, axes = plt.subplots(7, 8, figsize=(32, 28))
        axes_flat = axes.flatten()
        
        start_idx = fig_num * plots_per_fig
        end_idx = min((fig_num + 1) * plots_per_fig, n_plots)
        current_ids = significant_ids[start_idx:end_idx]
        
        for idx, system_id in enumerate(current_ids):
            ax = axes_flat[idx]
            
            df_nld = plot_df[(plot_df['system_id'] == system_id) & (plot_df['source'] == 'nld')]
            df_3dep = plot_df[(plot_df['system_id'] == system_id) & (plot_df['source'] == 'tep')]
            
            if not df_nld.empty and not df_3dep.empty:
                merged = pd.merge(df_nld, df_3dep, on=['system_id', 'distance_along_track'], 
                                suffixes=('_nld', '_tep'))
                
                nld_orig, dep_shifted, r_value, p_value, std_err, opt_shift, min_rmse = diff_optimal_shift(
                    merged['elevation_nld'], merged['elevation_tep'])
                
                slope, intercept, _, _, _ = linregress(nld_orig, dep_shifted)
                r_squared = r_value**2
                
                # Updated conditional coloring
                if slope < 0:
                    scatter_color = 'green'
                    line_color = 'darkgreen'
                elif slope > 1.5 or slope < 0.5:
                    scatter_color = 'purple'
                    line_color = 'darkviolet'
                else:
                    if r_squared < 0.5:
                        scatter_color = 'red'
                        line_color = 'darkred'
                    else:
                        # Check if system has peaks
                        if system_id in peak_system_ids:
                            scatter_color = 'blue'
                            line_color = 'darkblue'
                        else:
                            scatter_color = 'black'
                            line_color = 'blue'
                
                # Plot scatter points
                ax.scatter(nld_orig, dep_shifted, alpha=0.5, c=scatter_color, s=16)
                
                # Add diagonal line
                lims = [
                    min(ax.get_xlim()[0], ax.get_ylim()[0]),
                    max(ax.get_xlim()[1], ax.get_ylim()[1]),
                ]
                ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
                
                # Add regression line with confidence interval
                x_new = np.linspace(min(nld_orig), max(nld_orig), 100)
                y_new = slope * x_new + intercept
                
                # Calculate confidence interval
                n = len(nld_orig)
                x_mean = np.mean(nld_orig)
                x_ss = np.sum((nld_orig - x_mean)**2)
                y_hat = slope * nld_orig + intercept
                se = np.sqrt(np.sum((dep_shifted - y_hat)**2) / (n-2))
                
                alpha = 0.1  # 90% confidence interval
                t_value = stats.t.ppf(1 - alpha/2, n-2)
                pi = t_value * se * np.sqrt(1/n + (x_new - x_mean)**2 / x_ss)
                
                # Plot regression line and confidence interval
                ax.plot(x_new, y_new, '-', color=line_color, linewidth=1, alpha=0.8)
                ax.fill_between(x_new, y_new - pi, y_new + pi, color=line_color, alpha=0.1)
                
                # Update title to indicate if system has peaks
                title = f'ID: {system_id}\nslope: {slope:.3f}, shift: {opt_shift:.3f}\n'
                title += f'r²: {r_value**2:.3f}, RMSE: {min_rmse:.3f}'
                if system_id in peak_system_ids:
                    title += '\n(Has peaks)'
                ax.set_title(title, fontsize=8)
                
                ax.grid(True)
                ax.set_ylabel('3DEP Shifted', fontsize=8)
                ax.set_xlabel('NLD Original', fontsize=8)
                ax.tick_params(axis='both', labelsize=6)
                ax.set_aspect('equal')
        
        # Turn off unused subplots
        for idx in range(len(current_ids), plots_per_fig):
            axes_flat[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'optimal_shifted_scatter_plots_page{fig_num+1}.png'))
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
    
    # Get all significant system IDs
    significant_ids = filtered_significant['system_id'].unique()
    
    # Create a subdirectory for the plots
    plots_dir = os.path.join(output_dir, 'shifted_profiles')
    os.makedirs(plots_dir, exist_ok=True)
    
    print("\nPlotting difference profiles for well-aligned systems...")
    peak_systems = plot_difference_profiles(filtered_plot_df, filtered_significant, plots_dir)
    
    print("\nPlotting optimally shifted scatter plots...")
    plot_optimal_shifted_scatter(filtered_plot_df, filtered_significant, plots_dir, peak_systems)