import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import os
from shapely.geometry import Point
from shapely.geometry import LineString
from scipy.stats import zscore, linregress

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

    # Perform linear regression on the normalized profiles
    slope, intercept, r_value, p_value, std_err = linregress(nld_norm, dep_norm)

    # Calculate the residuals
    residuals = dep_norm - (slope * nld_norm + intercept)

    # Calculate the standard deviation of the residuals 
    residuals_std = np.std(residuals)
    
    return nld_norm, dep_norm, residuals, residuals_std, r_value, p_value, std_err

def plot_normalized_scatter(plot_df, filtered_significant, output_dir):
    """
    Plot scatter plots of normalized profiles for all significant cases.
    Creates figures with 7x8 subplots each.
    """
    # Get unique system IDs from filtered_significant
    significant_ids = filtered_significant['system_id'].unique()
    n_plots = len(significant_ids)
    plots_per_fig = 56  # 7 rows * 8 columns
    n_figs = (n_plots + plots_per_fig - 1) // plots_per_fig
    
    for fig_num in range(n_figs):
        fig, axes = plt.subplots(7, 8, figsize=(32, 28))
        axes_flat = axes.flatten()
        
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
                
                # Get normalized profiles and regression stats
                nld_norm, dep_norm, residuals, residuals_std, r_value, p_value, std_err = diff_normalize(
                    merged['elevation_nld'], merged['elevation_tep'])
                
                # Calculate slope and intercept from linregress
                slope, intercept, _, _, _ = linregress(nld_norm, dep_norm)
                
                # Calculate r_squared
                r_squared = r_value**2
                
                # Create scatter plot with conditional coloring
                if slope < 0:
                    scatter_color = 'green'
                else:
                    scatter_color = 'red' if r_squared < 0.5 else 'black'
                    
                ax.scatter(nld_norm, dep_norm, alpha=0.5, c=scatter_color, s=16)
                
                # Add diagonal line in black
                lims = [
                    min(ax.get_xlim()[0], ax.get_ylim()[0]),
                    max(ax.get_xlim()[1], ax.get_ylim()[1]),
                ]
                ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)  # 'k--' means black dashed line
                
                # Set equal limits
                ax.set_xlim(-4, 4)
                ax.set_ylim(-4, 4)
                
                # Update title with all statistics including residuals_std
                ax.set_title(f'ID: {system_id}\nslope: {slope:.3f}, intercept: {slope:.3f}\n'
                           f'r²: {r_value**2:.3f}, p: {p_value:.3e}\n'
                           f'std_err: {std_err:.3f}, res_std: {residuals_std:.3f}', 
                           fontsize=8)
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
        # Save to normalized_profiles directory
        plt.savefig(os.path.join(output_dir, f'normalized_scatter_plots_page{fig_num+1}.png'))
        plt.close()

def diff_optimal_shift(profile_nld, profile_3dep):
    """
    Shift the 3DEP profile by finding the optimal vertical shift that minimizes RMSE.
    
    Args:
        profile_nld (array-like): NLD elevation profile
        profile_3dep (array-like): 3DEP elevation profile
    
    Returns:
        tuple: Original NLD and optimally shifted 3DEP profiles, along with statistics
    """
    # Convert to numpy arrays
    profile_nld = np.array(profile_nld)
    profile_3dep = np.array(profile_3dep)
    
    # Find optimal shift
    optimal_shift, min_rmse = find_optimal_vertical_shift(profile_nld, profile_3dep)
    profile_3dep_shifted = profile_3dep + optimal_shift
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(profile_nld, profile_3dep_shifted)
    
    # Calculate residuals
    residuals = profile_3dep_shifted - (slope * profile_nld + intercept)
    residuals_std = np.std(residuals)
    
    return profile_nld, profile_3dep_shifted, residuals, residuals_std, r_value, p_value, std_err, optimal_shift, min_rmse

def plot_optimal_shifted_scatter(plot_df, filtered_significant, output_dir):
    """
    Plot scatter plots of optimally shifted profiles for all significant cases.
    Creates figures with 7x8 subplots each.
    """
    significant_ids = filtered_significant['system_id'].unique()
    n_plots = len(significant_ids)
    plots_per_fig = 56  # 7 rows * 8 columns
    n_figs = (n_plots + plots_per_fig - 1) // plots_per_fig
    
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
                
                nld_orig, dep_shifted, residuals, residuals_std, r_value, p_value, std_err, opt_shift, min_rmse = diff_optimal_shift(
                    merged['elevation_nld'], merged['elevation_tep'])
                
                slope, intercept, _, _, _ = linregress(nld_orig, dep_shifted)
                r_squared = r_value**2
                
                # Updated conditional coloring
                if slope < 0:
                    scatter_color = 'green'
                elif slope > 1.5 or slope < 0.5:
                    scatter_color = 'purple'
                else:
                    scatter_color = 'red' if r_squared < 0.5 else 'black'
                    
                ax.scatter(nld_orig, dep_shifted, alpha=0.5, c=scatter_color, s=16)
                
                # Add diagonal line
                lims = [
                    min(ax.get_xlim()[0], ax.get_ylim()[0]),
                    max(ax.get_xlim()[1], ax.get_ylim()[1]),
                ]
                ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
                
                ax.set_title(f'ID: {system_id}\nslope: {slope:.3f}, shift: {opt_shift:.3f}\n'
                           f'r²: {r_value**2:.3f}, RMSE: {min_rmse:.3f}\n'
                           f'std_err: {std_err:.3f}, res_std: {residuals_std:.3f}', 
                           fontsize=8)
                ax.grid(True)
                
                if idx % 8 != 0:
                    ax.set_yticklabels([])
                else:
                    ax.set_ylabel('3DEP Shifted', fontsize=8)
                
                if idx < (end_idx - 8):
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel('NLD Original', fontsize=8)
                
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
    
    # Create a subdirectory for the normalized profile plots
    normalized_plots_dir = os.path.join(output_dir, 'normalized_profiles')
    os.makedirs(normalized_plots_dir, exist_ok=True)
    
    print("\nPlotting normalized scatter plots...")
    plot_normalized_scatter(filtered_plot_df, filtered_significant, normalized_plots_dir)
    
    print("\nPlotting optimally shifted scatter plots...")
    plot_optimal_shifted_scatter(filtered_plot_df, filtered_significant, normalized_plots_dir)