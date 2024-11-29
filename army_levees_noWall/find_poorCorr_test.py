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
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

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
        
        # Create scatter plot (top two rows)
        ax_scatter = axes[row_idx, col_idx]
        
        # Plot data
        ax_scatter.scatter(merged_data['elevation_nld'], merged_data['elevation_tep'], 
                         alpha=0.5, c='blue', s=20, label='Data points')
        
        # Add 1:1 line for scatter plot
        min_elev = min(merged_data['elevation_nld'].min(), merged_data['elevation_tep'].min())
        max_elev = max(merged_data['elevation_nld'].max(), merged_data['elevation_tep'].max())
        ax_scatter.plot([min_elev, max_elev], [min_elev, max_elev], 'r--', label='1:1 line')
        
        # Format scatter plot
        ax_scatter.set_xlabel('NLD Elevation (m)', fontsize=8)
        ax_scatter.set_ylabel('TEP Elevation (m)', fontsize=8)
        ax_scatter.set_title(f'System ID: {system_id}\nCorr: {corr_coef:.4f}', 
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
        ax_profile.plot(tep_data['distance_along_track'], tep_data['elevation'], 'r-', 
                       linewidth=1, label='TEP')
        ax_profile.plot(tep_data['distance_along_track'], tep_data['elevation'], 'ro', 
                       markersize=4)
        
        # Format profile plot
        ax_profile.set_xlabel('Distance Along Track (m)', fontsize=8)
        ax_profile.set_ylabel('Elevation (m)', fontsize=8)
        ax_profile.set_title(f'System ID: {system_id}\nCorr: {corr_coef:.4f}', 
                           fontsize=9)
        ax_profile.grid(True, linestyle='--', alpha=0.7)
        ax_profile.legend(fontsize=8)
    
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

def analyze_profile_multiple_metrics(filtered_significant):
    """
    Use multiple metrics to identify problematic profiles
    """
    print("\nAnalyzing profiles using multiple metrics...")
    problematic_profiles = []
    
    for system_id in filtered_significant['system_id'].unique():
        df_nld = filtered_significant[
            (filtered_significant['system_id'] == system_id) & 
            (filtered_significant['source'] == 'nld')]
        df_3dep = filtered_significant[
            (filtered_significant['system_id'] == system_id) & 
            (filtered_significant['source'] == 'tep')]
            
        if not df_nld.empty and not df_3dep.empty:
            # Merge dataframes to ensure aligned points
            merged = pd.merge(df_nld, df_3dep, on=['system_id', 'distance_along_track'], 
                            suffixes=('_nld', '_tep'))
            
            if len(merged) > 0:
                # 1. Calculate correlation
                corr = np.corrcoef(merged['elevation_nld'], merged['elevation_tep'])[0,1]
                
                # 2. Calculate DTW distance
                # Reshape the arrays to 2D for DTW
                nld_seq = merged['elevation_nld'].values.reshape(-1, 1)
                dep_seq = merged['elevation_tep'].values.reshape(-1, 1)
                distance, _ = fastdtw(nld_seq, dep_seq, dist=euclidean)
                dtw_dist = distance / len(merged)
                
                # 3. Calculate shape similarity
                nld_norm = (merged['elevation_nld'] - merged['elevation_nld'].min()) / \
                          (merged['elevation_nld'].max() - merged['elevation_nld'].min())
                dep_norm = (merged['elevation_tep'] - merged['elevation_tep'].min()) / \
                          (merged['elevation_tep'].max() - merged['elevation_tep'].min())
                shape_rmse = np.sqrt(np.mean((nld_norm - dep_norm) ** 2))
                
                # 4. Calculate elevation difference statistics
                elev_diff = merged['elevation_tep'] - merged['elevation_nld']
                mean_diff = np.mean(elev_diff)
                std_diff = np.std(elev_diff)
                
                # 5. Perform ODR for slope analysis
                def linear_func(p, x):
                    return p[0] * x + p[1]
                
                linear = odr.Model(linear_func)
                data = odr.RealData(merged['elevation_nld'], merged['elevation_tep'])
                odr_obj = odr.ODR(data, linear, beta0=[1., 0.])
                results = odr_obj.run()
                slope = results.beta[0]
                
                problematic_profiles.append({
                    'system_id': system_id,
                    'correlation': corr,
                    'dtw_distance': dtw_dist,
                    'shape_rmse': shape_rmse,
                    'mean_diff': mean_diff,
                    'std_diff': std_diff,
                    'slope': slope,
                    'points_compared': len(merged)
                })
    
    df = pd.DataFrame(problematic_profiles)
    
    # Calculate thresholds
    dtw_threshold = df['dtw_distance'].mean() + df['dtw_distance'].std()
    shape_threshold = df['shape_rmse'].mean() + df['shape_rmse'].std()
    
    # Flag profiles as problematic if they meet any of these criteria
    df['is_problematic'] = (
        (df['correlation'] < 0.95) |  # Poor correlation
        (df['dtw_distance'] > dtw_threshold) |  # High DTW distance
        (df['shape_rmse'] > shape_threshold) |  # Poor shape match
        (abs(df['mean_diff']) > 1) |  # Large mean difference
        (df['std_diff'] > 2) |  # High variability in differences
        (abs(df['slope'] - 1) > 0.1)  # Significant deviation from 1:1 slope
    )
    
    # Print summary statistics
    print("\nProfile Analysis Summary:")
    print(f"Total profiles analyzed: {len(df)}")
    print(f"Problematic profiles identified: {df['is_problematic'].sum()}")
    print("\nThresholds used:")
    print(f"Correlation: < 0.95")
    print(f"DTW distance: > {dtw_threshold:.3f}")
    print(f"Shape RMSE: > {shape_threshold:.3f}")
    print(f"Mean difference: > 1m")
    print(f"Standard deviation: > 2m")
    print(f"Slope deviation: > 0.1 from 1:1")
    
    # Sort by various metrics and save detailed results
    results_file = os.path.join(output_dir, 'profile_analysis_results.csv')
    df.sort_values('correlation', ascending=True).to_csv(results_file, index=False)
    print(f"\nDetailed results saved to: {results_file}")
    
    return df

def plot_problematic_profiles(filtered_significant, analysis_results):
    """
    Plot profiles identified as problematic by the multiple metrics approach
    """
    problematic_ids = analysis_results[analysis_results['is_problematic']]['system_id'].values
    
    print(f"\nPlotting {len(problematic_ids)} problematic profiles...")
    
    # Use the existing plotting function with the problematic IDs
    plot_all_odr_scatter(filtered_significant[
        filtered_significant['system_id'].isin(problematic_ids)
    ])

def plot_all_odr_scatter(filtered_significant):
    """
    Create scatter plots with ODR analysis for all significant profiles in a 7x8 grid
    """
    def linear_func(p, x):
        """Linear function y = mx + b"""
        m, b = p
        return m*x + b
    
    # Get all system IDs from significant profiles
    system_ids = filtered_significant['system_id'].unique()
    n_plots = len(system_ids)
    plots_per_fig = 56  # 7 rows * 8 columns
    n_figs = (n_plots + plots_per_fig - 1) // plots_per_fig  # Ceiling division
    
    for fig_num in range(n_figs):
        # Create figure with 7x8 subplots
        fig, axes = plt.subplots(7, 8, figsize=(32, 28))
        fig.suptitle(f'ODR Analysis Scatter Plots - All Significant Profiles (Page {fig_num + 1})', fontsize=16)
        
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
                
                # Calculate axis limits for this profile
                max_elev = max(merged['elevation_nld'].max(), merged['elevation_tep'].max())
                min_elev = min(merged['elevation_nld'].min(), merged['elevation_tep'].min())
                
                # Add padding (5%)
                range_elev = max_elev - min_elev
                padding = range_elev * 0.05
                plot_max = max_elev + padding
                plot_min = min_elev - padding
                
                # Round to nice numbers
                plot_max = math.ceil(plot_max)
                plot_min = math.floor(plot_min)
                
                # Perform ODR
                linear = odr.Model(linear_func)
                data = odr.RealData(merged['elevation_nld'], merged['elevation_tep'])
                odr_obj = odr.ODR(data, linear, beta0=[1., 0.])
                results = odr_obj.run()
                
                slope = results.beta[0]
                intercept = results.beta[1]
                
                # Calculate R-squared
                y_pred = linear_func(results.beta, merged['elevation_nld'])
                r_squared = np.corrcoef(merged['elevation_tep'], y_pred)[0,1]**2
                
                # Create scatter plot
                ax.scatter(merged['elevation_nld'], merged['elevation_tep'], 
                          alpha=0.5, c='blue', s=10)
                
                # Add 1:1 line
                ax.plot([plot_min, plot_max], [plot_min, plot_max], 
                       'r--', linewidth=1, label='1:1 Line')
                
                # Add regression line
                x_range = np.array([plot_min, plot_max])
                y_range = slope * x_range + intercept
                ax.plot(x_range, y_range, 'g-', linewidth=1, 
                       label=f'Slope={slope:.2f}')
                
                # Format plot
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Set limits for this plot
                ax.set_xlim(plot_min, plot_max)
                ax.set_ylim(plot_min, plot_max)
                
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
                
                # Add title with system ID, R², slope and elevation range
                ax.set_title(f'ID: {system_id}\nR²: {r_squared:.3f}, Slope: {slope:.2f}\nRange: {plot_min:.0f}-{plot_max:.0f}m', 
                            fontsize=8)
        
        # Turn off any unused subplots
        for idx in range(len(current_ids), plots_per_fig):
            axes_flat[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'all_significant_odr_plots_page{fig_num+1}.png'))
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
    
    # After creating filtered_significant
    print("\nPlotting scatter plots for all significant profiles...")
    plot_all_significant_scatter(filtered_significant)
    
    print("\nPerforming multiple metric analysis...")
    analysis_results = analyze_profile_multiple_metrics(filtered_significant)
    
    print("\nPlotting problematic profiles...")
    plot_problematic_profiles(filtered_significant, analysis_results)

