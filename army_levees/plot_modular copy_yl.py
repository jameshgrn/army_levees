import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeatures
from cartopy.io.img_tiles import GoogleTiles
import matplotlib.colors as colors
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import zscore

# Directory where the elevation data files are saved
data_dir = '/Users/liyuan/projects/army_levees/data'
output_dir = '/Users/liyuan/projects/army_levees/plots'

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
            merged = merged[(merged['zscore_nld'].abs() <= 3) & (merged['zscore_tep'].abs() <= 3)]
            
            if len(merged) > 0:
                # Calculate the difference only for valid points
                elevation_diff = merged['elevation_tep'] - merged['elevation_nld']  # TEP - NLD
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
    
    diff_df = pd.DataFrame(differences)
    
    # Filter out profiles with more than 15m of mean difference
    filtered_df = diff_df[diff_df['mean_elevation_diff'].abs() <= 15]
    
    print(f"Filtered out {len(diff_df) - len(filtered_df)} profiles with >15m mean difference")
    print(f"Total points compared: {filtered_df['points_compared'].sum()}")
    
    # Merge differences back into the original DataFrame
    plot_df = plot_df.merge(filtered_df, on='system_id', how='left')
    
    return plot_df, filtered_df

def analyze_differences(mean_differences):
    positive_diff = mean_differences[mean_differences['mean_elevation_diff'] > 0]
    negative_diff = mean_differences[mean_differences['mean_elevation_diff'] < 0]
    
    print(f"Number of positive differences: {len(positive_diff)}")
    print(f"Number of negative differences: {len(negative_diff)}")
    
    print(f"Mean of positive differences: {positive_diff['mean_elevation_diff'].mean()}")
    print(f"Mean of negative differences: {negative_diff['mean_elevation_diff'].mean()}")
    
    print(f"Standard deviation of positive differences: {positive_diff['mean_elevation_diff'].std()}")
    print(f"Standard deviation of negative differences: {negative_diff['mean_elevation_diff'].std()}")

def plot_mean_elevation_diff_map(mean_differences):
    # Load the original data to get geometries
    gdf = load_and_process_data()
    
    # Merge geometries with mean differences
    mean_diff_gdf = gdf[['system_id', 'geometry']].drop_duplicates().merge(mean_differences, on='system_id')

    terrain_background = GoogleTiles(style='satellite')
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(15, 5))
    
    ax.add_image(terrain_background, 7)
    ax.add_feature(cfeatures.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '50m', 
                                                 facecolor='none', edgecolor='w', linestyle=':', lw=.5))
    ax.set_extent([-125, -66.5, 20, 50], crs=ccrs.PlateCarree())
    
    norm = colors.TwoSlopeNorm(vmin=mean_diff_gdf['mean_elevation_diff'].min(), 
                               vcenter=0, 
                               vmax=mean_diff_gdf['mean_elevation_diff'].max())
    scatter = mean_diff_gdf.plot(ax=ax, column='mean_elevation_diff', cmap='RdYlBu', 
                                 alpha=0.7, norm=norm, markersize=30, edgecolor='k',
                                 legend=False)  # Set legend to False here
    
    sm = plt.cm.ScalarMappable(cmap='RdYlBu', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', label='Mean Elevation Difference (m)')
    
    # Set font size to 5-7 pt
    plt.rcParams.update({'font.size': 5})
    ax.tick_params(labelsize=5)
    cbar.ax.tick_params(labelsize=5)
    cbar.ax.yaxis.label.set_size(5)
    
    
    output_path = os.path.join(output_dir, 'mean_elevation_diff_map.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved mean elevation difference map to {output_path}")
    plt.close()

def plot_location_map():
    # Load the original data to get geometries
    gdf = load_and_process_data()
    
    terrain_background = GoogleTiles(style='satellite')
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(15, 5))
    
    ax.add_image(terrain_background, 7)
    ax.add_feature(cfeatures.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '50m', 
                                                 facecolor='none', edgecolor='w', linestyle=':', lw=.5))
    ax.set_extent([-125, -66.5, 20, 50], crs=ccrs.PlateCarree())
    
    gdf.plot(ax=ax, color='black', markersize=30, edgecolor='k', alpha=0.7)
    
    # Set font size to 5-7 pt
    plt.rcParams.update({'font.size': 5})
    ax.tick_params(labelsize=5)
    
    output_path = os.path.join(output_dir, 'location_map.png')
    plt.savefig(output_path, dpi=150)
    print(f"Saved location map to {output_path}")
    plt.close()

def plot_cdf_differences(mean_differences):
    fig, ax = plt.subplots(figsize=(3, 2))  # Reduced width
    sorted_diff = np.sort(mean_differences['mean_elevation_diff'])
    yvals = np.arange(len(sorted_diff)) / float(len(sorted_diff) - 1)
    ax.plot(sorted_diff, yvals)
    ax.set_xlabel('Mean Elevation Difference (m)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution Function of Mean Elevation Differences')
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Set font size to 5-7 pt
    plt.rcParams.update({'font.size': 5})
    ax.tick_params(labelsize=5)
    ax.xaxis.label.set_size(5)
    ax.yaxis.label.set_size(5)
    ax.title.set_size(7)
    
    # Add n value as text
    n_value = len(mean_differences)
    ax.text(0.95, 0.05, f'n = {n_value}', transform=ax.transAxes, verticalalignment='bottom',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    output_path = os.path.join(output_dir, 'cdf_differences.pdf')
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved CDF of differences plot to {output_path}")
    plt.close()

def plot_mean_elevation_diff_histogram(mean_differences):
    fig, ax = plt.subplots(figsize=(3, 3))  # Reduced size
    sns.histplot(mean_differences['mean_elevation_diff'], kde=True, ax=ax)
    ax.set_title('Distribution of Mean Elevation Differences')
    ax.set_xlabel('Mean Elevation Difference (m)')
    ax.set_ylabel('Frequency')
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.7)
    
    stats_text = f"Mean: {mean_differences['mean_elevation_diff'].mean():.2f}m\n"
    stats_text += f"Median: {mean_differences['mean_elevation_diff'].median():.2f}m\n"
    stats_text += f"Std Dev: {mean_differences['mean_elevation_diff'].std():.2f}m\n"
    stats_text += f"Skewness: {mean_differences['mean_elevation_diff'].skew():.2f}\n"
    stats_text += f"Kurtosis: {mean_differences['mean_elevation_diff'].kurtosis():.2f}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Set font size to 5-7 pt
    plt.rcParams.update({'font.size': 5})
    ax.tick_params(labelsize=5)
    ax.xaxis.label.set_size(5)
    ax.yaxis.label.set_size(5)
    ax.title.set_size(7)
    
    # Add n value as text
    n_value = len(mean_differences)
    ax.text(0.95, 0.05, f'n = {n_value}', transform=ax.transAxes, verticalalignment='bottom',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    output_path = os.path.join(output_dir, 'mean_elevation_diff_histogram.pdf')
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved mean elevation difference histogram to {output_path}")
    plt.close()

def plot_boxplot(mean_differences):
    # Further filter for differences >= 0.1m
    significant_diff = mean_differences[mean_differences['mean_elevation_diff'].abs() >= 0.1]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Split data into negative and positive differences
    negative_diff = significant_diff[significant_diff['mean_elevation_diff'] < 0]['mean_elevation_diff']
    positive_diff = significant_diff[significant_diff['mean_elevation_diff'] > 0]['mean_elevation_diff']
    
    # Create a list of data and labels for the boxplot
    data = [negative_diff, positive_diff]
    labels = ['Negative Differences', 'Positive Differences']
    
    # Create boxplot
    bp = ax.boxplot(data, labels=labels, showfliers=True, patch_artist=True)
    
    # Color the boxes
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightgreen')
    
    ax.set_title('Boxplot of Mean Elevation Differences (≥0.1m)')
    ax.set_ylabel('Mean Elevation Difference (m)')
    
    # Add text annotations with statistics
    for i, d in enumerate(data):
        stats_text = f"Mean: {d.mean():.2f}m\n"
        stats_text += f"Median: {d.median():.2f}m\n"
        stats_text += f"Std Dev: {d.std():.2f}m\n"
        stats_text += f"Count: {len(d)}"
        ax.text(i+1, ax.get_ylim()[1]-.3, stats_text, 
                horizontalalignment='center', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Set font size to 5-7 pt
    plt.rcParams.update({'font.size': 5})
    ax.tick_params(labelsize=5)
    ax.xaxis.label.set_size(5)
    ax.yaxis.label.set_size(5)
    ax.title.set_size(7)
    
    # Add n value as text
    n_value = len(significant_diff)
    ax.text(0.95, 0.05, f'n = {n_value}', transform=ax.transAxes, verticalalignment='bottom',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    output_path = os.path.join(output_dir, 'mean_elevation_diff_boxplot.pdf')
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved mean elevation difference boxplot to {output_path}")
    plt.close()

def plot_elevation_diff_statistics(elevation_statistics):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Subplot 1: Mean vs Standard Deviation
    ax1.scatter(elevation_statistics['mean_elevation_diff'], 
                elevation_statistics['std_elevation_diff'], 
                color='darkblue', s=20, alpha=0.6)
    ax1.set_xlabel('Mean Elevation Difference (m)')
    ax1.set_ylabel('Standard Deviation of\nElevation Difference (m)')
    ax1.set_title('Mean vs Standard Deviation')
    
    # Subplot 2: Maximum vs Range
    ax2.scatter(elevation_statistics['max_elevation_diff'], 
                elevation_statistics['range_elevation_diff'], 
                color='darkblue', s=20, alpha=0.6)
    ax2.set_xlabel('Maximum Elevation Difference (m)')
    ax2.set_ylabel('Range of Elevation Difference (m)')
    ax2.set_title('Maximum vs Range')
    
    # Subplot 3: 25th Percentile vs 75th Percentile
    ax3.scatter(elevation_statistics['percentile_25_elevation_diff'], 
                elevation_statistics['percentile_75_elevation_diff'], 
                color='darkblue', s=20, alpha=0.6)
    ax3.set_xlabel('25th Percentile of Elevation Difference (m)')
    ax3.set_ylabel('75th Percentile of\nElevation Difference (m)')
    ax3.set_title('25th vs 75th Percentile')
    
    # Adjust layout
    plt.tight_layout(w_pad=4)
    
    # Set font size to larger values
    plt.rcParams.update({'font.size': 12})
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(labelsize=10)
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)
        ax.title.set_size(14)
    
    # Adjust y-axis labels for better readability
    ax1.yaxis.labelpad = 10
    ax2.yaxis.labelpad = 10
    ax3.yaxis.labelpad = 10
    
    output_path = os.path.join(output_dir, 'stats_elevation_diff_scatters.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved elevation difference statistics scatter plots to {output_path}")
    plt.close()

def plot_profiles(plot_df, elevation_statistics):
    # Filter for significant differences (>= 0.1m) and non-significant differences (< 0.1m)
    significant_diff = elevation_statistics[elevation_statistics['mean_elevation_diff'].abs() >= 0.1]
    non_significant_diff = elevation_statistics[elevation_statistics['mean_elevation_diff'].abs() < 0.1]
    
    def create_profile_plot(diff_df, output_filename, title_prefix):
        # Create a 5x10 subplot grid
        fig, axes = plt.subplots(5, 10, figsize=(30, 15))
        axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing
        
        for i, system_id in enumerate(diff_df['system_id'].tolist()[:50]):  # Limit to 50 profiles (5x10 grid)
            ax = axes[i]
            
            # Filter data for this system_id
            system_data = plot_df[plot_df['system_id'] == system_id]
            
            # Apply the same filtering as in calculate_mean_differences
            nld_data = system_data[(system_data['source'] == 'nld') & (system_data['elevation'] != 0)]
            tep_data = system_data[system_data['source'] == 'tep']
            
            # Merge the two dataframes on 'system_id' and 'distance_along_track' to align the measurements
            merged = pd.merge(nld_data, tep_data, on=['system_id', 'distance_along_track'], suffixes=('_nld', '_tep'))
            
            # Remove rows where either elevation is NaN, None, or empty string
            merged = merged.dropna(subset=['elevation_nld', 'elevation_tep'])
            merged = merged[(merged['elevation_nld'] != '') & (merged['elevation_tep'] != '')]
            
            # Ensure elevations are numeric
            merged['elevation_nld'] = pd.to_numeric(merged['elevation_nld'], errors='coerce')
            merged['elevation_tep'] = pd.to_numeric(merged['elevation_tep'], errors='coerce')
            
            # Filter out rows with NaN values in elevation columns
            merged = merged.dropna(subset=['elevation_nld', 'elevation_tep'])
            
            # Calculate z-scores for elevation columns
            merged['zscore_nld'] = stats.zscore(merged['elevation_nld'])
            merged['zscore_tep'] = stats.zscore(merged['elevation_tep'])
            
            # Remove rows with z-scores beyond a threshold (e.g., |z| > 3)
            merged = merged[(merged['zscore_nld'].abs() <= 3) & (merged['zscore_tep'].abs() <= 3)]
            
            if len(merged) > 0:
                # Sort by distance_along_track to ensure correct line plotting
                merged = merged.sort_values('distance_along_track')
                
                # Plot NLD profile (blue solid line with blue circles)
                ax.plot(merged['distance_along_track'], merged['elevation_nld'], color='blue', linestyle='-', linewidth=1, label='NLD')
                ax.scatter(merged['distance_along_track'], merged['elevation_nld'], color='blue', s=10, marker='o')
                
                # Plot TEP profile (red dashed line with red circles)
                ax.plot(merged['distance_along_track'], merged['elevation_tep'], color='red', linestyle='--', linewidth=1, label='TEP')
                ax.scatter(merged['distance_along_track'], merged['elevation_tep'], color='red', s=10, marker='o')
                
                ax.set_title(f'System ID: {system_id}', fontsize=8)
                ax.tick_params(labelsize=6)
                ax.set_xlabel('Distance (m)', fontsize=6)
                ax.set_ylabel('Elevation (m)', fontsize=6)
                
                # Only add legend to the first subplot
                if i == 0:
                    ax.legend(fontsize=6)
        
        # Remove any unused subplots
        for j in range(i+1, 50):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved {title_prefix} elevation profiles to {output_path}")
        plt.close()

    # Create plots for significant and non-significant differences
    create_profile_plot(significant_diff, 'elevation_filtered_profiles_sig.png', 'Significant')
    create_profile_plot(non_significant_diff, 'elevation_filtered_profiles_nonsig.png', 'Non-significant')

if __name__ == "__main__":
    plot_df = load_and_process_data()
    print(f"Combined DataFrame shape: {plot_df.shape}")
    print(f"Combined DataFrame columns: {plot_df.columns}")
    
    if plot_df.empty:
        print("WARNING: No data loaded. Check your data directory and file formats.")
        exit()
    
    plot_df, elevation_statistics = calculate_mean_differences(plot_df)
    
    # Sort and print system_id and statistics by difference magnitude
    sorted_statistics = elevation_statistics.sort_values(by='mean_elevation_diff', key=abs)
    print("System ID and Elevation Difference Statistics (sorted by magnitude of mean difference):")
    print(sorted_statistics[['system_id', 'mean_elevation_diff', 'mean_diff_significant', 'std_elevation_diff', 'max_elevation_diff', 'range_elevation_diff', 'percentile_25_elevation_diff', 'percentile_75_elevation_diff']])
    
    # Print the entire DataFrame
    print("Entire DataFrame of Elevation Difference Statistics:")
    print(elevation_statistics)
    
    # Save elevation_difference_statistics.csv to the 'data' folder
    statistics_path = os.path.join(data_dir, 'elevation_difference_statistics.csv')
    elevation_statistics.to_csv(statistics_path, index=False)
    print(f"Saved elevation difference statistics to {statistics_path}")
    
    # Analyze differences
    analyze_differences(elevation_statistics)
    
    print("Plotting mean elevation difference map...")
    plot_mean_elevation_diff_map(elevation_statistics)
    
    print("Plotting location map...")
    plot_location_map()
    
    print("Plotting mean elevation difference histogram...")
    plot_mean_elevation_diff_histogram(elevation_statistics)
    
    print("Plotting CDF of differences...")
    plot_cdf_differences(elevation_statistics)
    
    print("Plotting mean elevation difference boxplot...")
    plot_boxplot(elevation_statistics)
    
    print("Plotting elevation difference statistics scatter plots...")
    plot_elevation_diff_statistics(elevation_statistics)
    
    print("Plotting filtered elevation profiles...")
    plot_profiles(plot_df, elevation_statistics)
    
    print("All plotting and analysis operations completed.")