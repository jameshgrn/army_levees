'''
NLD and TEP Elevation Profile Comparison

Created by Yuan Li (yl246@iu.edu) on Sep 25, 2024

This script analyzes and visualizes the differences between National Levee Database (NLD) 
and 3DEP (TEP) elevation data for multiple levee profiles.

What this script does:

## Data Processing Steps

1. Load NLD and 3DEP data from parquet files
2. Filter out zero and NaN values
3. Merge NLD and 3DEP data
4. Calculate elevation differences (3DEP - NLD)
5. Remove extreme differences (|diff| > 1000)
6. Apply z-score filtering to remove outliers
7. Sort sites by mean elevation difference
8. Select top 30 sites with largest mean differences

## Visualization

- **Elevation Profiles:**
  - Blue line: NLD elevation
  - Red line: 3DEP elevation
  - 'elevation_filtered_profiles.png': 5x6 grid of elevation profile plots
 
- **Elevation Difference Distributions:**
  - Histogram of elevation differences
  - Kernel Density Estimation (KDE) plot
  - Mean and standard deviation of differences
  - 'elevation_filtered_difference_distributions.png': 5x6 grid of elevation difference distribution plots



'''

import os
import pandas as pd
import geopandas as gpd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from scipy.stats import zscore
import numpy as np
from scipy import stats

data_dir = 'data'  # Directory where the elevation data files are saved
plots_dir = 'plots'  # Directory where the plots will be saved

# Make sure the plots directory exists
os.makedirs(plots_dir, exist_ok=True)

def process_and_filter_data(path_nld, path_3dep, zero_threshold=0.5, min_non_zero_values=10):
    df_nld = gpd.read_parquet(path_nld)
    df_3dep = gpd.read_parquet(path_3dep)

    # Filter out rows where NLD elevation values are zero
    df_nld = df_nld[df_nld['elevation'] != 0]

    # Check if there are enough non-zero values
    if len(df_nld) < min_non_zero_values:
        return None

    # Merge dataframes
    merged_df = pd.merge(df_nld, df_3dep, on='distance_along_track', suffixes=('_nld', '_3dep'))

    # Filter out rows with NaN values in elevation columns
    merged_df = merged_df.dropna(subset=['elevation_nld', 'elevation_3dep'])

    # Calculate elevation differences
    merged_df['elevation_diff'] = merged_df['elevation_3dep'] - merged_df['elevation_nld']

    # Filter out rows with extreme elevation differences (e.g., |diff| > 1000)
    merged_df = merged_df[merged_df['elevation_diff'].abs() <= 1000]

    # Calculate z-scores and filter
    for col in ['elevation_nld', 'elevation_3dep', 'elevation_diff']:
        merged_df[f'zscore_{col}'] = zscore(merged_df[col])
        merged_df = merged_df[merged_df[f'zscore_{col}'].abs() <= 3]

    # Check if the merged dataframe is empty after filtering
    if merged_df.empty:
        return None

    return merged_df

def plot_profiles_and_distributions(data_list):
    fig_profiles, axs_profiles = plt.subplots(5, 6, figsize=(30, 25))
    fig_distributions, axs_distributions = plt.subplots(5, 6, figsize=(30, 25))

    for idx, (system_id, merged_df) in enumerate(data_list):
        row = idx // 6
        col = idx % 6

        # Plot profiles
        ax_profile = axs_profiles[row, col]
        ax_profile.plot(merged_df['distance_along_track'], merged_df['elevation_nld'], label='NLD', color='blue', linewidth=1)
        ax_profile.plot(merged_df['distance_along_track'], merged_df['elevation_3dep'], label='3DEP', color='red', linewidth=1)
        ax_profile.set_title(f'System ID: {system_id}')
        ax_profile.set_xlabel('Distance Along Track (m)')
        ax_profile.set_ylabel('Elevation (m)')
        ax_profile.legend()

        # Plot distributions
        ax_distribution = axs_distributions[row, col]
        
        # Histogram
        ax_distribution.hist(merged_df['elevation_diff'], bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Kernel Density Estimation
        kde = stats.gaussian_kde(merged_df['elevation_diff'])
        x_range = np.linspace(merged_df['elevation_diff'].min(), merged_df['elevation_diff'].max(), 100)
        ax_distribution.plot(x_range, kde(x_range), 'r-', linewidth=2)
        
        # Calculate and display mean and standard deviation
        mean_diff = merged_df['elevation_diff'].mean()
        std_diff = merged_df['elevation_diff'].std()
        ax_distribution.text(0.05, 0.95, f'Mean: {mean_diff:.2f}\nStd: {std_diff:.2f}', 
                             transform=ax_distribution.transAxes, verticalalignment='top')
        
        ax_distribution.set_title(f'System ID: {system_id}')
        ax_distribution.set_xlabel('Elevation Difference (m)')
        ax_distribution.set_ylabel('Density')

    # Adjust layout and save figures
    fig_profiles.tight_layout()
    fig_profiles.savefig(os.path.join(plots_dir, 'elevation_filtered_profiles.png'), dpi=300)
    plt.close(fig_profiles)

    fig_distributions.tight_layout()
    fig_distributions.savefig(os.path.join(plots_dir, 'elevation_filtered_difference_distributions.png'), dpi=300)
    plt.close(fig_distributions)

def process_all_files(data_dir):
    files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
    nld_files = [f for f in files if 'nld' in f]
    tep_files = [f for f in files if 'tep' in f]

    processed_data = []
    for nld_file in nld_files:
        system_id = nld_file.split('_')[2]
        corresponding_tep_file = next((f for f in tep_files if system_id in f), None)
        if corresponding_tep_file:
            merged_df = process_and_filter_data(os.path.join(data_dir, nld_file), os.path.join(data_dir, corresponding_tep_file))
            if merged_df is not None:
                processed_data.append((system_id, merged_df))

    # Sort by mean elevation difference in descending order
    processed_data.sort(key=lambda x: x[1]['elevation_diff'].mean(), reverse=True)

    # Take the top 30 sites
    top_30_data = processed_data[:30]

    plot_profiles_and_distributions(top_30_data)

if __name__ == "__main__":
    process_all_files(data_dir)