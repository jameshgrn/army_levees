"""
NLD and TEP Elevation Profile Comparison

Created by Yuan Li (yl246@iu.edu) on Sep 19, 2024

This script analyzes and visualizes the differences between National Levee Database (NLD) 
and Topographically Extracted Profile (TEP) elevation data for multiple levee profiles.

What this script does:

1. Loads paired NLD and TEP elevation profile data from parquet files in a specified 'data' folder.

2. Calculates the differences between NLD and TEP elevations for each profile pair.

3. Creates two types of visualizations:

   a. Elevation Difference Distributions:
      - Generates a figure with multiple subplots (5 rows x 6 columns by default).
      - Each subplot shows a histogram and kernel density estimation of elevation differences 
        for a single profile pair.
      - Includes mean and standard deviation of differences on each subplot.
      - Covers the subplot with a transparent gray polygon if the mean difference is negative.
      - Covers the subplot with a transparent magenta polygon if the Kurtosis value is negative.
      - Covers the subplot with a transparent green polygon if the Kurtosis value is positive.

   b. Original Elevation Profiles:
      - Generates another figure with multiple subplots (5 rows x 6 columns by default).
      - Each subplot displays the original NLD (blue solid line) and TEP (red dashed line) 
        elevation profiles for a single pair.
      - Covers the subplot with a transparent gray polygon if the mean difference is negative.
      - Covers the subplot with a transparent magenta polygon if the Kurtosis value is negative.
      - Covers the subplot with a transparent green polygon if the Kurtosis value is positive.

4. Saves both figures as high-resolution PNG files in a 'plots' folder.

5. Provides a summary of how many plots of each type were generated.

This script is useful for visually comparing NLD and TEP elevation data, identifying 
discrepancies between the two datasets, and analyzing the distribution of these differences 
across multiple levee profiles.

Usage:
Ensure that paired NLD and TEP parquet files are in the 'data' folder, then run the script. 
The resulting plots will be saved in the 'plots' folder.

Note: This script assumes that the NLD and TEP files follow a specific naming convention 
and contain 'distance_along_track' and 'elevation' columns.
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.stats import kurtosis
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as patches
from sklearn.covariance import EllipticEnvelope

def load_and_process_data(path_nld, path_tep):
    df_nld = gpd.read_parquet(path_nld)
    df_tep = gpd.read_parquet(path_tep)
    
    merged_df = pd.merge(df_nld, df_tep, on='distance_along_track', suffixes=('_nld', '_tep'))
    merged_df['elevation_diff'] = merged_df['elevation_nld'] - merged_df['elevation_tep']
    
    return merged_df

def calculate_statistics(merged_df):
    diff = merged_df['elevation_diff']
    return {
        'mean': diff.mean(),
        'median': diff.median(),
        'std': diff.std(),
        'rmse': np.sqrt(np.mean(diff**2)),
        'mse': np.mean(diff**2),
        'kurtosis': kurtosis(diff)
    }

def plot_difference_distributions(data_folder, num_rows=5, num_cols=6):
    # Get all NLD files
    nld_files = [f for f in os.listdir(data_folder) if '_nld_' in f and f.endswith('.parquet')]
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 16))
    fig.suptitle('Distribution of Elevation Differences (NLD - TEP)', fontsize=16)
    
    processed_pairs = 0
    
    for idx, nld_file in enumerate(nld_files):
        # Extract the ID and EPSG code from the NLD filename
        id_epsg_match = re.search(r'elevation_data_(\d+)_nld_(epsg\d+)', nld_file)
        if id_epsg_match:
            id_number, epsg_code = id_epsg_match.groups()
            
            # Construct the corresponding TEP file name pattern
            tep_file = f'elevation_data_{id_number}_tep_{epsg_code}.parquet'
            
            if tep_file in os.listdir(data_folder):
                nld_path = os.path.join(data_folder, nld_file)
                tep_path = os.path.join(data_folder, tep_file)
                
                merged_df = load_and_process_data(nld_path, tep_path)
                
                row = processed_pairs // num_cols
                col = processed_pairs % num_cols
                ax = axs[row, col]
                
                # Plot histogram and kernel density estimation
                sns.histplot(merged_df['elevation_diff'], kde=True, ax=ax)
                ax.set_title(f'ID: {id_number}')
                ax.set_xlabel('Elevation Difference (m)')
                ax.set_ylabel('Frequency')
                
                # Calculate statistics
                mean = merged_df['elevation_diff'].mean()
                std = merged_df['elevation_diff'].std()
                kurt = kurtosis(merged_df['elevation_diff'])
                
                # Add Mean and Std statistics as text
                ax.text(0.05, 0.95, f'Mean: {mean:.2f}\nStd: {std:.2f}', 
                        transform=ax.transAxes, verticalalignment='top')
                
                # Add Kurtosis statistic as red text
                ax.text(0.05, 0.85, f'Kurtosis: {kurt:.2f}', 
                        transform=ax.transAxes, verticalalignment='top', color='red')
                
                # Cover subplot with a transparent gray polygon if mean difference is negative
                if mean < 0:
                    ax.fill_betweenx([0, 1], 0, 1, color='gray', alpha=0.3, transform=ax.transAxes)
                else:
                    # Cover subplot with a transparent magenta polygon if Kurtosis is negative
                    if kurt < 0:
                        ax.fill_betweenx([0, 1], 0, 1, color='magenta', alpha=0.3, transform=ax.transAxes)
                    # Cover subplot with a transparent green polygon if Kurtosis is positive
                    else:
                        ax.fill_betweenx([0, 1], 0, 1, color='green', alpha=0.3, transform=ax.transAxes)
                
                processed_pairs += 1
                if processed_pairs >= num_rows * num_cols:
                    break
            else:
                print(f"Warning: No matching TEP file found for {nld_file}")
        else:
            print(f"Warning: Unable to extract ID and EPSG code from {nld_file}")
    
    # Remove any unused subplots
    for i in range(processed_pairs, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        fig.delaxes(axs[row, col])
    
    plt.tight_layout()
    
    # Ensure the plots folder exists
    os.makedirs('plots', exist_ok=True)
    
    # Save the figure in the plots folder
    plt.savefig('plots/elevation_difference_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return processed_pairs

def plot_elevation_profiles(data_folder, num_rows=5, num_cols=6):
    # Get all NLD files
    nld_files = [f for f in os.listdir(data_folder) if '_nld_' in f and f.endswith('.parquet')]
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 16))
    fig.suptitle('NLD and TEP Elevation Profiles', fontsize=16)
    
    processed_pairs = 0
    
    for idx, nld_file in enumerate(nld_files):
        # Extract the ID and EPSG code from the NLD filename
        id_epsg_match = re.search(r'elevation_data_(\d+)_nld_(epsg\d+)', nld_file)
        if id_epsg_match:
            id_number, epsg_code = id_epsg_match.groups()
            
            # Construct the corresponding TEP file name pattern
            tep_file = f'elevation_data_{id_number}_tep_{epsg_code}.parquet'
            
            if tep_file in os.listdir(data_folder):
                nld_path = os.path.join(data_folder, nld_file)
                tep_path = os.path.join(data_folder, tep_file)
                
                merged_df = load_and_process_data(nld_path, tep_path)
                
                row = processed_pairs // num_cols
                col = processed_pairs % num_cols
                ax = axs[row, col]
                
                # Plot NLD and TEP elevation profiles
                ax.plot(merged_df['distance_along_track'], merged_df['elevation_nld'], 
                        color='blue', linestyle='-', label='NLD')
                ax.plot(merged_df['distance_along_track'], merged_df['elevation_tep'], 
                        color='red', linestyle='--', label='TEP')
                
                ax.set_title(f'ID: {id_number}')
                ax.set_xlabel('Distance Along Track (m)')
                ax.set_ylabel('Elevation (m)')
                ax.legend()
                
                # Calculate mean difference and kurtosis
                mean = merged_df['elevation_diff'].mean()
                kurt = kurtosis(merged_df['elevation_diff'])
                
                # Cover subplot with a transparent gray polygon if mean difference is negative
                if mean < 0:
                    ax.fill_betweenx([0, 1], 0, 1, color='gray', alpha=0.3, transform=ax.transAxes)
                else:
                    # Cover subplot with a transparent magenta polygon if Kurtosis is negative
                    if kurt < 0:
                        ax.fill_betweenx([0, 1], 0, 1, color='magenta', alpha=0.3, transform=ax.transAxes)
                    # Cover subplot with a transparent green polygon if Kurtosis is positive
                    else:
                        ax.fill_betweenx([0, 1], 0, 1, color='green', alpha=0.3, transform=ax.transAxes)
                
                processed_pairs += 1
                if processed_pairs >= num_rows * num_cols:
                    break
            else:
                print(f"Warning: No matching TEP file found for {nld_file}")
        else:
            print(f"Warning: Unable to extract ID and EPSG code from {nld_file}")
    
    # Remove any unused subplots
    for i in range(processed_pairs, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        fig.delaxes(axs[row, col])
    
    plt.tight_layout()
    
    # Ensure the plots folder exists
    os.makedirs('plots', exist_ok=True)
    
    # Save the figure in the plots folder
    plt.savefig('plots/elevation_profiles.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return processed_pairs

def perform_kmeans_clustering(data_folder, n_clusters=3):
    nld_files = [f for f in os.listdir(data_folder) if '_nld_' in f and f.endswith('.parquet')]
    statistics = []

    for nld_file in nld_files:
        id_epsg_match = re.search(r'elevation_data_(\d+)_nld_(epsg\d+)', nld_file)
        if id_epsg_match:
            id_number, epsg_code = id_epsg_match.groups()
            tep_file = f'elevation_data_{id_number}_tep_{epsg_code}.parquet'
            
            if tep_file in os.listdir(data_folder):
                nld_path = os.path.join(data_folder, nld_file)
                tep_path = os.path.join(data_folder, tep_file)
                merged_df = load_and_process_data(nld_path, tep_path)
                stats = calculate_statistics(merged_df)
                statistics.append(stats)

    df_stats = pd.DataFrame(statistics)
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_stats), columns=df_stats.columns)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_scaled['cluster'] = kmeans.fit_predict(df_scaled)

    fig, axs = plt.subplots(6, 5, figsize=(25, 30))
    fig.suptitle('K-means Clustering of Elevation Difference Statistics', fontsize=16)

    all_params = ['mean', 'median', 'std', 'rmse', 'mse', 'kurtosis']
    row_params = ['mean', 'median', 'std', 'rmse', 'mse', 'kurtosis']

    for row, row_param in enumerate(row_params):
        other_params = [p for p in all_params if p != row_param]
        for col, col_param in enumerate(other_params):
            ax = axs[row, col]
            scatter = ax.scatter(df_stats[row_param], df_stats[col_param], c=df_scaled['cluster'], cmap='viridis', alpha=0.7)
            ax.set_xlabel(row_param.upper())
            ax.set_ylabel(col_param.upper())
            ax.set_title(f'{row_param.upper()} vs {col_param.upper()}')

            # Add cluster circles
            for cluster in range(n_clusters):
                cluster_points = df_stats[df_scaled['cluster'] == cluster]
                
                if len(cluster_points) >= 2:  # Check if there are at least 2 points in the cluster
                    cluster_center = cluster_points[[row_param, col_param]].mean()
                    
                    try:
                        # Use EllipticEnvelope to estimate the shape of the cluster
                        ee = EllipticEnvelope(random_state=42, support_fraction=1.0)
                        ee.fit(cluster_points[[row_param, col_param]])
                        
                        # Get the axes of the ellipse
                        _, s, _ = np.linalg.svd(ee.covariance_)
                        width, height = 2 * np.sqrt(s)
                        
                        # Create and add the ellipse patch
                        ellipse = patches.Ellipse(cluster_center, width, height,
                                                  fill=False, edgecolor='k', linestyle='--')
                        ax.add_patch(ellipse)
                    except ValueError as e:
                        print(f"Skipping ellipse for cluster {cluster} in subplot {row},{col}: {str(e)}")
                else:
                    print(f"Not enough points in cluster {cluster} for subplot {row},{col}. Skipping ellipse.")

    # Adjust layout to make room for the colorbar
    plt.tight_layout(rect=[0, 0, 0.95, 0.98])
    
    # Add colorbar to the right of the subplots
    cbar_ax = fig.add_axes([0.96, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('Cluster Assignment', rotation=270, labelpad=20)
    
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/elevation_difference_kmean.png', dpi=300, bbox_inches='tight')
    plt.close()

    return df_stats, df_scaled['cluster']

# Run the processing for all files in the data folder
total_diff_plots = plot_difference_distributions('data')
total_profile_plots = plot_elevation_profiles('data')
stats, clusters = perform_kmeans_clustering('data')

print(f"\nScript execution completed.")
print(f"{total_diff_plots} difference distribution plots have been generated and saved in the 'plots' folder.")
print(f"{total_profile_plots} elevation profile plots have been generated and saved in the 'plots' folder.")
print("K-means clustering plot has been generated and saved as 'elevation_difference_kmean.png' in the 'plots' folder.")