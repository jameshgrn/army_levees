import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import os
import re

'''
Updates on _yl:
- Plot elevation profiles for every site and export to "plots"
'''


def plot_profiles(path_nld, path_tep):
    df_nld = gpd.read_parquet(path_nld)
    df_tep = gpd.read_parquet(path_tep)

    # Assuming 'distance_along_track' is a common column
    merged_df = pd.merge(df_nld, df_tep, on='distance_along_track', suffixes=('_nld', '_tep'))

    # Calculate MAE and RMSE
    mae = mean_absolute_error(merged_df['elevation_nld'], merged_df['elevation_tep'])
    rmse = sqrt(mean_squared_error(merged_df['elevation_nld'], merged_df['elevation_tep']))
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(merged_df['distance_along_track'], merged_df['elevation_nld'], label='NLD Profile', color='blue', marker='o', linestyle='-', markersize=3, linewidth=1)
    plt.plot(merged_df['distance_along_track'], merged_df['elevation_tep'], label='TEP Profile', color='red', marker='x', linestyle='--', markersize=3, linewidth=1)
    
    # Extract ID from file name
    id_match = re.search(r'(\d+)_nld', os.path.basename(path_nld))
    id_number = id_match.group(1) if id_match else 'Unknown'
    
    plt.title(f'Elevation Profiles Comparison - ID: {id_number}')
    plt.xlabel('Distance Along Track (m)')
    plt.ylabel('Elevation (m)')
    plt.legend()
    plt.grid(True)
    
    # Add MAE and RMSE to the plot
    plt.text(0.02, 0.98, f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}', transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'plots/profile_comparison_{id_number}.png')
    plt.close()

def process_all_files(data_folder):
    # Ensure the plots folder exists
    os.makedirs('plots', exist_ok=True)
    
    # Get all NLD files
    nld_files = [f for f in os.listdir(data_folder) if '_nld_' in f and f.endswith('.parquet')]
    
    plots_generated = 0
    
    for nld_file in nld_files:
        # Extract the ID and EPSG code from the NLD filename
        id_epsg_match = re.search(r'(\d+)_nld_(epsg\d+)', nld_file)
        if id_epsg_match:
            id_number, epsg_code = id_epsg_match.groups()
            # Construct the corresponding TEP file name pattern
            tep_pattern = f'{id_number}_tep_{epsg_code}.parquet'
            
            # Find the matching TEP file
            matching_tep_files = [f for f in os.listdir(data_folder) if f.endswith(tep_pattern)]
            
            if matching_tep_files:
                tep_file = matching_tep_files[0]
                nld_path = os.path.join(data_folder, nld_file)
                tep_path = os.path.join(data_folder, tep_file)
                
                print(f"Processing: {nld_file} and {tep_file}")
                plot_profiles(nld_path, tep_path)
                plots_generated += 1
            else:
                print(f"Warning: No matching TEP file found for {nld_file}")
        else:
            print(f"Warning: Unable to extract ID and EPSG code from {nld_file}")
    
    return plots_generated

# Run the processing for all files in the data folder
total_plots = process_all_files('data')

print(f"\nScript execution completed. {total_plots} profile comparisons have been generated and saved in the 'plots' folder.")