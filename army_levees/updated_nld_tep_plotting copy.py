import os
import pandas as pd
import geopandas as gpd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from scipy.stats import zscore

data_dir = 'data'  # Directory where the elevation data files are saved
plots_dir = 'plots'  # Directory where the plots will be saved

# Make sure the plots directory exists
os.makedirs(plots_dir, exist_ok=True)

def calculate_mean_difference(path_nld, path_3dep, zero_threshold=0.5, min_non_zero_values=10):
    df_nld = gpd.read_parquet(path_nld)
    df_3dep = gpd.read_parquet(path_3dep)

    # Filter out rows where NLD elevation values are consistently zero
    zero_count = (df_nld['elevation'] == 0).sum()
    total_count = len(df_nld)
    non_zero_count = total_count - zero_count
    if zero_count / total_count > zero_threshold or non_zero_count < min_non_zero_values:
        return None

    # Assuming 'distance_along_track' is a common column
    merged_df = pd.merge(df_nld, df_3dep, on='distance_along_track', suffixes=('_nld', '_3dep'))

    # Filter out rows with NaN values in elevation columns
    merged_df = merged_df.dropna(subset=['elevation_nld', 'elevation_3dep'])

    # Calculate elevation differences
    merged_df['elevation_diff'] = merged_df['elevation_3dep'] - merged_df['elevation_nld']

    # Filter out rows with extreme elevation differences (e.g., |diff| > 1000)
    merged_df = merged_df[merged_df['elevation_diff'].abs() <= 1000]

    # Calculate z-scores for elevation columns
    merged_df['zscore_nld'] = zscore(merged_df['elevation_nld'])
    merged_df['zscore_3dep'] = zscore(merged_df['elevation_3dep'])

    # Debug: print z-score statistics
    print(f"Z-score NLD: mean={merged_df['zscore_nld'].mean()}, std={merged_df['zscore_nld'].std()}")
    print(f"Z-score 3DEP: mean={merged_df['zscore_3dep'].mean()}, std={merged_df['zscore_3dep'].std()}")

    # Remove rows with NaN z-scores
    merged_df = merged_df.dropna(subset=['zscore_nld', 'zscore_3dep'])

    # Remove rows with z-scores beyond a threshold (e.g., |z| > 3)
    merged_df = merged_df[(merged_df['zscore_nld'].abs() <= 3) & (merged_df['zscore_3dep'].abs() <= 3)]

    # Check if the merged dataframe is empty after filtering
    if merged_df.empty:
        return None

    # Calculate z-scores for elevation differences
    merged_df['zscore_diff'] = zscore(merged_df['elevation_diff'])

    # Debug: print z-score difference statistics
    print(f"Z-score Diff: mean={merged_df['zscore_diff'].mean()}, std={merged_df['zscore_diff'].std()}")

    # Remove rows with z-scores for elevation differences beyond a threshold (e.g., |z| > 3)
    merged_df = merged_df[merged_df['zscore_diff'].abs() <= 3]

    # Check if the merged dataframe is empty after filtering
    if merged_df.empty:
        return None

    # Calculate mean elevation difference
    mean_diff = merged_df['elevation_diff'].mean()
    return mean_diff

def plot_profiles(path_nld, path_3dep, system_id, zero_threshold=0.5, min_non_zero_values=10):
    df_nld = gpd.read_parquet(path_nld)
    df_3dep = gpd.read_parquet(path_3dep)

    # Filter out rows where NLD elevation values are zero
    df_nld = df_nld[df_nld['elevation'] != 0]

    # Check if there are enough non-zero values
    if len(df_nld) < min_non_zero_values:
        print(f"System ID: {system_id} - Insufficient non-zero values in NLD profile.")
        return

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
        print(f"System ID: {system_id} - No valid data after filtering.")
        return

    # Calculate statistics
    mae = mean_absolute_error(merged_df['elevation_nld'], merged_df['elevation_3dep'])
    rmse = sqrt(mean_squared_error(merged_df['elevation_nld'], merged_df['elevation_3dep']))
    mean_diff = merged_df['elevation_diff'].mean()
    print(f"System ID: {system_id}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Elevation Difference: {mean_diff}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(merged_df['distance_along_track'], merged_df['elevation_nld'], label='NLD Profile', color='blue', linestyle='-', markersize=3, linewidth=1)
    plt.plot(merged_df['distance_along_track'], merged_df['elevation_3dep'], label='3DEP Profile', color='red', linestyle='-', markersize=3, linewidth=1)
    plt.title(f'Elevation Profiles Comparison for System ID: {system_id}\nMean Elevation Difference: {mean_diff:.2f} m')
    plt.xlabel('Distance Along Track (m)')
    plt.ylabel('Elevation (m)')
    plt.legend()
    plt.grid(True)
    
    # Save the plot instead of showing it
    plot_filename = os.path.join(plots_dir, f'profile_filtered_{system_id}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved as {plot_filename}")

def process_all_files(data_dir):
    files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
    nld_files = [f for f in files if 'nld' in f]
    tep_files = [f for f in files if 'tep' in f]

    mean_diffs = []
    for nld_file in nld_files:
        system_id = nld_file.split('_')[2]
        corresponding_tep_file = next((f for f in tep_files if system_id in f), None)
        if corresponding_tep_file:
            mean_diff = calculate_mean_difference(os.path.join(data_dir, nld_file), os.path.join(data_dir, corresponding_tep_file))
            if mean_diff is not None:
                mean_diffs.append((system_id, nld_file, corresponding_tep_file, mean_diff))

    # Sort by mean elevation difference in descending order
    mean_diffs.sort(key=lambda x: x[3], reverse=True)

    for system_id, nld_file, tep_file, mean_diff in mean_diffs:
        print(f"Processing NLD file: {nld_file} and TEP file: {tep_file} with mean difference: {mean_diff}")
        plot_profiles(os.path.join(data_dir, nld_file), os.path.join(data_dir, tep_file), system_id)

if __name__ == "__main__":
    process_all_files(data_dir)