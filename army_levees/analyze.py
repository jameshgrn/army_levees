#%%
import os
os.chdir('/Users/jakegearon/CursorProjects/army_levees')
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, ks_2samp
from scipy.integrate import simps
from scipy.signal import correlate


def read_and_parse_elevation_data(filepath, system_ids=None):
    import geopandas as gpd
    """
    Reads elevation data for specified system IDs from a Parquet file without loading the entire DataFrame into memory.

    Parameters:
    - filepath: str, path to the Parquet file containing elevation data.
    - system_ids: list of str, system IDs to filter the data. If None, all data is loaded.

    Returns:
    - DataFrame with parsed elevation data for the specified system IDs.
    """
    try:
        # If system_ids is provided, prepare a filter
        if system_ids:
            filters = [('system_id', 'in', system_ids)]
            # Use the 'columns' parameter if you want to load specific columns only, e.g., ['system_id', 'elevation']
            df = gpd.read_parquet(filepath, filters=filters)
        else:
            df = gpd.read_parquet(filepath)
    except Exception as e:
        print(f"Failed to read the Parquet file: {e}")
        return None

    # Further processing can be done here if needed
    return df

# Example usage
#%%
filepath = 'elevation_data.parquet'
system_ids = [2105000001]  # Replace with actual system IDs you're interested in
# Assuming 'source' column exists and can differentiate between 'nld' and '3dep' data
elevation_data_df = read_and_parse_elevation_data(filepath, system_ids)
if elevation_data_df is not None:
    df_nld = elevation_data_df[elevation_data_df['source'] == 'nld']
    # df_nld['elevation'] = df_nld['elevation'] * .3048
    df_3dep = elevation_data_df[elevation_data_df['source'] == 'tep']


# Assuming df_nld and df_3dep are your DataFrames for each profile

# Function to calculate per-profile statistics
def calculate_statistics(df):
    stats = {
        'mean': np.mean(df['elevation']),
        'median': np.median(df['elevation']),
        'std_dev': np.std(df['elevation']),
        'kurtosis': kurtosis(df['elevation']),
        'roughness': np.sum(np.abs(np.diff(df['elevation'])))/len(df)
    }
    return stats

# Function to calculate combined statistics
def calculate_combined_statistics(df1, df2):
    # Ensure the data is sorted or interpolated so they align for comparison
    # This is a placeholder for actual alignment/interpolation code
    combined_stats = {}
    area_between_curves = simps(np.abs(df1['elevation'] - df2['elevation']))
    combined_stats['area_between_curves'] = area_between_curves
    
    # Cross-correlation
    corr = correlate(df1['elevation'], df2['elevation'])
    combined_stats['cross_correlation'] = np.max(corr)
    
    # KS Test
    ks_stat, ks_pvalue = ks_2samp(df1['elevation'], df2['elevation'])
    combined_stats['ks_stat'] = ks_stat
    combined_stats['ks_pvalue'] = ks_pvalue
    
    return combined_stats

# Example usage
stats_nld = calculate_statistics(df_nld)
stats_3dep = calculate_statistics(df_3dep)
combined_stats = calculate_combined_statistics(df_nld, df_3dep)

# Combine all stats into a single dictionary or DataFrame for further processing or saving
all_stats = {**stats_nld, **stats_3dep, **combined_stats}

# Convert dictionary to DataFrame
stats_df = pd.DataFrame([all_stats])

# Save to Parquet
stats_df.to_parquet('profile_statistics.parquet')


#%%
from utils import plot_profiles
# Plot the profiles
plot_profiles(df_nld, df_3dep)
# %%
