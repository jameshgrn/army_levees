import pandas as pd
import geopandas as gpd


def plot_profiles(path_nld, path_3dep):
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from math import sqrt
    import matplotlib.pyplot as plt
    df_nld = gpd.read_parquet(path_nld)
    df_3dep = gpd.read_parquet(path_3dep)


    # Assuming 'distance_along_track' is a common column
    merged_df = pd.merge(df_nld, df_3dep, on='distance_along_track', suffixes=('_nld', '_3dep'))

    # Calculate MAE and RMSE
    mae = mean_absolute_error(merged_df['elevation_nld'], merged_df['elevation_3dep'])
    rmse = sqrt(mean_squared_error(merged_df['elevation_nld'], merged_df['elevation_3dep']))
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(merged_df['distance_along_track'], merged_df['elevation_nld'], label='NLD Profile', color='blue', marker='o', linestyle='-', markersize=3, linewidth=1)
    plt.plot(merged_df['distance_along_track'], merged_df['elevation_3dep'], label='3DEP Profile', color='red', marker='x', linestyle='--', markersize=3, linewidth=1)
    plt.title(f'Elevation Profiles Comparison')
    plt.xlabel('Distance Along Track (m)')
    plt.ylabel('Elevation (m)')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_profiles('data/elevation_data_3405000052_nld_epsg26917.parquet', 'data/elevation_data_3405000052_tep_epsg26917.parquet')