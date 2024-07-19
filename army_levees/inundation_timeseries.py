import requests, json, random
import numpy as np
import pandas as pd
import rasterio
import os
import geopandas as gpd
import matplotlib.pyplot as plt
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from shapely.geometry import LineString, MultiLineString
from tqdm import tqdm
from shapely.geometry import Polygon
import statsmodels.api as sm
import eemont
import py3dep
from requests.exceptions import RetryError
import time
import ee
from sklearn.metrics import mean_squared_error
import geemap
from scipy.stats import linregress
from utils import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import matplotlib.dates as mdates
# Initialize the Earth Engine library.
ee.Authenticate()
ee.Initialize()

CRS = "EPSG:4269"
get_url = 'https://levees.sec.usace.army.mil:443/api-local/system-categories/usace-nonusace'


def plot_index_time_series(index_name, roi_first, roi_second, title_first, title_second):
    ts_first = col.getTimeSeriesByRegion(reducer=[ee.Reducer.mean()],
                                         geometry=roi_first,
                                         bands=[index_name],
                                         scale=30)
    ts_df_first = geemap.ee_to_df(ts_first, columns=['date', index_name])
    ts_df_first['date'] = pd.to_datetime(ts_df_first['date'])
    ts_df_first = ts_df_first.sort_values('date')
    ts_df_first[index_name] = ts_df_first[index_name].apply(lambda x: x if x > -1 else np.nan)
    ts_df_clean_first = ts_df_first.dropna()
    
    # Time series and data extraction for the second geometry
    ts_second = col.getTimeSeriesByRegion(reducer=[ee.Reducer.mean()],
                                          geometry=roi_second,
                                          bands=[index_name],
                                          scale=30)
    ts_df_second = geemap.ee_to_df(ts_second, columns=['date', index_name])
    ts_df_second['date'] = pd.to_datetime(ts_df_second['date'])
    ts_df_second = ts_df_second.sort_values('date')
    ts_df_second[index_name] = ts_df_second[index_name].apply(lambda x: x if x > -1 else np.nan)
    ts_df_clean_second = ts_df_second.dropna()
    
    # Read and resample the discharge data
    df_discharge = pd.read_csv('/Users/jakegearon/CursorProjects/army_levees/Wabash_northofclifton.txt', sep='\t', header=None)
    df_discharge.columns = ['source', 'station_id', 'date', 'discharge', 'flag']
    df_discharge['date'] = pd.to_datetime(df_discharge['date'])
    df_discharge.set_index('date', inplace=True)
    df_discharge_monthly = df_discharge['discharge'].resample('M').mean().reset_index()
        # Clip the discharge data to match the time series date range
    min_date = min(ts_df_clean_first['date'].min(), ts_df_clean_second['date'].min())
    max_date = max(ts_df_clean_first['date'].max(), ts_df_clean_second['date'].max())
    df_discharge_monthly_clipped = df_discharge_monthly[(df_discharge_monthly['date'] >= min_date) & (df_discharge_monthly['date'] <= max_date)]
    
    # Plotting setup
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot time series for the first geometry on the correct axis
    axs[0, 0].plot(ts_df_clean_first['date'], ts_df_clean_first[index_name], color='blue', label=title_first)
    axs[0, 0].set_title(title_first)
    axs[0, 0].set_ylabel(index_name)
    axs[0, 0].legend(loc='upper left')
    
    # Plot resampled discharge data on a twin axis for the first geometry
    ax2 = axs[0, 0].twinx()
    ax2.plot(df_discharge_monthly_clipped['date'], df_discharge_monthly_clipped['discharge'], color='purple', linestyle='--', label='Discharge')
    ax2.set_ylabel('Discharge')
    ax2.legend(loc='upper right')
    
    # Plot time series for the second geometry on the correct axis
    axs[0, 1].plot(ts_df_clean_second['date'], ts_df_clean_second[index_name], color='red', label=title_second)
    axs[0, 1].set_title(title_second)
    axs[0, 1].set_ylabel(index_name)
    axs[0, 1].legend(loc='upper left')
    
    # Plot resampled discharge data on a twin axis for the second geometry
    ax3 = axs[0, 1].twinx()
    ax3.plot(df_discharge_monthly_clipped['date'], df_discharge_monthly_clipped['discharge'], color='purple', linestyle='--', label='Discharge')
    ax3.set_ylabel('Discharge')
    ax3.legend(loc='upper right')
    merged_df = pd.merge(ts_df_clean_first, ts_df_clean_second, on='date', suffixes=('_first', '_second'))
    slope, intercept, r_value, p_value, std_err = linregress(merged_df[f'{index_name}_first'], merged_df[f'{index_name}_second'])
    
    # Convert dates to a numerical format for color coding
    merged_df['date_num'] = pd.to_datetime(merged_df['date']).apply(lambda x: x.toordinal())
    # Normalize the date_num to use in color mapping
    date_num_normalized = (merged_df['date_num'] - merged_df['date_num'].min()) / (merged_df['date_num'].max() - merged_df['date_num'].min())
    
    # Create a scatter plot with points colored by date
    scatter = axs[1, 0].scatter(merged_df[f'{index_name}_first'], merged_df[f'{index_name}_second'], c=date_num_normalized, cmap='Greens', label='Data points by date')
    axs[1, 0].plot(merged_df[f'{index_name}_first'], intercept + slope * merged_df[f'{index_name}_first'], 'g', label='fitted line')
    axs[1, 0].plot(merged_df[f'{index_name}_first'], merged_df[f'{index_name}_first'], 'k--', linewidth=1.5, label='1:1 line')  # Added black thin 1:1 line
    axs[1, 0].set_xlabel(f'{title_first} {index_name}', color='blue')  # Color the xlabel blue
    axs[1, 0].set_ylabel(f'{title_second} {index_name}', color='red')  # Color the ylabel red
    axs[1, 0].legend()
    axs[1, 0].set_title('Linear Regression between Geometries')
    
    # Adding a colorbar to indicate the mapping of color to date
    cbar = plt.colorbar(scatter, ax=axs[1, 0])
    cbar.set_label('Date (older to newer)')
    
    # Histograms of the index values for both geometries
    axs[1, 1].hist(ts_df_clean_first[index_name], bins=20, alpha=0.5, label=title_first, color='blue')
    axs[1, 1].hist(ts_df_clean_second[index_name], bins=20, alpha=0.5, label=title_second, color='red')
    axs[1, 1].set_xlabel(f'{index_name} Value')
    axs[1, 1].set_ylabel('Frequency')
    axs[1, 1].legend()
    axs[1, 1].set_title('Histograms of Index Values')
    
    plt.tight_layout()
    plt.show()

def plot_index_time_series_with_regression(index_name, roi_first, roi_second, title_first, title_second):
    # Assuming previous steps to prepare ts_df_clean_first, ts_df_clean_second, and df_discharge_monthly_clipped are correct
    ts_first = col.getTimeSeriesByRegion(reducer=[ee.Reducer.mean()],
                                         geometry=roi_first,
                                         bands=[index_name],
                                         scale=30)
    ts_df_first = geemap.ee_to_df(ts_first, columns=['date', index_name])
    ts_df_first['date'] = pd.to_datetime(ts_df_first['date'])
    ts_df_first['date'] = pd.to_datetime(ts_df_first['date']).dt.date

    ts_df_first = ts_df_first.sort_values('date')
    ts_df_first[index_name] = ts_df_first[index_name].apply(lambda x: x if x > -1 else np.nan)
    ts_df_clean_first = ts_df_first.dropna()
    
    # Time series and data extraction for the second geometry
    ts_second = col.getTimeSeriesByRegion(reducer=[ee.Reducer.mean()],
                                          geometry=roi_second,
                                          bands=[index_name],
                                          scale=30)
    ts_df_second = geemap.ee_to_df(ts_second, columns=['date', index_name])
    ts_df_second['date'] = pd.to_datetime(ts_df_second['date'])
    ts_df_second['date'] = pd.to_datetime(ts_df_second['date']).dt.date

    ts_df_second = ts_df_second.sort_values('date')
    ts_df_second[index_name] = ts_df_second[index_name].apply(lambda x: x if x > -1 else np.nan)
    ts_df_clean_second = ts_df_second.dropna()
    
    # Convert all date columns to datetime.date format
    
    # Read and resample the discharge data
    df_discharge = pd.read_csv('/Users/jakegearon/CursorProjects/army_levees/Wabash_northofclifton.txt', sep='\t', header=None)
    df_discharge.columns = ['source', 'station_id', 'date', 'discharge', 'flag']
    df_discharge['date'] = pd.to_datetime(df_discharge['date']).dt.date

    # Ensure 'date' column is in datetime.date format for both ts_df_clean_first and ts_df_clean_second
    ts_df_clean_first['date'] = pd.to_datetime(ts_df_clean_first['date']).dt.date
    ts_df_clean_second['date'] = pd.to_datetime(ts_df_clean_second['date']).dt.date

    # Filter the discharge data to only include dates that exist in both ts_df_clean_first and ts_df_clean_second
    common_dates = set(ts_df_clean_first['date']).intersection(set(ts_df_clean_second['date']))
    df_discharge_filtered = df_discharge[df_discharge['date'].isin(common_dates)]
    #df_discharge_filtered.set_index('date', inplace=True)


# Now df_discharge_aligned should have discharge data aligned with the time series data
# You can now proceed with merging this dataframe with ts_df_clean_first and ts_df_clean_second
    # Clip the discharge data to match the time series date range
    min_date = min(ts_df_clean_first['date'].min(), ts_df_clean_second['date'].min())
    max_date = max(ts_df_clean_first['date'].max(), ts_df_clean_second['date'].max())
    df_discharge_aligned_clipped = df_discharge_filtered[(df_discharge_filtered['date'] >= min_date) & (df_discharge_filtered['date'] <= max_date)]
    # Convert datetime to date for merging


    # Then proceed with the merging as before
    # Plotting setup with 3 rows and 2 columns
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    
    # Plot time series for the first geometry on the correct axis
    axs[0, 0].plot(ts_df_clean_first['date'], ts_df_clean_first[index_name], color='blue', label=title_first)
    axs[0, 0].set_title(title_first)
    axs[0, 0].set_ylabel(index_name)
    axs[0, 0].legend(loc='upper left')
    
    # Plot resampled discharge data on a twin axis for the first geometry
    ax2 = axs[0, 0].twinx()
    ax2.plot(df_discharge_aligned_clipped['date'], df_discharge_aligned_clipped['discharge'], color='black', linestyle='--', label='Discharge', lw=.5)
    ax2.set_ylabel('Discharge')
    ax2.legend(loc='upper right')
    
    # Plot time series for the second geometry on the correct axis
    axs[0, 1].plot(ts_df_clean_second['date'], ts_df_clean_second[index_name], color='red', label=title_second)
    axs[0, 1].set_title(title_second)
    axs[0, 1].set_ylabel(index_name)
    axs[0, 1].legend(loc='upper left')
    
    # Plot resampled discharge data on a twin axis for the second geometry
    ax3 = axs[0, 1].twinx()
    ax3.plot(df_discharge_aligned_clipped['date'], df_discharge_aligned_clipped['discharge'], color='black', linestyle='--', label='Discharge', lw=0.5)
    ax3.set_ylabel('Discharge')
    ax3.legend(loc='upper right')
    
    # Linear Regression Scatter Plot between the two geometries
    merged_df = pd.merge(ts_df_clean_first, ts_df_clean_second, on='date', suffixes=('_first', '_second'))
    slope, intercept, r_value, p_value, std_err = linregress(merged_df[f'{index_name}_first'], merged_df[f'{index_name}_second'])
    merged_df['date_num'] = pd.to_datetime(merged_df['date']).apply(lambda x: x.toordinal())
    # Normalize the date_num to use in color mapping
    date_num_normalized = (merged_df['date_num'] - merged_df['date_num'].min()) / (merged_df['date_num'].max() - merged_df['date_num'].min())
    scatter = axs[1, 0].scatter(merged_df[f'{index_name}_first'], merged_df[f'{index_name}_second'], c=date_num_normalized, cmap='Greens', label='Data points by date')
    axs[1, 0].plot(merged_df[f'{index_name}_first'], intercept + slope * merged_df[f'{index_name}_first'], 'g', label='fitted line')
    axs[1, 0].plot(merged_df[f'{index_name}_first'], merged_df[f'{index_name}_first'], 'k--', linewidth=1.5, label='1:1 line')  # Added black thin 1:1 line
    axs[1, 0].set_xlabel(f'{title_first} {index_name}', color='blue')  # Color the xlabel blue
    axs[1, 0].set_ylabel(f'{title_second} {index_name}', color='red')  # Color the ylabel red
    axs[1, 0].legend()
    axs[1, 0].set_title('Linear Regression between Geometries')
    
    # Histograms of the index values for both geometries
    axs[1, 1].hist(ts_df_clean_first[index_name], bins=20, alpha=0.5, label=title_first, color='blue')
    axs[1, 1].hist(ts_df_clean_second[index_name], bins=20, alpha=0.5, label=title_second, color='red')
    axs[1, 1].set_xlabel(f'{index_name} Value')
    axs[1, 1].set_ylabel('Frequency')
    axs[1, 1].legend()
    axs[1, 1].set_title('Histograms of Index Values')

    # Merge discharge data on date with the merged_df for regression analysis
    merged_with_discharge = pd.merge(merged_df, df_discharge_aligned_clipped, on='date')

    # Check if the merged data frame is empty
    if merged_with_discharge.empty:
        print("No overlapping data between time series and discharge data. Skipping regression analysis.")
        return

    # Log-transform the 'discharge' values for linear regression
    log_discharge = np.log(merged_with_discharge['discharge'])

    # Convert series to numpy arrays for statsmodels
    X_first = sm.add_constant(log_discharge)  # Adding a constant for the intercept
    Y_first = merged_with_discharge[f'{index_name}_first'].values
    Y_second = merged_with_discharge[f'{index_name}_second'].values

    # Perform robust linear regression using RANSAC for the first geometry
    ransac_first = sm.RLM(Y_first, X_first, M=sm.robust.norms.HuberT())
    results_first = ransac_first.fit()

    # Perform robust linear regression using RANSAC for the second geometry
    ransac_second = sm.RLM(Y_second, X_first, M=sm.robust.norms.HuberT())
    results_second = ransac_second.fit()


    # Calculate the fit line using the robust regression results
    new_x_first = np.logspace(np.log10(merged_with_discharge['discharge'].min()), np.log10(merged_with_discharge['discharge'].max()), base=10)
    new_X_first = sm.add_constant(np.log(new_x_first))
    # Predictions from the robust regression models
    predictions_first = results_first.predict(X_first)
    predictions_second = results_second.predict(X_first)

    # Calculate MSE for the first geometry
    mse_first = mean_squared_error(Y_first, predictions_first)

    # Calculate MSE for the second geometry
    mse_second = mean_squared_error(Y_second, predictions_second)

        # Extract p-values for the slope coefficients from the robust regression models
    p_value_first = results_first.pvalues[1]  # p-value for the slope of the first geometry model
    p_value_second = results_second.pvalues[1]  # p-value for the slope of the second geometry model

    # Use the p-values in your plots or analysis
    # For example, adding p-values to the plot legends
    axs[2, 0].scatter(merged_with_discharge['discharge'], Y_first, color='blue', label='First Geometry')
    axs[2, 0].plot(new_x_first, results_first.predict(new_X_first), 'b--', label=f'Fit Line (MSE={mse_first:.2f}, p={p_value_first:.2e})')
    axs[2, 0].set_xlabel('Discharge')
    axs[2, 0].set_ylabel(f'{index_name} First Geometry')
    axs[2, 0].legend()
    axs[2, 0].set_title('Robust Regression: First Geometry Index vs. Discharge')
    axs[2, 0].set_xscale('log')

    axs[2, 1].scatter(merged_with_discharge['discharge'], Y_second, color='red', label='Second Geometry')
    axs[2, 1].plot(new_x_first, results_second.predict(new_X_first), 'r--', label=f'Fit Line (MSE={mse_second:.2f}, p={p_value_second:.2e})')
    axs[2, 1].set_xlabel('Discharge')
    axs[2, 1].set_ylabel(f'{index_name} Second Geometry')
    axs[2, 1].legend()
    axs[2, 1].set_title('Robust Regression: Second Geometry Index vs. Discharge')
    axs[2, 1].set_xscale('log')

    plt.tight_layout()
    plt.show()

    
if __name__ == '__main__':
    system_ids = load_usace_system_ids()
    usace_ids_url = 'https://levees.sec.usace.army.mil:443/api-local/system-categories/usace-nonusace'
    download_usace_system_ids(usace_ids_url)
    with open('./indiana_system_ids.json', 'r') as file:
        indiana_system_ids = json.load(file)
    indiana_system_ids = [3905000014, 3905000020]
    for system_id in indiana_system_ids:
        states = get_state_info(system_id)
        if 'Indiana' in states:
            print("System ID", system_id, "is in Indiana.")
            coords_2d = get_leveed_area(system_id)
            profile_data, profile_gdf = get_profile_data(system_id)
            # Create a Polygon from coords_2d
            polygon = Polygon(coords_2d)
            from shapely.geometry import LineString

            # Assuming profile_gdf is a GeoDataFrame with Point geometries
            points = [point for point in profile_gdf.geometry]

            # Create a LineString from these points
            line = LineString(points)

            # If you need to create a GeoDataFrame from this LineString
            line_gdf = gpd.GeoDataFrame(index=[0], crs=profile_gdf.crs, geometry=[line])
            #line_gdf.to_crs("EPSG:3857").plot()
            profile_gdf_buffered_first = line_gdf.to_crs("EPSG:3857").buffer(250, single_sided=True).to_crs("EPSG:4326").simplify(0.0001)
            profile_gdf_buffered_second = line_gdf.to_crs("EPSG:3857").buffer(-250, single_sided=True).to_crs("EPSG:4326").simplify(0.0001)
            fig, ax = plt.subplots()
            profile_gdf_buffered_first.plot(ax=ax, color='blue')
            profile_gdf_buffered_second.plot(ax=ax, color='red')
            plt.show()
        
            # Create a GeoDataFrame
            gdf = gpd.GeoDataFrame(index=[0], crs="EPSG:4629", geometry=[polygon])
            #ee_geometry = ee.Geometry.Polygon([coords_2d])
            # Extract the coordinates as a list of lists
            coords_list_first = profile_gdf_buffered_first.geometry.apply(extract_polygon_coords).tolist()
            ee_geometry_first = ee.Geometry.Polygon(coords_list_first[0])
            coords_list_second = profile_gdf_buffered_second.geometry.apply(extract_polygon_coords).tolist()
            ee_geometry_second = ee.Geometry.Polygon(coords_list_second[0])
            # Create an ee.Geometry.Polygon
            #ee_geometry = ee.Geometry.Polygon([coords_list])
            oliCol = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR');
            etmCol = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR');
            tmCol = ee.ImageCollection('LANDSAT/LT05/C01/T1_SR');
            colFilter = ee.Filter.And(
                ee.Filter.bounds(ee_geometry_first),
                ee.Filter.lt('CLOUD_COVER', 50), ee.Filter.lt('GEOMETRIC_RMSE_MODEL', 10),
                ee.Filter.Or(
                    ee.Filter.eq('IMAGE_QUALITY', 9),
                    ee.Filter.eq('IMAGE_QUALITY_OLI', 9)))

            # Filter collections and prepare them for merging.
            oliCol = oliCol.filter(colFilter).map(prep_oli)
            etmCol = etmCol.filter(colFilter).map(prep_etm)
            tmCol = tmCol.filter(colFilter).map(prep_tm)

            # Merge the collections.
            col = oliCol.merge(etmCol).merge(tmCol)
        
            #plot_water_area_time_series(col, ee_geometry_first, ee_geometry_second, 'First Geometry', 'Second Geometry')
            plot_index_time_series_with_regression('NDWI_NS', ee_geometry_first, ee_geometry_second, 'First Geometry', 'Second Geometry')
        else:
            print("System ID", system_id, "is not in Indiana and therefore useless.")

#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/Users/jakegearon/CursorProjects/army_levees/Wabash_northofclifton.txt', sep='\t', header=None)
df.columns = ['source', 'station_id', 'date', 'discharge', 'flag']
df['date'] = pd.to_datetime(df['date'])
# %%
