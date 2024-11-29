#%%
import numpy as np
import cartopy.feature as cfeatures
import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from cartopy.io.img_tiles import GoogleTiles
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.colors import Normalize

import matplotlib.patheffects as pe
from utils import read_and_parse_elevation_data, plot_profiles

import ee
# ee.Authenticate()
ee.Initialize()

filepath = '/Users/jakegearon/projects/army_levees/elevation_data_100sample.parquet'
df = gpd.read_parquet(filepath)
print(type(df))
print(df.crs)
# Use boolean indexing for filtering rows
plot_df = df[df['source'] == "tep"].groupby('system_id').first()
# plot_df['x'] = plot_df.geometry.x
# plot_df['y'] = plot_df.geometry.y
plot_df = gpd.GeoDataFrame(plot_df, geometry=gpd.points_from_xy(plot_df['x'], plot_df['y']), crs="EPSG:3857")
plot_df = plot_df.to_crs(epsg=3857)  # Convert to Web Mercator projection for compatibility with the Stamen terrain background

# Create a Stamen terrain background instance
terrain_background = GoogleTiles(style='satellite')

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

# Add the background at a certain zoom level
ax.add_image(terrain_background, 2)

# Increase the line width and change the color for better visibility
states_provinces = cfeatures.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',  # Consider using a more detailed scale like '50m' if '110m' is not detailed enough
    facecolor='none'
)

# Plot the administrative lines after the satellite image
ax.add_feature(states_provinces, edgecolor='w', linestyle=':', lw=.5)  # Changed color to red, linestyle to dotted, and increased line width

# Normalize the colormap
norm = colors.Normalize(vmin=plot_df['elevation'].min(), vmax=plot_df['elevation'].max())

#add bounds that include the us
ax.set_extent([-125, -66.5, 20, 50], crs=ccrs.PlateCarree())

# Now plot your data
plot_df.plot(ax=ax, column='elevation', cmap='cool', legend=True, alpha=0.5, norm=norm)

#%%
system_ids = df['system_id'].unique()  # Replace with actual system IDs you're interested in
# system_ids = [3905000014, 3905000020]  # Replace with actual system IDs you're interested in
# Assuming 'source' column exists and can differentiate between 'nld' and '3dep' data
for system_id in system_ids:
    elevation_data_df = df[df['system_id'] == system_id]
    if elevation_data_df is not None:
        df_nld = elevation_data_df[elevation_data_df['source'] == 'nld']
        # df_nld['elevation'] = df_nld['elevation'] * .3048
        df_3dep = elevation_data_df[elevation_data_df['source'] == 'tep']
        # Thresholding to identify spikes
        elevation_change_threshold = 25  # Set a threshold for maximum allowed elevation change between consecutive points
        elevation_diff = df_3dep['elevation'].diff().abs()
        spikes = elevation_diff > elevation_change_threshold
        df_3dep.loc[spikes, 'elevation'] = np.nan
        # set 0 values to nan and then interpolate
        df_3dep['elevation'].replace(0, np.nan, inplace=True)

        # Interpolation to fill in the gaps
        df_3dep['elevation'] = df_3dep['elevation'].interpolate()

        # Remove the first and last values of both profiles before plotting
        df_nld_trimmed = df_nld.iloc[2:-2]
        df_3dep_trimmed = df_3dep.iloc[2:-2]

        plot_profiles(df_nld_trimmed, df_3dep_trimmed)

#%%
# Create a Stamen terrain background instance
terrain_background = GoogleTiles(style='satellite')

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 10))

# Add the background at a certain zoom level
ax.add_image(terrain_background, 15)

# Increase the line width and change the color for better visibility
states_provinces = cfeatures.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',  # Consider using a more detailed scale like '50m' if '110m' is not detailed enough
    facecolor='none'
)

# Plot the administrative lines after the satellite image
ax.add_feature(states_provinces, edgecolor='w', linestyle=':', lw=.5)  # Changed color to red, linestyle to dotted, and increased line width

# Normalize the colormap

diff_df = df_3dep[['geometry', 'system_id']].copy()
diff_df = diff_df.to_crs(epsg=4326)  # Convert to Web Mercator projection for compatibility with the Stamen terrain background
diff_df['elevation_diff'] = df_3dep['elevation'] - df_nld['elevation']
#plot diff in mapview
diff_df.dropna(subset=['elevation_diff'], inplace=True)
norm = colors.Normalize(vmin=diff_df['elevation_diff'].min(), vmax=diff_df['elevation_diff'].max())

diff_df.plot(ax=ax, cmap='rainbow', legend=True, markersize=10, edgecolor='k', linewidth=0.01, alpha=1, norm=norm, column='elevation_diff', legend_kwds={'label': 'Elevation Difference (m)'})
#plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='cool'), ax=ax, orientation='horizontal', label='Elevation Difference (m)')

# %%
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

# Assuming df is your initial DataFrame with all the data
elevation_data_df = gpd.read_parquet("/Users/jakegearon/projects/army_levees/elevation_data_100sample.parquet")
system_ids = elevation_data_df['system_id'].unique()

# Initialize an empty DataFrame to store mean differences for each system_id
mean_differences = pd.DataFrame(columns=['system_id', 'mean_elevation_diff'])

for system_id in system_ids:
    if elevation_data_df is not None:
        elevation_data_df_use = elevation_data_df[elevation_data_df['system_id'] == system_id]
        df_nld = elevation_data_df_use[elevation_data_df_use['source'] == 'nld']
        df_3dep = elevation_data_df_use[elevation_data_df_use['source'] == 'tep']
        
        # Ensure both datasets are not empty
        if not df_nld.empty and not df_3dep.empty:
            # Calculate mean elevation for each source
            mean_nld = df_nld['elevation'].mean()
            mean_3dep = df_3dep['elevation'].mean()
            print(mean_nld, mean_3dep)
            
            # Calculate the difference in mean elevations
            mean_diff = mean_3dep - mean_nld
            
            # Create a temporary DataFrame for the current system_id and its mean difference
            temp_df = pd.DataFrame({'system_id': [system_id], 'mean_elevation_diff': [mean_diff]})
            
            # Use concat instead of append
            mean_differences = pd.concat([mean_differences, temp_df], ignore_index=True)

# Now, you have a DataFrame with mean elevation differences for each system_id
# You can sort this DataFrame by the magnitude of the mean difference for plotting
mean_differences_sorted = mean_differences.abs().sort_values(by='mean_elevation_diff', ascending=False)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
mean_differences_sorted.plot(kind='bar', x='system_id', y='mean_elevation_diff', ax=ax, color='skyblue', legend=False)
ax.set_title('Mean Elevation Difference between NLD and TEP Datasets by System ID')
ax.set_xlabel('System ID')  # Adjust as necessary, e.g., to use names instead of IDs
ax.set_ylabel('Mean Elevation Difference (m)')
plt.xticks(rotation=45, ha="right")  # Improve label readability
plt.tight_layout()
plt.savefig('mean_elevation_diff_barplot.png', dpi=300)
plt.show()
# %%
import seaborn as sns
from scipy.stats import shapiro

# Log the values
logged_values = np.log(mean_differences_sorted['mean_elevation_diff'].dropna())

# Perform Shapiro-Wilk test for normality
stat, p_value = shapiro(logged_values)
print(f'Shapiro-Wilk Statistic: {stat}, p-value: {p_value}')

if p_value > 0.05:
    print('Data is normally distributed')
else:
    print('Data is not normally distributed')
# Plot distribution
sns.displot(logged_values, kde=True, color='skyblue')
plt.xlabel('Log of Mean Elevation Difference (m)')
# %%

# %%

# %%
import numpy as np
import cartopy.feature as cfeatures
import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from cartopy.io.img_tiles import GoogleTiles
from utils import read_and_parse_elevation_data, plot_profiles

# Load the elevation data
filepath = '/Users/jakegearon/projects/army_levees/elevation_data_100sample.parquet'
df = gpd.read_parquet(filepath)

# Filter and prepare plot_df
plot_df = df[df['source'] == "tep"].groupby('system_id').first()
plot_df = gpd.GeoDataFrame(plot_df, geometry=gpd.points_from_xy(plot_df['x'], plot_df['y']), crs="EPSG:3857")
plot_df = plot_df.to_crs(epsg=3857)  # Convert to Web Mercator projection

# Merge the mean differences DataFrame
mean_differences['system_id'] = mean_differences['system_id'].astype(int)  # Ensure system_id is integer if not already
plot_df = plot_df.merge(mean_differences, on='system_id', how='left')

# Handle NaN values and ensure data types are correct
plot_df['mean_elevation_diff'] = plot_df['mean_elevation_diff'].fillna(0).astype(float)

# Bin the mean differences into three categories
plot_df['category'] = pd.cut(plot_df['mean_elevation_diff'], bins=3, labels=['Low', 'Medium', 'High'])

# Create a Stamen terrain background instance
terrain_background = GoogleTiles(style='satellite')
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

# Add the background at a certain zoom level
ax.add_image(terrain_background, 7)

# Add administrative lines for better visibility
states_provinces = cfeatures.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none'
)
ax.add_feature(states_provinces, edgecolor='w', linestyle=':', lw=.5)

# Set bounds to include the US
ax.set_extent([-125, -66.5, 20, 50], crs=ccrs.PlateCarree())

# Plot the data using categories as the color
scatter = plot_df.plot(ax=ax, column='category', cmap='viridis', legend=True, alpha=1)

# Create and add the colorbar manually
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=2))
sm._A = []
plt.colorbar(sm, ax=ax, label='Elevation Difference Category')
plt.savefig('elevation_diff_map_categorized.png', dpi=300)
plt.show()
# %%
