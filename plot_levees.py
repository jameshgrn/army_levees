#%%
import cartopy.feature as cfeatures
import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from cartopy.io.img_tiles import GoogleTiles
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.patheffects as pe

df = gpd.read_parquet('elevation_data.parquet')

plot_df = df[df['source'] == "tep"].groupby('system_id').first()
# Create a Stamen terrain background instance
terrain_background = GoogleTiles(style='satellite')

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

# Add the background at a certain zoom level
ax.add_image(terrain_background, 4)

states_provinces = cfeatures.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='110m',
    facecolor='none')


# # Normalize the colormap
norm = colors.Normalize(vmin=plot_df['elevation'].min(), vmax=plot_df['elevation'].max())

# # Now plot your data
plot_df.plot(ax=ax, column='elevation', cmap='cool', legend=True, alpha=0.5, norm=norm)
ax.add_feature(states_provinces, edgecolor='k', lw=0.5)
# # %%
# import seaborn as sns
# # Plotting the distribution of 'relief'

# plt.figure(figsize=(10, 3))
# hist = sns.histplot(elevation_data_full['elevation'], kde=True, bins=50, stat='percent')
# plt.title('Distribution of Indiana Levee Relief')
# plt.xlabel('Relief')
# plt.ylabel('Percent of Levees')
# # Adding labels to each bar in the histogram
# for rect in hist.patches:
#     height = rect.get_height()
#     if height > 0:
#         plt.text(rect.get_x() + rect.get_width() / 2, height, f'{rect.get_x():.0f}', ha='center', va='bottom')
# plt.show()


# #%%
# plt.plot(elevation_data_full['riverMile'], elevation_data_full['relief'], 'o', alpha=0.5)



# # %%
# levee_centerline = gpd.read_file('/Users/jakegearon/CursorProjects/SideStatus/breaches/biggum/elevation_data_center.geojson')

# # Find the number of unique segment IDs
# num_segment_ids = levee_centerline['segmentId'].nunique()
# print(f"Number of unique segment IDs: {num_segment_ids}")

# # Group by 'segmentId' and check if any value in each group is NaN
# nan_segments = levee_centerline.groupby('segmentId').apply(lambda x: x.isna().any())

# # Count the number of groups that contain NaN values
# num_nan_segments = nan_segments.sum()

# print(f"Number of segments with NaN values: {num_nan_segments}")

# # %%
# import geopandas as gpd
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# valdf = pd.read_csv('/Users/jakegearon/CursorProjects/SideStatus/breaches/system_3705000033_profile.csv')
# spacing = np.floor(valdf['Distance'].diff().mean())



#eldf = plot_segment(3705000033)
# # %%
# valdf.loc[valdf['Elevation'] < 250, 'Elevation'] = np.nan
# valdf['Elevation'] = valdf['Elevation'].interpolate()
# valdf['INTERP'] = np.where(valdf['Elevation'].isna(), 1, 0)
# plt.plot(valdf['Distance'], valdf['Elevation'], 'o-', alpha=0.5)
# std_dev = np.std(valdf['Elevation'])
# residuals = valdf['Elevation'] - np.poly1d(np.polyfit(valdf['Distance'], valdf['Elevation'], 1))(valdf['Distance'])
# mean_residuals = np.mean(residuals)
# plt.title('System 1505000034 Profile')
# plt.text(0.05, 0.95, f'Standard Deviation: {std_dev:.2f}', transform=plt.gca().transAxes)
# plt.text(0.05, 0.90, f'Mean Residuals: {mean_residuals:.2f}', transform=plt.gca().transAxes)
# plt.xlabel('Distance along river (ft)')
# plt.ylabel('Elevation (ft)')
# plt.show()
# plt.close()
# # %%
# # Assuming levee1 and levee2 are your two dataframes
# plt.figure(figsize=(10, 6))

# max_height_df = elevation_data_full.groupby('distance')['elevation'].max().reset_index()
# # Apply a rolling median filter
# max_height_df['elevation'] = max_height_df['elevation'].rolling(window=6, center=True).median().fillna(method='bfill').fillna(method='ffill')
# # Convert distance and height to feet
# max_height_df['distance'] = max_height_df['distance'] * 3.281
# max_height_df['elevation'] = max_height_df['elevation'] * 3.281
# # Plot the maximum elevation for each distance
# # Plotting the two levees
# plt.plot(max_height_df['distance'], max_height_df['elevation'], 'o-', alpha=0.5, label='3DEP', markersize=3, color='red')
# plt.plot(valdf['Distance'], valdf['Elevation'], 'o-', alpha=0.5, label='NLD', markersize=3, color='blue')

# # Filling the area between the two levees

# plt.title('Comparison of 3DEP and NLD Levee Elevations')
# plt.xlabel('Distance along river (ft)')
# plt.ylabel('Elevation (ft)')
# plt.legend()
# plt.grid(True)
# plt.show()

# # %%
# plt.figure(figsize=(6, 3))
# sns.histplot(max_height_df['elevation'], bins=50, alpha=0.5, label='3DEP', color='red', stat='density')
# sns.histplot(valdf['Elevation'], bins=50, alpha=0.5, label='NLD', color='blue', stat='density')
# plt.title('Normalized Histogram of 3DEP and NLD Levee Elevations')
# plt.xlabel('Elevation (ft)')
# plt.ylabel('Density')
# plt.legend()
# plt.grid(True)
# plt.show()


# # %%
# import numpy as np

# # Assuming levee1 and levee2 are your two dataframes
# plt.figure(figsize=(10, 6))

# # Convert distance and height to feet for max_height_df

# # Interpolate max_height_df to match the spacing of valdf
# interpolated_elevations = np.interp(valdf['Distance'], max_height_df['distance'], max_height_df['elevation'])

# # Plotting the two levees
# plt.plot(valdf['Distance'], interpolated_elevations, 'o-', alpha=0.5, label='3DEP', markersize=3, color='red')
# plt.plot(valdf['Distance'], valdf['Elevation'], 'o-', alpha=0.5, label='NLD', markersize=3, color='blue')

# # Filling the area between the two levees
# plt.fill_between(valdf['Distance'], interpolated_elevations, valdf['Elevation'], color='gray', alpha=0.5)

# # Calculate the area of missing elevation and multiply by 1 m width to recover eroded volume
# missing_elevation = np.where(valdf['Elevation'] > interpolated_elevations, valdf['Elevation'] - interpolated_elevations, 0)
# eroded_volume = np.sum(missing_elevation) * 1  # Multiply by 1 m width
# print(f"Eroded volume: {eroded_volume} cubic feet")

# plt.title('Comparison of 3DEP and NLD Levee Elevations')
# plt.xlabel('Distance along river (ft)')
# plt.ylabel('Elevation (ft)')
# plt.legend()
# plt.grid(True)
# plt.show()
# # %%



#%%

#%%

# # Create a Stamen terrain background instance
# terrain_background = GoogleTiles(style='satellite')

# fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

# # Add the background at a certain zoom level
# ax.add_image(terrain_background, 15)

# states_provinces = cfeatures.NaturalEarthFeature(
#     category='cultural',
#     name='admin_1_states_provinces_lines',
#     scale='50m',
#     facecolor='none')

# ax.add_feature(states_provinces, edgecolor='k')

# # Normalize the colormap
# norm = colors.Normalize(vmin=elevation_data_full['elevation'].min(), vmax=elevation_data_full['elevation'].max())

# # Now plot your data
# elevation_data_full.plot(ax=ax, column='elevation', cmap='cool', legend=True, alpha=0.5, norm=norm, s=15)

# # Set bounds to the Indianapolis greater area
# #ax.set_extent([-86.33, -85.91, 39.63, 39.93])

# # Add title
# plt.title('Indiana Levee elevation from USGS 3DEP (m)')

# plt.show()
#%%
# el_df = elevation_data_full.copy()
# column = 'elevation'
# el_df = gpd.GeoDataFrame(el_df, geometry=gpd.points_from_xy(el_df['x'], el_df['y']), crs=4269)
# # Convert to a UTM CRS for accurate distance measurements
# #el_df = el_df.to_crs(epsg=4269)  # Replace with the correct UTM zone for your area
# if len(el_df) > 1:
#     # Sort by latitude (y) since the river flows south
#     el_df.sort_values(by='y', ascending=True, inplace=True)
#     max_height_df = el_df.groupby('distance')[column].max().reset_index()
#     # Apply a rolling median filter
#     max_height_df[column] = max_height_df[column].rolling(window=6, center=True).median().fillna(method='bfill').fillna(method='ffill')
#     # Convert distance and height to feet
#     max_height_df['distance'] = max_height_df['distance'] * 3.281
#     max_height_df[column] = max_height_df[column] * 3.281
#     # Plot the maximum elevation for each distance
#     plt.plot(max_height_df['distance'], max_height_df[column], 'o-', alpha=0.5, markersize=3)
#     # Calculate the standard deviation and mean residuals
#     std_dev = np.std(max_height_df[column])
#     residuals = max_height_df[column] - np.poly1d(np.polyfit(max_height_df['distance'], max_height_df[column], 1))(max_height_df['distance'])
#     mean_residuals = np.mean(residuals)
#     # Add plot details
#     # system_name = el_df['systemName'].iloc[0]
#     # plt.title(f'Segment ID: {segment_id} - System Name: {system_name}')
#     plt.text(0.05, 0.95, f'Standard Deviation: {std_dev:.2f}', transform=plt.gca().transAxes)
#     plt.text(0.05, 0.90, f'Mean Residuals: {mean_residuals:.2f}', transform=plt.gca().transAxes)
#     plt.xlabel('Distance along river (ft)')
#     plt.ylabel(f'Maximum {column} (ft)')
#     plt.show()
#     plt.close()

# %%
