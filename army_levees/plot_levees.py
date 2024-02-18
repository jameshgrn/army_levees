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
import matplotlib.patheffects as pe
from utils import read_and_parse_elevation_data, plot_profiles

df = gpd.read_parquet('../elevation_data.parquet')

# Use boolean indexing for filtering rows
plot_df = df[df['source'] == "tep"].groupby('system_id').first()

# Create a Stamen terrain background instance
terrain_background = GoogleTiles(style='satellite')

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

# Add the background at a certain zoom level
ax.add_image(terrain_background, 4)

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

# Now plot your data
plot_df.plot(ax=ax, column='elevation', cmap='cool', legend=True, alpha=0.5, norm=norm)


filepath = '../elevation_data.parquet'
system_ids = df['system_id'].unique()  # Replace with actual system IDs you're interested in
#system_ids = [2005100805]  # Replace with actual system IDs you're interested in
# Assuming 'source' column exists and can differentiate between 'nld' and '3dep' data
for system_id in system_ids:
    elevation_data_df = read_and_parse_elevation_data(filepath, [system_id])
    if elevation_data_df is not None:
        df_nld = elevation_data_df[elevation_data_df['source'] == 'nld']
        # df_nld['elevation'] = df_nld['elevation'] * .3048
        df_3dep = elevation_data_df[elevation_data_df['source'] == 'tep']
        # Thresholding to identify spikes
        elevation_change_threshold = 100  # Set a threshold for maximum allowed elevation change between consecutive points
        elevation_diff = df_3dep['elevation'].diff().abs()
        spikes = elevation_diff > elevation_change_threshold
        df_3dep.loc[spikes, 'elevation'] = np.nan
        # set 0 values to nan and then interpolate
        df_3dep['elevation'].replace(0, np.nan, inplace=True)

        # Interpolation to fill in the gaps
        df_3dep['elevation'] = df_3dep['elevation'].interpolate()

        # Remove the first and last values of both profiles before plotting
        df_nld_trimmed = df_nld.iloc[1:-1]
        df_3dep_trimmed = df_3dep.iloc[1:-1]

        plot_profiles(df_nld_trimmed, df_3dep_trimmed)



# %%
