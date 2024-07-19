#%%
import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeatures
from cartopy.io.img_tiles import GoogleTiles
import matplotlib.colors as colors
# Directory where the elevation data files are saved
data_dir = '/Users/jakegearon/projects/army_levees/data'

# Initialize an empty list to store GeoDataFrames
gdfs = []

# Loop through each file in the directory
for filename in os.listdir(data_dir):
    if filename.endswith('.parquet'):
        filepath = os.path.join(data_dir, filename)
        gdf = gpd.read_parquet(filepath)
        gdf = gdf.to_crs(epsg=4326)  # Convert to EPSG:3857 for consistency
        gdfs.append(gdf)

# Concatenate all GeoDataFrames into a single GeoDataFrame
plot_df = pd.concat(gdfs, ignore_index=True)


#%%
# Create a Stamen terrain background instance
terrain_background = GoogleTiles(style='satellite')

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 10))

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

# Normalize the colormap
norm = colors.Normalize(vmin=plot_df['elevation'].min(), vmax=plot_df['elevation'].max())

# Plot the data
scatter = plot_df.plot(ax=ax, column='elevation', cmap='cool', legend=False, alpha=0.7, norm=norm, markersize=30, edgecolor='k')

# Create and add the colorbar manually
sm = plt.cm.ScalarMappable(cmap='cool', norm=norm)
sm._A = []
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', label='Elevation (m)')

plt.savefig('elevation_diff_map.png', dpi=300)
plt.show()
#%%
for filename in os.listdir(data_dir):
    if filename.endswith('.parquet'):
        filepath = os.path.join(data_dir, filename)
        gdf = gpd.read_parquet(filepath)
        
        # Plotting individual profiles
        fig, ax = plt.subplots()
        gdf.plot(ax=ax, column='elevation', cmap='cool', legend=True)
        plt.title(f'Elevation Profile for {filename}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.savefig(f'{filename}_profile.png', dpi=300)
        plt.show()
# %%
