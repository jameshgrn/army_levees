import requests
import json
from get_3dep import process_segment
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
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from shapely.geometry import shape

# Convert TopoJSON to GeoJSON (this is a simplified example, you might need to adjust it)
def topojson_to_geojson(topojson):
    # This function assumes that there is only one object and it's a LineString
    line_data = topojson['geometry']['arcs'][0]
    # Convert line data to a list of tuples (ignoring the third and fourth elements)
    line_coords = [(x, y) for x, y, *_ in line_data]
    # Create a LineString geometry
    geometry = {'type': 'LineString', 'coordinates': line_coords}
    # Return as GeoJSON Feature
    return {'type': 'Feature', 'geometry': geometry}

def get_request(url):
    headers = {
        'accept': 'application/json'
    }
    response = requests.get(url, headers=headers)
    try:
        return response.json()  # This should return a dictionary if the response is JSON
    except json.JSONDecodeError:
        # If response is not JSON, print the response and return an empty dictionary
        print(f"Failed to decode JSON from response: {response.text}")
        return {}

get_url = 'https://levees.sec.usace.army.mil:443/api-local/system-categories/usace-nonusace'

# Perform a GET request
get_response = get_request(get_url)

# Check if 'USACE' key exists and if its value is a string
if 'USACE' in get_response and isinstance(get_response['USACE'], str):
    # Convert the string representation of the list into an actual list
    usace_system_ids = json.loads(get_response['USACE'])
else:
    print("The 'USACE' key is not present or not a string.")

# Create lists to store valid and invalid system IDs
valid_system_ids = []
invalid_system_ids = []

for system_id in usace_system_ids[:15]:
    print(f"Processing system ID: {system_id}")
    geojson_download_url = f'https://levees.sec.usace.army.mil:443/api-local/geometries/query?type=centerline&systemId={system_id}&format=geo&props=true&coll=true'
    geojson_data = get_request(geojson_download_url)
    gdf = gpd.GeoDataFrame.from_features(geojson_data['features'])
        # Check if the GeoDataFrame is empty
    if gdf.empty:
        print(f"No data found for system ID: {system_id}")
        invalid_system_ids.append(system_id)
        continue  # Skip to the next iteration of the loop
    seg_id = gdf['segmentId'].iloc[0]
    segment_info_url = f'https://levees.sec.usace.army.mil:443/api-local/segments/{seg_id}'
    segment_info = get_request(segment_info_url)
    crs = 'EPSG:4269'
    elevation_data = []

    for i, g in gdf.groupby('segmentId'):
        for i, row in g.iterrows():
            data = process_segment(row, crs)
            elevation_data.append(data)

    elevation_data_full = pd.concat(elevation_data)
    try:
        elevation_data_full = elevation_data_full.drop(['name'], axis=1)
    except:
        print('No name column')
    elevation_data_full = gpd.GeoDataFrame(elevation_data_full, geometry='geometry', crs=crs)

    #Get Measured Profile from NLD
    try:
        profile_data = get_request(f'https://levees.sec.usace.army.mil:443/api-local/system/{system_id}/route')
        geojson_feature = topojson_to_geojson(profile_data)
        geometry = shape(geojson_feature['geometry'])
        # Check if 'z' values are present in the geometry
        # if not any(len(coord) == 3 and coord[2] is not None for coord in geometry.coords):
        #     print(f"No 'z' values found for system ID: {system_id}")
        #     invalid_system_ids.append(system_id)
        #     continue  # Skip to the next iteration of the loop
    
        profile_gdf = gpd.GeoDataFrame([{'geometry': geometry}], crs='EPSG:3857')
        profile_gdf = profile_gdf.to_crs(epsg=4269)
        profile_gdf.to_file(f'/Users/jakegearon/CursorProjects/army_levees/data/NLD_profile_{system_id}.geojson', index=False)
        elevation_data_full.to_file(f'/Users/jakegearon/CursorProjects/army_levees/data/3DEP_{system_id}.geojson', index=False)
        # If the system ID is valid, add it to the valid_system_ids list
        valid_system_ids.append(system_id)
    except:
        print(f"Failed to get profile data for system ID: {system_id}")
        # If the system ID is invalid, add it to the invalid_system_ids list
        invalid_system_ids.append(system_id)

# Save valid and invalid system IDs to files
with open('valid_system_ids.txt', 'w') as f:
    for system_id in valid_system_ids:
        f.write(f"{system_id}\n")

with open('invalid_system_ids.txt', 'w') as f:
    for system_id in invalid_system_ids:
        f.write(f"{system_id}\n")

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

