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

def topojson_to_geojson(topojson):
    # Check if the geometry type is 'FeatureCollection'
    if topojson['type'] == 'FeatureCollection':
        # Process each feature in the collection
        features = []
        for feature in topojson['features']:
            geometry = shape(feature['geometry'])
            properties = feature['properties']
            features.append({'geometry': geometry, 'properties': properties})
        # Create a GeoDataFrame from the features
        gdf = gpd.GeoDataFrame(features)
        return gdf
    else:
        # Handle other types (e.g., LineString) as before
        line_data = topojson['geometry']['arcs'][0]
        line_coords = [(x, y) for x, y, z, m in line_data]
        geometry = LineString(line_coords)
        df = pd.DataFrame(line_data, columns=['x', 'y', 'z', 'm'])
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y))
        return gdf

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
    
    # Get Measured Profile from NLD first
    try:
        profile_data = get_request(f'https://levees.sec.usace.army.mil:443/api-local/system/{system_id}/route')
        geojson_feature = topojson_to_geojson(profile_data)
        geometry = shape(geojson_feature['geometry'])
        
        # Check if 'z' values are present in the geometry
        if not any(len(coord) == 3 and coord[2] is not None for coord in geometry.coords):
            print(f"No 'z' values found for system ID: {system_id}")
            invalid_system_ids.append(system_id)
            continue  # Skip to the next iteration of the loop
        
        profile_gdf = gpd.GeoDataFrame([{'geometry': geometry}], crs='EPSG:3857')
        profile_gdf = profile_gdf.to_crs(epsg=4269)
        
        # If the profile is valid, proceed to get the 3DEP data
        geojson_download_url = f'https://levees.sec.usace.army.mil:443/api-local/geometries/query?type=centerline&systemId={system_id}&format=geo&props=true&coll=true'
        topo_data = get_request(geojson_download_url)
        gdf = topojson_to_geojson(topo_data)
        print(gdf)
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

        # Save the profile data
        profile_gdf.to_file(f'/Users/jakegearon/CursorProjects/army_levees/data/NLD_profile_{system_id}.geojson', index=False)
        elevation_data_full.to_file(f'/Users/jakegearon/CursorProjects/army_levees/data/3DEP_{system_id}.geojson', index=False)
        # If the system ID is valid, add it to the valid_system_ids list
        valid_system_ids.append(system_id)
        
    except Exception as e:
        print(f"Failed to get profile data for system ID: {system_id}: {e}")
        invalid_system_ids.append(system_id)

# Save valid and invalid system IDs to files
with open('valid_system_ids.txt', 'w') as f:
    for system_id in valid_system_ids:
        f.write(f"{system_id}\n")

with open('invalid_system_ids.txt', 'w') as f:
    for system_id in invalid_system_ids:
        f.write(f"{system_id}\n")
