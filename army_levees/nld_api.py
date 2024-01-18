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

from shapely.geometry import LineString, MultiLineString
import geopandas as gpd

def topojson_to_geojson(data):
    if 'type' in data:
        if data['type'] == 'Topology':
            arcs = data['arcs']
            objects = data['objects']
            
            for key in objects:
                geom = objects[key]
                if geom['type'] == 'LineString':
                    try:
                        arc_indices = geom['arcs'] if isinstance(geom['arcs'], list) else [geom['arcs']]
                        line_coords = [arcs[index] for index in arc_indices]
                        # Slice each coordinate tuple to the first two or three elements
                        flat_coords = [coord[:2] for sublist in line_coords for coord in sublist]  # 2D coordinates
                        # flat_coords = [coord[:3] for sublist in line_coords for coord in sublist]  # 3D coordinates, if elevation is needed
                        geometry = LineString(flat_coords)
                    except Exception as e:
                        print(f"Error processing LineString for key {key}: {e}")
                        return None
                elif geom['type'] == 'MultiLineString':
                    try:
                        multi_line_coords = [[arcs[index] for index in part] for part in geom['arcs']]
                        # Slice each coordinate tuple to the first two or three elements
                        flat_multi_coords = [[coord[:2] for coord in part] for part in multi_line_coords]  # 2D coordinates
                        # flat_multi_coords = [[coord[:3] for coord in part] for part in multi_line_coords]  # 3D coordinates, if elevation is needed
                        geometry = MultiLineString([LineString(part) for part in flat_multi_coords])
                    except Exception as e:
                        print(f"Error processing MultiLineString for key {key}: {e}")
                        return None
                else:
                    print(f"Unsupported geometry type for key {key}: {geom['type']}")
                    return None
                
                # Create a GeoDataFrame
                gdf = gpd.GeoDataFrame([{'geometry': geometry}])
                return gdf
        elif data['type'] == 'FeatureCollection':
            features = data['features']
            geometries = [shape(feature['geometry']) for feature in features]
            gdf = gpd.GeoDataFrame(geometry=geometries)
            return gdf
        else:
            print(f"Unsupported data type: {data['type']}")
            return None
    else:
        print("No 'type' in data")
        return None

def get_request(url):
    headers = {
        'accept': 'application/json'
    }
    try:
        response = requests.get(url, headers=headers)
        #print(f"Full response: {response.text}")  # Print the full response
        response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code
        try:
            return response.json()  # This should return a dictionary if the response is JSON
        except json.JSONDecodeError:
            # If response is not JSON, print the response and return None
            print(f"Failed to decode JSON from response: {response.text}")
            return None
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")  # Print the HTTP error
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred: {conn_err}")  # Print the connection error
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error occurred: {timeout_err}")  # Print the timeout error
    except requests.exceptions.RequestException as req_err:
        print(f"Request exception occurred: {req_err}")  # Print any other request exception
    return None

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

for system_id in usace_system_ids[5:8]:
    print(f"Processing system ID: {system_id}")
    
    # Get Measured Profile from NLD first
    try:
        print(f"Attempting to get profile data for system ID: {system_id}")  # Debug print
        profile_data = get_request(f'https://levees.sec.usace.army.mil:443/api-local/system/{system_id}/route')
        print(f"Profile data received: {profile_data}")  # Debug print
        if profile_data is None:
            print(f"No profile data found for system ID: {system_id}")
            invalid_system_ids.append(system_id)
            continue  # Skip to the next iteration of the loop


        print("Attempting to convert profile data to GeoJSON")  # Debug print
        # Pass the 'geometry' key of the profile_data
        geojson_feature = topojson_to_geojson(profile_data['geometry'])

        # Check if the GeoDataFrame is not None and has a geometry column
        if geojson_feature is not None and 'geometry' in geojson_feature:
            print(f"GeoJSON feature: {geojson_feature}")  # Debug print
            # Convert the GeoDataFrame to a GeoJSON FeatureCollection
            geojson_feature_collection = geojson_feature.__geo_interface__
            print(f"GeoJSON FeatureCollection: {geojson_feature_collection}")

            # Now you can use the geojson_feature_collection as a dictionary
            # representing a GeoJSON FeatureCollection
            # For example, if you need to access the geometry of the first feature:
            if geojson_feature_collection:
                geometry = geojson_feature_collection['features'][0]['geometry']
                # Assuming 'geometry' is a dictionary representing a GeoJSON geometry
                from shapely.geometry import shape

                # Convert the GeoJSON geometry to a Shapely geometry object
                shapely_geometry = shape(geometry)

                # Now create the GeoDataFrame with the Shapely geometry object
                profile_gdf = gpd.GeoDataFrame([{'geometry': shapely_geometry}], crs='EPSG:3857')

                # Convert the CRS to EPSG:4269
                profile_gdf = profile_gdf.to_crs(epsg=4269)
        
        # If the profile is valid, proceed to get the 3DEP data
        print("Attempting to get 3DEP data")  # Debug print
        geojson_download_url = f'https://levees.sec.usace.army.mil:443/api-local/geometries/query?type=centerline&systemId={system_id}&format=geo&props=true&coll=true'
        topo_data = get_request(geojson_download_url)
        print(f"Topo data received: {topo_data}")  # Debug print
        gdf = topojson_to_geojson(topo_data)
        print(gdf)
        # Check if the GeoDataFrame is empty
        if gdf.empty:
            print(f"No data found for system ID: {system_id}")
            invalid_system_ids.append(system_id)
            continue  # Skip to the next iteration of the loop
        
        # seg_id = gdf['segmentId'].iloc[0]
        # segment_info_url = f'https://levees.sec.usace.army.mil:443/api-local/segments/{seg_id}'
        # segment_info = get_request(segment_info_url)
        # crs = 'EPSG:4269'
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
