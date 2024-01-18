import geopandas as gpd
import pandas as pd
import py3dep 
from shapely.geometry import MultiLineString, LineString


def load_geojson(geojson_file):
    gdf = gpd.read_file(geojson_file)
    return gdf, gdf.crs

def load_shapefile(shapefile):
    gdf = gpd.read_file(shapefile)
    return gdf, gdf.crs

def process_segment(row, crs):
    try:
        # Create a DataFrame to hold the elevation data
        elevation_data_list = []
        if isinstance(row.geometry, MultiLineString):
            for line in row.geometry.geoms:
                elevation = py3dep.elevation_profile(line, crs=crs, spacing=10, dem_res=10)
                elevation_df = elevation.to_dataframe(name='elevation').reset_index()        
                elevation_data_list.append(elevation_df)
        else:
            line = row.geometry
            elevation = py3dep.elevation_profile(line, crs=crs, spacing=10, dem_res=10)
            elevation_df = elevation.to_dataframe(name='elevation').reset_index()        
            elevation_data_list.append(elevation_df)
        
        # Concatenate all elevation data
        elevation_data_full = pd.concat(elevation_data_list)
        elevation_gdf = gpd.GeoDataFrame(elevation_data_full, geometry=gpd.points_from_xy(elevation_data_full.x, elevation_data_full.y))
        
        # Add the 'name' and other row information to the GeoDataFrame
        for column in row.index:
            elevation_gdf[column] = row[column]
        
        return elevation_gdf
    except ValueError as e:
        print(f"Caught an error: {e}")

def main():
    gdf, crs = load_geojson('/Users/jakegearon/Downloads/levees-geojson 2/Centerline.geojson')
    elevation_data = []

    for segment_id, segment_group in gdf.groupby('segmentId'):
        for _, row in segment_group.iterrows():
            print(row.geometry)
            data = process_segment(row, crs)
            elevation_data.append(data)

    # Concatenate all the elevation data into one GeoDataFrame
    elevation_data_full = pd.concat(elevation_data)
    elevation_data_full = gpd.GeoDataFrame(elevation_data_full, geometry='geometry', crs=crs)
    
    # Save the final GeoDataFrame to a file
    elevation_data_full.to_file('/Users/jakegearon/Downloads/levees-geojson 2/Centerline_elev.geojson', driver='GeoJSON', index=False)

main()
