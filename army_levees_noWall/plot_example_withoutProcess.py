'''script that plot the elevation profiles without any processing of the data'''

import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import os

def plot_elevation_profiles(gdfs, system_id, output_dir):
    """
    Plot elevation profiles from both NLD and DEM data.
    
    Args:
        gdfs: List of GeoDataFrames containing both NLD and DEM data
        system_id: System ID for the title
        output_dir: Directory to save the plot
    """
    # Filter data for the specific system_id
    profile_gdf = None
    elevation_data_full = None
    
    for gdf in gdfs:
        if 'system_id' in gdf.columns and 'source' in gdf.columns:
            system_data = gdf[gdf['system_id'] == system_id]
            if not system_data.empty:
                if system_data['source'].iloc[0] == 'nld':
                    profile_gdf = system_data
                elif system_data['source'].iloc[0] == 'tep':
                    elevation_data_full = system_data
    
    if profile_gdf is None or elevation_data_full is None:
        print(f"Could not find data for system ID: {system_id}")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Plot both profiles
    plt.plot(profile_gdf['distance_along_track'], 
             profile_gdf['elevation'], 
             'b-', label='NLD Profile', linewidth=2)
    
    plt.plot(elevation_data_full['distance_along_track'], 
             elevation_data_full['elevation'], 
             'r--', label='DEM Profile', linewidth=2)
    
    # Add labels and title
    plt.xlabel('Distance Along Track (m)')
    plt.ylabel('Elevation (m)')
    plt.title(f'Elevation Profiles Comparison - System ID: {system_id}')
    plt.grid(True)
    plt.legend()
    
    # Save the plot
    plot_path = os.path.join(output_dir, f'elevation_profiles_{system_id}_fromData.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Elevation profiles plot saved to: {plot_path}")


if __name__ == "__main__":
    # read data in /Users/liyuan/projects/army_levees/data_no_floodwall
    data_dir = '/Users/liyuan/projects/army_levees/data_no_floodwall'
    
    # read all the parquet files in the data_dir
    gdfs = [gpd.read_parquet(os.path.join(data_dir, f)) 
            for f in os.listdir(data_dir) if f.endswith('.parquet')]
    
    # plot the elevation profile for system_id = 5705000030
    system_id = 5705000030
    
    # Create output directory if it doesn't exist
    output_dir = '/Users/liyuan/Documents/Files/Postdoc - project/Artificial Levee'
    os.makedirs(output_dir, exist_ok=True)
    
    # plot the elevation profile for the system_id
    plot_elevation_profiles(gdfs, system_id, output_dir)

