import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeatures
from cartopy.io.img_tiles import GoogleTiles
import matplotlib.colors as colors
import numpy as np
from shapely.geometry import Point

'''
Updates on _yl:
- Export a excel table (site_coordinates.xlsx) with system_id and lat/lon of the sites
- Save the excel table to "data"
'''

# Directory where the elevation data files are saved
data_dir = '/Users/liyuan/projects/army_levees/data'
output_dir = '/Users/liyuan/projects/army_levees/plots'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

def load_and_process_data():
    gdfs = [gpd.read_parquet(os.path.join(data_dir, f)).to_crs(epsg=4326) 
            for f in os.listdir(data_dir) if f.endswith('.parquet')]
    if not gdfs:
        return gpd.GeoDataFrame()  # return empty GeoDataFrame if no files found
    combined_gdf = pd.concat(gdfs, ignore_index=True)
    
    # Create and export the Excel file
    create_coordinates_excel(combined_gdf)
    
    return combined_gdf

def calculate_mean_differences(plot_df):
    mean_differences = []
    for system_id in plot_df['system_id'].unique():
        df_nld = plot_df[(plot_df['system_id'] == system_id) & (plot_df['source'] == 'nld')]
        df_3dep = plot_df[(plot_df['system_id'] == system_id) & (plot_df['source'] == 'tep')]
        
        if not df_nld.empty and not df_3dep.empty:
            mean_diff = df_3dep['elevation'].mean() - df_nld['elevation'].mean()
            mean_differences.append({'system_id': system_id, 'mean_elevation_diff': mean_diff})
    
    df = pd.DataFrame(mean_differences).sort_values(by='mean_elevation_diff', key=abs, ascending=False)
    
    # Filter out profiles with more than 20m of mean difference
    filtered_df = df[df['mean_elevation_diff'].abs() <= 20]
    
    print(f"Filtered out {len(df) - len(filtered_df)} profiles with >20m mean difference")
    return filtered_df

def plot_mean_elevation_diff_map(mean_differences):
    # Load the original data to get geometries
    gdf = load_and_process_data()
    
    # Merge geometries with mean differences
    mean_diff_gdf = gdf[['system_id', 'geometry']].drop_duplicates().merge(mean_differences, on='system_id')

    terrain_background = GoogleTiles(style='satellite')
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(15, 5))
    
    ax.add_image(terrain_background, 7)
    ax.add_feature(cfeatures.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '50m', 
                                                 facecolor='none', edgecolor='w', linestyle=':', lw=.5))
    ax.set_extent([-125, -66.5, 20, 50], crs=ccrs.PlateCarree())
    
    norm = colors.TwoSlopeNorm(vmin=mean_diff_gdf['mean_elevation_diff'].min(), 
                               vcenter=0, 
                               vmax=mean_diff_gdf['mean_elevation_diff'].max())
    scatter = mean_diff_gdf.plot(ax=ax, column='mean_elevation_diff', cmap='RdYlBu', 
                                 alpha=0.7, norm=norm, markersize=30, edgecolor='k',
                                 legend=False)  # Set legend to False here
    
    sm = plt.cm.ScalarMappable(cmap='RdYlBu', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', label='Mean Elevation Difference (m)')
    
    output_path = os.path.join(output_dir, 'mean_elevation_diff_map.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved mean elevation difference map to {output_path}")
    plt.close()

def plot_mean_elevation_diff(mean_differences):
    fig, ax = plt.subplots(figsize=(10, 6))
    mean_differences.plot(kind='bar', x='system_id', y='mean_elevation_diff', ax=ax, color='skyblue', legend=False)
    ax.set_title('Mean Elevation Difference between NLD and TEP Datasets by System ID')
    ax.set_xlabel('System ID')
    ax.set_ylabel('Mean Elevation Difference (m)')
    plt.xticks(rotation=90, ha="right", fontsize=5)  # Set fontsize to 5 here
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'mean_elevation_diff_barplot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved mean elevation difference bar plot to {output_path}")
    plt.close()

def plot_mean_elevation_diff_histogram(mean_differences):
    fig, ax = plt.subplots(figsize=(10, 6))
    mean_differences['mean_elevation_diff'].hist(ax=ax, bins=30, edgecolor='black')
    ax.set_title('Histogram of Mean Elevation Differences (|diff| <= 20m)')
    ax.set_xlabel('Mean Elevation Difference (m)')
    ax.set_ylabel('Frequency')
    
    # Add vertical line at x=0 for reference
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.7)
    
    # Add text with summary statistics
    stats_text = f"Mean: {mean_differences['mean_elevation_diff'].mean():.2f}m\n"
    stats_text += f"Median: {mean_differences['mean_elevation_diff'].median():.2f}m\n"
    stats_text += f"Std Dev: {mean_differences['mean_elevation_diff'].std():.2f}m"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'mean_elevation_diff_histogram.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved mean elevation difference histogram to {output_path}")
    plt.close()

def create_coordinates_excel(gdf):
    # Create a new DataFrame for the Excel table
    excel_data = []
    for system_id, group in gdf.groupby('system_id'):
        first_row = group.iloc[0]
        if isinstance(first_row['geometry'], Point):
            lat, lon = first_row['geometry'].y, first_row['geometry'].x
        else:
            # If geometry is not a Point, take the first point of the geometry
            lat, lon = first_row['geometry'].coords[0][1], first_row['geometry'].coords[0][0]
        excel_data.append({'system_id': system_id, 'latitude': lat, 'longitude': lon})
    
    excel_df = pd.DataFrame(excel_data)

    # Export the DataFrame to Excel
    excel_output_path = os.path.join(data_dir, 'site_coordinates.xlsx')
    excel_df.to_excel(excel_output_path, index=False)
    print(f"Saved site coordinates to {excel_output_path}")

if __name__ == "__main__":
    plot_df = load_and_process_data()
    print(f"Combined DataFrame shape: {plot_df.shape}")
    print(f"Combined DataFrame columns: {plot_df.columns}")
    
    if plot_df.empty:
        print("WARNING: No data loaded. Check your data directory and file formats.")
        exit()
    
    mean_differences = calculate_mean_differences(plot_df)
    
    print("Plotting mean elevation difference map...")
    plot_mean_elevation_diff_map(mean_differences)
    
    print("Plotting mean elevation difference bar plot...")
    plot_mean_elevation_diff(mean_differences)
    
    print("Plotting mean elevation difference histogram...")
    plot_mean_elevation_diff_histogram(mean_differences)
    
    print("All plotting operations completed.")