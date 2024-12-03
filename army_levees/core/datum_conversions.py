"""Tools for analyzing and converting between different vertical datums using VERTCON."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import folium
from tqdm import tqdm
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
import bilinear  # We'll create this from vertcon-web's bilinear.py
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# VERTCON grid parameters
GRID_SPACING = 3/60  # 3 arc-minutes
XMIN, YMIN = -130.0, 24.0
XMAX, YMAX = -65.0, 50.0
NROWS = int((YMAX - YMIN) / GRID_SPACING)
NCOLS = int((XMAX - XMIN) / GRID_SPACING)

def load_vertcon_grid(grid_path: str = "vertcon_88-29.tif") -> np.ndarray:
    """Load the VERTCON grid from TIF file."""
    with rasterio.open(grid_path) as src:
        return src.read(1)  # Read first band

def interpolate_vertcon(lon: float, lat: float, grid: np.ndarray) -> Optional[float]:
    """
    Interpolate VERTCON offset for a given point using bilinear interpolation.
    Returns difference in meters (NAVD88 - NGVD29).
    """
    # Convert coordinates to grid indices
    x = (lon - XMIN) / GRID_SPACING
    y = (lat - YMIN) / GRID_SPACING
    
    # Check bounds with some padding
    if not (1 <= x <= NCOLS-2 and 1 <= y <= NROWS-2):
        return None
        
    # Get surrounding grid points
    x0, x1 = int(np.floor(x)), int(np.ceil(x))
    y0, y1 = int(np.floor(y)), int(np.ceil(y))
    
    # Get values at grid points and check for invalid data
    q11 = grid[y0, x0]
    q12 = grid[y0, x1]
    q21 = grid[y1, x0]
    q22 = grid[y1, x1]
    
    # VERTCON uses large values (like 9999) to indicate invalid/missing data
    if any(abs(v) > 1000 for v in [q11, q12, q21, q22]):
        return None
    
    # Convert from millimeters to meters
    q11, q12, q21, q22 = [v/1000.0 for v in [q11, q12, q21, q22]]
    
    # Perform bilinear interpolation
    return bilinear.interpolate(x-x0, y-y0, q11, q12, q21, q22)

def analyze_system_vertcon(
    gdf: gpd.GeoDataFrame,
    system_id: str,
    vertcon_grid: np.ndarray,
    metadata: Optional[Dict] = None,
    plot: bool = True
) -> Optional[Dict]:
    """Analyze a single system's elevation differences using VERTCON."""
    # Convert to geographic coordinates
    gdf_4326 = gdf.to_crs(4326)
    
    # Get VERTCON offsets for each point
    offsets = []
    points_with_offsets = []  # Track which points have valid offsets
    for idx, row in gdf_4326.iterrows():
        offset = interpolate_vertcon(row.geometry.x, row.geometry.y, vertcon_grid)
        if offset is not None:
            offsets.append(offset)
            points_with_offsets.append(idx)
    
    # Require at least 50% of points to have valid offsets
    if len(offsets) < len(gdf) * 0.5:
        logger.warning(f"Insufficient valid VERTCON offsets for system {system_id} "
                      f"({len(offsets)}/{len(gdf)} points)")
        return None
        
    offsets = np.array(offsets)
    
    # Get elevations only for points with valid offsets
    valid_gdf = gdf.loc[points_with_offsets]
    nld_elevs = valid_gdf['elevation'].to_numpy()
    dep_elevs = valid_gdf['dep_elevation'].to_numpy()
    distances = valid_gdf['distance_along_track'].to_numpy()
    
    # Calculate NGVD29->NAVD88 transformed NLD elevations
    nld_transformed = nld_elevs + offsets
    
    # Check for unreasonable values
    differences = nld_transformed - dep_elevs
    if np.any(np.abs(differences) > 100):  # More than 100m difference is suspicious
        logger.warning(f"Unreasonable elevation differences found in system {system_id}")
        return None
    
    # Create visualization
    if plot:
        create_elevation_plot(
            system_id=system_id,
            distances=distances,
            nld_elevs=nld_elevs,
            dep_elevs=dep_elevs,
            nld_transformed=nld_transformed,
            offsets=offsets,
            differences=differences
        )
    
    return {
        'system_id': system_id,
        'vertcon_stats': {
            'mean_offset': float(np.mean(offsets)),
            'std_offset': float(np.std(offsets)),
            'min_offset': float(np.min(offsets)),
            'max_offset': float(np.max(offsets)),
            'valid_points': len(offsets)
        },
        'difference_stats': {
            'mean_diff': float(np.mean(differences)),
            'std_diff': float(np.std(differences)),
            'min_diff': float(np.min(differences)),
            'max_diff': float(np.max(differences))
        },
        'location': {
            'latitude': float(gdf_4326.geometry.y.mean()),
            'longitude': float(gdf_4326.geometry.x.mean())
        },
        'metadata': metadata or {}
    }

def create_elevation_plot(
    system_id: str,
    distances: np.ndarray,
    nld_elevs: np.ndarray,
    dep_elevs: np.ndarray,
    nld_transformed: np.ndarray,
    offsets: np.ndarray,
    differences: np.ndarray
) -> None:
    """Create a multi-panel plot showing elevations and differences."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), height_ratios=[3, 1, 1])
    
    # Plot elevations
    ax1.plot(distances, nld_elevs, 'b-', label='NLD (Original)', alpha=0.7)
    ax1.plot(distances, dep_elevs, 'r-', label='3DEP', alpha=0.7)
    ax1.plot(distances, nld_transformed, 'g--', label='NLD (NAVD88)', alpha=0.7)
    ax1.grid(True)
    ax1.legend()
    ax1.set_title(f'Elevation Profiles - System {system_id}')
    ax1.set_ylabel('Elevation (m)')
    
    # Plot VERTCON offsets
    ax2.plot(distances, offsets, 'k-')
    ax2.axhline(y=0, color='r', linestyle=':')
    ax2.grid(True)
    ax2.set_title('VERTCON Offsets (NAVD88 - NGVD29)')
    ax2.set_ylabel('Offset (m)')
    
    # Plot residual differences
    ax3.plot(distances, differences, 'k-')
    ax3.axhline(y=0, color='r', linestyle=':')
    ax3.fill_between(
        distances,
        -0.5, 0.5,
        alpha=0.2,
        color='g',
        label='±0.5m'
    )
    ax3.grid(True)
    ax3.set_title('Residual Differences (NLD[NAVD88] - 3DEP)')
    ax3.set_xlabel('Distance Along Levee (m)')
    ax3.set_ylabel('Difference (m)')
    ax3.legend()
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("investigation/vertcon_plots")
    output_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_dir / f"system_{system_id}_vertcon.png")
    plt.close()

def analyze_all_systems(vertcon_grid_path: str) -> pd.DataFrame:
    """Analyze all available systems using VERTCON."""
    # Load VERTCON grid
    grid = load_vertcon_grid(vertcon_grid_path)
    
    # Get all processed files
    data_dir = Path("data/processed")
    parquet_files = list(data_dir.glob("levee_*.parquet"))
    
    results = []
    for file_path in tqdm(parquet_files, desc="Analyzing systems"):
        try:
            gdf = gpd.read_parquet(file_path)
            system_id = file_path.stem.split("_")[1]
            
            analysis = analyze_system_vertcon(gdf, system_id, grid)
            if analysis:
                results.append(analysis)
                
        except Exception as e:
            logger.error(f"Error analyzing system {file_path}: {e}")
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    output_dir = Path("investigation")
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / "vertcon_analysis.csv", index=False)
    
    # Create visualization
    create_vertcon_map(df)
    
    return df

def create_vertcon_map(df: pd.DataFrame) -> None:
    """Create an interactive map showing VERTCON analysis results."""
    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
    
    def get_color(diff: float) -> str:
        """Get color based on difference magnitude after VERTCON correction."""
        if abs(diff) < 0.5:
            return 'green'
        elif abs(diff) < 2.0:
            return 'yellow'
        else:
            return 'red'
    
    for _, row in df.iterrows():
        loc = row['location']
        stats = row['difference_stats']
        vertcon = row['vertcon_stats']
        
        popup_content = f"""
        <strong>System {row['system_id']}</strong><br>
        Location: {loc['latitude']:.2f}°N, {loc['longitude']:.2f}°E<br>
        <br>
        <strong>VERTCON Offset:</strong><br>
        Mean: {vertcon['mean_offset']:.2f}m<br>
        Std Dev: {vertcon['std_offset']:.2f}m<br>
        Range: {vertcon['min_offset']:.2f}m to {vertcon['max_offset']:.2f}m<br>
        <br>
        <strong>After Correction:</strong><br>
        Mean Diff: {stats['mean_diff']:.2f}m<br>
        Std Dev: {stats['std_diff']:.2f}m<br>
        Range: {stats['min_diff']:.2f}m to {stats['max_diff']:.2f}m
        """
        
        folium.CircleMarker(
            location=[loc['latitude'], loc['longitude']],
            radius=8,
            color=get_color(stats['mean_diff']),
            fill=True,
            popup=popup_content
        ).add_to(m)
    
    # Add legend
    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; right: 50px; 
                border:2px solid grey; z-index:9999; 
                background-color:white;
                padding: 10px;">
        <p><strong>Residual Differences</strong><br>
        <small>After VERTCON correction</small></p>
        <p><span style="color:green;">●</span> < 0.5m</p>
        <p><span style="color:yellow;">●</span> 0.5m - 2.0m</p>
        <p><span style="color:red;">●</span> > 2.0m</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    m.save("investigation/vertcon_analysis_map.html")

if __name__ == "__main__":
    # Update this path to where you downloaded the VERTCON grid
    VERTCON_GRID = "vertcon_88-29.tif"
    
    df = analyze_all_systems(VERTCON_GRID)
    
    # Print summary statistics
    print("\nVERTCON Analysis Summary:")
    print(f"Total systems analyzed: {len(df)}")
    
    # Calculate overall statistics
    vertcon_means = df['vertcon_stats'].apply(lambda x: x['mean_offset'])
    diff_means = df['difference_stats'].apply(lambda x: x['mean_diff'])
    
    print("\nVERTCON Offsets:")
    print(f"Mean: {vertcon_means.mean():.2f}m ± {vertcon_means.std():.2f}m")
    print(f"Range: {vertcon_means.min():.2f}m to {vertcon_means.max():.2f}m")
    
    print("\nResidual Differences:")
    print(f"Mean: {diff_means.mean():.2f}m ± {diff_means.std():.2f}m")
    print(f"Range: {diff_means.min():.2f}m to {diff_means.max():.2f}m") 