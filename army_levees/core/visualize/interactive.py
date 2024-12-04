"""Functions for creating interactive visualizations."""

import folium
from pathlib import Path
import geopandas as gpd
from typing import Optional, List

from .utils import load_system_data, get_processed_systems


def create_summary_map(
    save_path: str | Path = "plots/levee_summary_map.html",
    data_dir: str | Path = "data/segments",
    raw_data: bool = False
) -> Optional[folium.Map]:
    """Create interactive map showing levee locations and statistics."""
    
    # Get all system IDs
    system_ids = get_processed_systems(data_dir=data_dir)
        
    if not system_ids:
        print("No levee data found")
        return None
        
    # Initialize map
    m = folium.Map()
    bounds = []
    
    # Add each system
    for system_id in system_ids:
        data = load_system_data(
            system_id, 
            data_dir=data_dir,
            raw_data=raw_data
        )
        if data is None:
            continue
            
        # Add full profile
        profile: gpd.GeoDataFrame = data
        _add_segment_to_map(profile, system_id, None, m, bounds)
    
    # Fit map to bounds
    if bounds:
        m.fit_bounds(bounds)
    
    # Save map
    m.save(str(save_path))
    return m


def _add_segment_to_map(
    gdf: gpd.GeoDataFrame, 
    system_id: str, 
    segment_idx: Optional[int],
    m: folium.Map,
    bounds: list
) -> None:
    """Add a segment to the map."""
    # Convert to WGS84 for mapping
    gdf_4326 = gdf.to_crs(4326)
    
    # Calculate statistics
    diff = gdf['elevation'] - gdf['dep_elevation']
    mean_diff = diff.mean()
    std_diff = diff.std()
    
    # Create popup content
    popup_content = [
        f"System ID: {system_id}",
        f"{'Segment: ' + str(segment_idx) if segment_idx is not None else 'Full Profile'}",
        f"Points: {len(gdf)}",
        f"Length: {gdf['distance_along_track'].max():.1f}m",
        f"Mean Difference: {mean_diff:.1f}m",
        f"Std Difference: {std_diff:.1f}m"
    ]
    
    # Add line to map
    folium.PolyLine(
        locations=[[p.y, p.x] for p in gdf_4326.geometry],
        popup=folium.Popup("<br>".join(popup_content)),
        color='red' if abs(mean_diff) > 5 else 'blue',
        weight=2,
        opacity=0.8
    ).add_to(m)
    
    # Update bounds
    bounds.extend([
        [gdf_4326.geometry.y.min(), gdf_4326.geometry.x.min()],
        [gdf_4326.geometry.y.max(), gdf_4326.geometry.x.max()]
    ])
