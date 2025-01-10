"""Functions for visualizing individual levee profiles."""

import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import numpy as np

from .utils import load_system_data


def plot_elevation_profile(
    system_id: str,
    save_dir: str | Path = "plots",
    data_dir: str | Path = "data/segments",
    raw_data: bool = False,
    show: bool = False
) -> None:
    """Plot elevation profile for a levee system."""
    data = load_system_data(system_id, data_dir=data_dir, raw_data=raw_data)
    if data is None:
        print(f"No data found for system {system_id}")
        return
        
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot full profile
    profile: gpd.GeoDataFrame = data
    ax.plot(
        profile['distance_along_track'],
        profile['elevation'],
        'b-',
        label='NLD',
        alpha=0.8
    )
    ax.plot(
        profile['distance_along_track'],
        profile['dep_elevation'],
        'r-',
        label='3DEP',
        alpha=0.8
    )
    # Add points
    ax.scatter(
        profile['distance_along_track'],
        profile['elevation'],
        c='blue',
        s=20,
        alpha=0.5
    )
    ax.scatter(
        profile['distance_along_track'],
        profile['dep_elevation'],
        c='red',
        s=20,
        alpha=0.5
    )
    
    # Customize plot
    ax.set_title(f"System ID: {system_id}")
    ax.set_xlabel("Distance Along Track (m)")
    ax.set_ylabel("Elevation (m)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save plot
    save_path = Path(save_dir) / f"system_{system_id}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def diagnose_elevation_differences(
    system_id: str,
    data_dir: str | Path = "data/processed",
    segments_dir: str | Path = "data/segments",
    use_segments: bool = False
) -> None:
    """Print diagnostic information about elevation differences."""
    
    data = load_system_data(
        system_id, 
        data_dir=data_dir,
        segments_dir=segments_dir,
        use_segments=use_segments
    )
    if data is None:
        print(f"No data found for system {system_id}")
        return
        
    if use_segments:
        # Analyze each segment
        segments: List[gpd.GeoDataFrame] = data
        for i, segment in enumerate(segments):
            print(f"\nSegment {i}:")
            _print_segment_stats(segment)
    else:
        # Analyze full profile
        profile: gpd.GeoDataFrame = data
        _print_segment_stats(profile)


def _print_segment_stats(gdf: gpd.GeoDataFrame) -> None:
    """Print statistics for a segment."""
    diff = gdf['elevation'] - gdf['dep_elevation']
    
    print(f"Points: {len(gdf)}")
    print(f"Length: {gdf['distance_along_track'].max():.1f}m")
    print("\nElevation ranges:")
    print(f"  NLD:  {gdf['elevation'].min():.1f}m to {gdf['elevation'].max():.1f}m")
    print(f"  3DEP: {gdf['dep_elevation'].min():.1f}m to {gdf['dep_elevation'].max():.1f}m")
    print("\nDifferences (NLD - 3DEP):")
    print(f"  Mean: {diff.mean():.1f}m")
    print(f"  Std:  {diff.std():.1f}m")
    print(f"  Min:  {diff.min():.1f}m")
    print(f"  Max:  {diff.max():.1f}m")
