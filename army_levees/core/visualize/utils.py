"""Utility functions for visualization."""

from pathlib import Path
import geopandas as gpd
from typing import Optional, List, Set, cast, Dict
import pandas as pd
import matplotlib.pyplot as plt


def get_segment_files(data_dir: str | Path) -> Dict[str, List[Path]]:
    """Get all segment files grouped by system ID.

    Args:
        data_dir: Directory containing segment files

    Returns:
        Dictionary mapping system IDs to lists of segment file paths
    """
    data_dir = Path(data_dir)
    segments: Dict[str, List[Path]] = {}

    # Group segment files by system ID
    for file in sorted(data_dir.glob("levee_*_segment_*.parquet")):
        system_id = file.stem.split("_segment_")[0].replace("levee_", "")
        if system_id not in segments:
            segments[system_id] = []
        segments[system_id].append(file)

    return segments


def get_processed_systems(data_dir: str | Path = "data/segments") -> Set[str]:
    """Get set of system IDs that have been processed."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return set()

    # For segmented data, extract unique system IDs from segment files
    return {
        f.stem.split('_segment_')[0].replace('levee_', '')
        for f in data_dir.glob("levee_*_segment_*.parquet")
    }


def get_system_segments(
    system_id: str,
    segments_dir: str | Path = "data/segments"
) -> List[gpd.GeoDataFrame]:
    """Get all segments for a given system ID.

    Returns list of GeoDataFrames, one for each segment.
    """
    segments_dir = Path(segments_dir)
    segment_files = sorted(segments_dir.glob(f"levee_{system_id}_segment_*.parquet"))

    return [gpd.read_parquet(f) for f in segment_files]


def load_system_data(
    system_id: str,
    data_dir: str | Path = "data/segments",
    raw_data: bool = False
) -> Optional[gpd.GeoDataFrame]:
    """Load data for a system."""
    try:
        if raw_data:
            profile_path = Path("data/processed") / f"levee_{system_id}.parquet"
            if not profile_path.exists():
                return None
            return gpd.read_parquet(profile_path)

        else:
            # Load and combine segments
            data_dir = Path(data_dir)
            segment_files = sorted(data_dir.glob(f"levee_{system_id}_segment_*.parquet"))
            if not segment_files:
                return None

            segments = [gpd.read_parquet(f) for f in segment_files]
            # Use GeoDataFrame.concat to preserve geometry
            return gpd.GeoDataFrame(pd.concat(segments, ignore_index=True))

    except Exception as e:
        print(f"Error loading data for system {system_id}: {str(e)}")
        return None


def get_utm_crs(lon: float, lat: float) -> int:
    """Get UTM zone EPSG code for a given lon/lat."""
    zone_number = int((lon + 180) / 6) + 1

    if lat >= 0:
        # Northern hemisphere
        return 32600 + zone_number
    else:
        # Southern hemisphere
        return 32700 + zone_number


def filter_valid_segments(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Filter out invalid segments from a GeoDataFrame.

    This is a simpler version of the filtering in filter_levees.py,
    meant for quick visualization cleanup.
    """
    # Remove points with zero/nan elevations
    valid_mask = (
        (gdf['elevation'].notna()) &
        (gdf['dep_elevation'].notna()) &
        (gdf['elevation'] != 0) &
        (gdf['dep_elevation'] != 0)
    )

    # Use boolean indexing and cast result to GeoDataFrame
    filtered = gdf[valid_mask]
    return cast(gpd.GeoDataFrame, filtered)


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

    # Create figure with exact size
    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    ax = fig.add_subplot(111)

    # Plot full profile
    profile: gpd.GeoDataFrame = data

    # Plot filled area between lines
    ax.fill_between(
        profile['distance_along_track'],
        profile['elevation'],
        profile['dep_elevation'],
        color='gray',
        alpha=0.2,
        label='Difference'
    )

    # Plot lines on top of fill
    ax.plot(
        profile['distance_along_track'],
        profile['elevation'],
        'b-',
        label='NLD',
        alpha=0.8,
        linewidth=2
    )
    ax.plot(
        profile['distance_along_track'],
        profile['dep_elevation'],
        'r-',
        label='3DEP',
        alpha=0.8,
        linewidth=2
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
    if show:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300)
        plt.close()


def diagnose_elevation_differences(
    system_id: str,
    data_dir: str | Path = "data/segments",
    raw_data: bool = False
) -> None:
    """Print diagnostic information about elevation differences."""
    data = load_system_data(system_id, data_dir=data_dir, raw_data=raw_data)
    if data is None:
        print(f"No data found for system {system_id}")
        return

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
