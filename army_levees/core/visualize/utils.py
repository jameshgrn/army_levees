"""Utility functions for visualization."""

from pathlib import Path
import geopandas as gpd
from typing import Optional, List, Set
import pandas as pd


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
    
    return gdf[valid_mask].copy()
