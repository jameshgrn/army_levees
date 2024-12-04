"""Functions for creating summary visualizations of levee data."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import geopandas as gpd
from tqdm import tqdm

from .utils import load_system_data, get_processed_systems


def plot_summary(
    save_dir: str | Path = "plots",
    data_dir: str | Path = "data/segments",
    raw_data: bool = False
) -> None:
    """Create summary plots for all processed levees."""
    
    # Get all system IDs
    system_ids = get_processed_systems(data_dir=data_dir)
        
    if not system_ids:
        print("No levee data found")
        return
        
    # Collect statistics
    stats: Dict[str, List[float]] = {
        "mean_diff": [],
        "std_diff": [],
        "max_diff": [],
        "min_diff": [],
        "length": []
    }
    
    # Process each system
    for system_id in tqdm(system_ids, desc="Processing systems"):
        data = load_system_data(system_id, data_dir=data_dir, raw_data=raw_data)
        if data is None:
            continue
            
        # Process full profile
        profile: gpd.GeoDataFrame = data
        _collect_segment_stats(profile, stats)
    
    # Create plots
    _create_summary_plots(stats, save_dir)


def _collect_segment_stats(gdf: gpd.GeoDataFrame, stats: Dict[str, List[float]]) -> None:
    """Collect statistics from a segment."""
    diff = gdf['elevation'] - gdf['dep_elevation']
    
    stats["mean_diff"].append(diff.mean())
    stats["std_diff"].append(diff.std())
    stats["max_diff"].append(diff.max())
    stats["min_diff"].append(diff.min())
    stats["length"].append(gdf['distance_along_track'].max())


def _create_summary_plots(stats: Dict[str, List[float]], save_dir: Path | str) -> None:
    """Create and save summary plots."""
    save_dir = Path(save_dir)
    
    # Elevation difference histogram
    plt.figure(figsize=(10, 6))
    plt.hist(stats["mean_diff"], bins=50, alpha=0.7)
    plt.xlabel("Mean Elevation Difference (m)")
    plt.ylabel("Count")
    plt.title("Distribution of Mean Elevation Differences")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / "mean_differences_hist.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Standard deviation vs mean difference
    plt.figure(figsize=(10, 6))
    plt.scatter(stats["mean_diff"], stats["std_diff"], alpha=0.5)
    plt.xlabel("Mean Difference (m)")
    plt.ylabel("Standard Deviation (m)")
    plt.title("Standard Deviation vs Mean Difference")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / "std_vs_mean.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Length histogram
    plt.figure(figsize=(10, 6))
    plt.hist(stats["length"], bins=50, alpha=0.7)
    plt.xlabel("Length (m)")
    plt.ylabel("Count")
    plt.title("Distribution of Profile Lengths")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / "length_hist.png", dpi=300, bbox_inches='tight')
    plt.close()
