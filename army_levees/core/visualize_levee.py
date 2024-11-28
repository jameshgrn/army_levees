"""Functions for visualizing levee elevation comparisons.

This module provides visualization tools for comparing NLD and 3DEP elevations:
1. Elevation profile plots showing both NLD and 3DEP
2. Histogram of elevation differences

Typical usage:
    >>> from army_levees import plot_levee_system
    >>> plot_levee_system("5205000591", save_dir="plots")

The plots show:
    1. Top panel: Elevation profiles
       - Blue line: NLD elevation
       - Red line: 3DEP elevation
       - X-axis: Distance along levee
       - Y-axis: Elevation (meters)
    
    2. Bottom panel: Differences
       - Histogram of NLD - 3DEP differences
       - X-axis: Difference (meters)
       - Y-axis: Count
"""

import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def plot_levee_system(system_id: str, save_dir: Optional[Path] = None) -> Optional[Figure]:
    """Create elevation comparison plots for a levee system.
    
    Args:
        system_id: USACE system ID (e.g. "5205000591")
        save_dir: Directory to save plots (optional)
    
    Returns:
        matplotlib Figure object with two subplots:
            1. Elevation profiles (NLD vs 3DEP)
            2. Histogram of differences
        None if data loading fails
    
    Note:
        If save_dir is provided, saves plot as:
            save_dir/levee_comparison_SYSTEMID.pdf
    """
    try:
        # Load data
        data = gpd.read_parquet(f"data/processed/levee_{system_id}.parquet")
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Elevation profiles
        ax1.plot(data['distance_along_track'], data['nld_elevation'], 'b-', label='NLD')
        ax1.plot(data['distance_along_track'], data['dep_elevation'], 'r--', label='3DEP')
        ax1.set_title(f'Elevation Profiles - System {system_id}')
        ax1.set_xlabel('Distance Along Levee (m)')
        ax1.set_ylabel('Elevation (m)')
        ax1.legend()
        ax1.grid(True)
        
        # Add summary stats to first plot
        mean_diff = data['difference'].mean()
        std_diff = data['difference'].std()
        rmse = np.sqrt((data['difference']**2).mean())
        stats_text = (
            f'Mean Diff: {mean_diff:.2f}m\n'
            f'Std Dev: {std_diff:.2f}m\n'
            f'RMSE: {rmse:.2f}m'
        )
        ax1.text(0.02, 0.98, stats_text,
                transform=ax1.transAxes,
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8))
        
        # Plot 2: Differences histogram
        sns.histplot(data['difference'], bins=50, ax=ax2)
        ax2.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        ax2.set_title('Elevation Differences (NLD - 3DEP)')
        ax2.set_xlabel('Difference (m)')
        ax2.set_ylabel('Count')
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save if directory provided
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / f'levee_comparison_{system_id}.pdf')
            logger.info(f"Saved plot to {save_dir}/levee_comparison_{system_id}.pdf")
        
        return fig
        
    except Exception as e:
        logger.error(f"Error plotting system {system_id}: {str(e)}")
        return None

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Create elevation comparison plots for levee systems',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__  # Use module docstring as extended help
    )
    parser.add_argument('system_id', help='USACE system ID to plot')
    parser.add_argument('--save_dir', type=str, default='plots',
                      help='Directory to save plots (default: plots)')
    args = parser.parse_args()
    
    plot_levee_system(args.system_id, save_dir=Path(args.save_dir))