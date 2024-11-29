"""Functions for visualizing levee elevation comparisons."""

import logging
from pathlib import Path
from typing import Optional, List

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import numpy as np
import seaborn as sns
import pandas as pd
from scipy import stats

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def filter_valid_segments(data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Filter out invalid elevation segments.
    
    Only compares sections where:
    1. NLD elevation > 0
    2. 3DEP elevation > 0
    3. No missing data in either dataset
    """
    # Create mask for valid comparison points
    valid_mask = (
        (data['elevation'] > 0) &         # NLD elevation must be positive
        (data['dep_elevation'] > 0) &     # 3DEP elevation must be positive
        (data['elevation'].notna()) &     # No missing NLD data
        (data['dep_elevation'].notna())   # No missing 3DEP data
    )
    
    # Get valid segments and maintain as GeoDataFrame
    valid_data = gpd.GeoDataFrame(
        data[valid_mask].copy(), 
        geometry='geometry',
        crs=data.crs
    )
    
    # Calculate segment lengths
    valid_data['segment_length'] = valid_data['distance_along_track'].diff()
    
    # Log statistics
    total_length = data['distance_along_track'].max()
    valid_length = valid_data['distance_along_track'].max()
    
    logger.info(f"Total levee length: {total_length:.1f}m")
    logger.info(f"Valid length: {valid_length:.1f}m ({100*valid_length/total_length:.1f}%)")
    
    # Log comparison points
    n_original = len(data)
    n_valid = len(valid_data)
    logger.info(f"Valid comparison points: {n_valid} of {n_original} ({100*n_valid/n_original:.1f}%)")
    
    return valid_data

def plot_levee_system(system_id: str, save_dir: Optional[Path] = None) -> Optional[Figure]:
    """Create elevation comparison plots for a levee system."""
    try:
        # Load data
        data = gpd.read_parquet(f"data/processed/levee_{system_id}.parquet")
        
        # Create mask for valid comparison points (non-zero elevations)
        MIN_VALID_ELEV = 1.0  # Minimum valid elevation in meters
        valid_mask = (
            (data['elevation'] > MIN_VALID_ELEV) &     # NLD elevation must be significantly positive
            (data['dep_elevation'] > MIN_VALID_ELEV) & # 3DEP elevation must be significantly positive
            (data['elevation'].notna()) &              # No missing NLD data
            (data['dep_elevation'].notna())            # No missing 3DEP data
        )
        
        # Get valid data and sort by distance
        valid_data = data[valid_mask].copy()
        valid_data = valid_data.sort_values(by='distance_along_track')
        
        # Split into continuous sections
        distance_diff = valid_data['distance_along_track'].diff()
        valid_data['section_break'] = distance_diff.gt(100)  # Break if gap > 100m
        valid_data['section'] = valid_data['section_break'].cumsum()
        
        if len(valid_data) >= 2:
            valid_data['difference'] = valid_data['elevation'] - valid_data['dep_elevation']
            mean_diff = valid_data['difference'].mean()
            std_diff = valid_data['difference'].std()
            rmse = np.sqrt((valid_data['difference']**2).mean())
        else:
            mean_diff = std_diff = rmse = float('nan')
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Plot each continuous section separately
        for section_id, section_data in valid_data.groupby('section'):
            if len(section_data) > 1:  # Only plot sections with multiple points
                ax1.plot(section_data['distance_along_track'], section_data['elevation'], 'b-', label='NLD' if section_id == 0 else "")
                ax1.plot(section_data['distance_along_track'], section_data['dep_elevation'], 'r--', label='3DEP' if section_id == 0 else "")
        
        ax1.set_title(f'Elevation Profiles - System {system_id}')
        ax1.set_xlabel('Distance Along Levee (m)')
        ax1.set_ylabel('Elevation (m)')
        ax1.legend()
        ax1.grid(True)
        
        # Add summary stats to first plot (only for valid sections)
        total_length = data['distance_along_track'].max()
        valid_length = valid_data['distance_along_track'].max() - valid_data['distance_along_track'].min()
        valid_percent = 100 * len(valid_data) / len(data)
        
        stats_text = (
            f'Total Length: {total_length:.0f}m\n'
            f'Valid Points: {len(valid_data)} of {len(data)} ({valid_percent:.1f}%)\n'
            f'Mean Diff: {mean_diff:.2f}m\n'
            f'Std Dev: {std_diff:.2f}m\n'
            f'RMSE: {rmse:.2f}m'
        )
        ax1.text(0.02, 0.98, stats_text,
                transform=ax1.transAxes,
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8))
        
        # Plot 2: Differences histogram (only for valid sections)
        if len(valid_data) >= 2:
            sns.histplot(data=pd.DataFrame({'difference': valid_data['difference']}), 
                        x='difference', bins=50, ax=ax2)
            ax2.axvline(x=0, color='r', linestyle='--', alpha=0.5)
            ax2.set_title('Elevation Differences (NLD - 3DEP) for Valid Sections')
            ax2.set_xlabel('Difference (m)')
            ax2.set_ylabel('Count')
            ax2.grid(True)
        else:
            ax2.text(0.5, 0.5, 'No valid comparison points',
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax2.transAxes)
        
        plt.tight_layout()
        
        # Save if directory provided
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / f'levee_comparison_{system_id}.png', 
                       dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_dir}/levee_comparison_{system_id}.png")
        
        return fig
        
    except Exception as e:
        logger.error(f"Error plotting system {system_id}: {str(e)}")
        return None

def plot_summary(save_dir: Optional[Path] = None) -> Optional[Figure]:
    """Create summary plots for all processed levee systems."""
    try:
        # Get all processed levee files
        processed_dir = Path('data/processed')
        levee_files = list(processed_dir.glob('levee_*.parquet'))
        if not levee_files:
            logger.error("No processed levee files found")
            return None
            
        # Collect statistics for each levee
        stats = []
        MIN_VALID_ELEV = 1.0  # Minimum valid elevation in meters
        
        for file in levee_files:
            system_id = file.stem.replace('levee_', '')
            data = gpd.read_parquet(file)
            
            # Create mask for valid comparison points
            valid_mask = (
                (data['elevation'] > MIN_VALID_ELEV) &     # NLD elevation must be significantly positive
                (data['dep_elevation'] > MIN_VALID_ELEV) & # 3DEP elevation must be significantly positive
                (data['elevation'].notna()) &              # No missing NLD data
                (data['dep_elevation'].notna())            # No missing 3DEP data
            )
            
            # Get valid data and calculate differences
            valid_data = data[valid_mask].copy()
            if len(valid_data) >= 2:
                valid_data['difference'] = valid_data['elevation'] - valid_data['dep_elevation']
                
                # Remove outliers using manual z-score calculation
                mean_diff = valid_data['difference'].mean()
                std_diff = valid_data['difference'].std()
                if std_diff > 0:  # Only filter if there's variation
                    z_scores = np.abs((valid_data['difference'] - mean_diff) / std_diff)
                    valid_data = valid_data[z_scores < 3]  # Keep points within 3 standard deviations
                
                if len(valid_data) >= 2:
                    stats.append({
                        'system_id': system_id,
                        'total_length': data['distance_along_track'].max(),
                        'valid_length': valid_data['distance_along_track'].max() - valid_data['distance_along_track'].min(),
                        'mean_diff': valid_data['difference'].mean(),
                        'median_diff': valid_data['difference'].median(),
                        'std_diff': valid_data['difference'].std(),
                        'rmse': np.sqrt((valid_data['difference']**2).mean()),
                        'skewness': valid_data['difference'].skew(),
                        'kurtosis': valid_data['difference'].kurtosis(),
                        'n_points': len(valid_data),
                        'total_points': len(data),
                        'valid_percent': 100 * len(valid_data) / len(data)
                    })
        
        if not stats:
            logger.error("No valid statistics found")
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(stats)
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(15, 12))
        gs = plt.GridSpec(3, 2)
        
        # Plot 1: Distribution of levee lengths
        ax1 = fig.add_subplot(gs[0, 0])
        sns.histplot(data=df, x='total_length', bins=20, ax=ax1)
        ax1.set_title('Distribution of Levee Lengths')
        ax1.set_xlabel('Length (m)')
        ax1.set_ylabel('Count')
        ax1.grid(True)
        
        # Plot 2: Mean difference vs length
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(df['total_length'], df['mean_diff'])
        ax2.set_title('Mean Elevation Difference vs Length')
        ax2.set_xlabel('Length (m)')
        ax2.set_ylabel('Mean Difference (m)')
        ax2.grid(True)
        
        # Plot 3: CDF of differences
        ax3 = fig.add_subplot(gs[1, 0])
        sorted_diff = np.sort(df['mean_diff'])
        yvals = np.arange(len(sorted_diff)) / float(len(sorted_diff) - 1)
        ax3.plot(sorted_diff, yvals)
        ax3.set_xlabel('Mean Elevation Difference (m)')
        ax3.set_ylabel('Cumulative Probability')
        ax3.set_title('Cumulative Distribution Function of Mean Differences')
        ax3.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        ax3.grid(True)
        
        # Plot 4: Boxplot of positive vs negative differences
        ax4 = fig.add_subplot(gs[1, 1])
        significant_diff = df[df['mean_diff'].abs() >= 0.1]  # Filter for differences >= 0.1m
        negative_diff = significant_diff[significant_diff['mean_diff'] < 0]['mean_diff']
        positive_diff = significant_diff[significant_diff['mean_diff'] > 0]['mean_diff']
        bp = ax4.boxplot([negative_diff, positive_diff], tick_labels=['Negative', 'Positive'], patch_artist=True)
        
        # Color the boxes
        colors = ['lightblue', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
                
        ax4.set_title('Boxplot of Mean Differences (≥0.1m)')
        ax4.set_ylabel('Mean Difference (m)')
        ax4.grid(True)
        
        # Plot 5: Distribution of mean differences with KDE
        ax5 = fig.add_subplot(gs[2, 0])
        sns.histplot(data=df, x='mean_diff', bins=20, kde=True, ax=ax5)
        ax5.set_title('Distribution of Mean Differences')
        ax5.set_xlabel('Mean Difference (m)')
        ax5.set_ylabel('Count')
        ax5.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        ax5.grid(True)
        
        # Plot 6: Valid data coverage
        ax6 = fig.add_subplot(gs[2, 1])
        sns.histplot(data=df, x='valid_percent', bins=20, ax=ax6)
        ax6.set_title('Distribution of Valid Data Coverage')
        ax6.set_xlabel('Valid Points (%)')
        ax6.set_ylabel('Count')
        ax6.grid(True)
        
        # Add overall statistics as text
        stats_text = (
            f'Total Systems: {len(df)}\n'
            f'Mean Length: {df["total_length"].mean():.0f}m\n'
            f'Mean Valid Coverage: {df["valid_percent"].mean():.1f}%\n'
            f'Mean Difference: {df["mean_diff"].mean():.2f}m ± {df["mean_diff"].std():.2f}m\n'
            f'Median Difference: {df["mean_diff"].median():.2f}m\n'
            f'Mean RMSE: {df["rmse"].mean():.2f}m\n'
            f'Skewness: {df["mean_diff"].skew():.2f}\n'
            f'Kurtosis: {df["mean_diff"].kurtosis():.2f}'
        )
        fig.text(0.02, 0.98, stats_text, fontsize=10, va='top')
        
        plt.tight_layout()
        
        # Save if directory provided
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / 'levee_summary.png', dpi=300, bbox_inches='tight')
            logger.info(f"\nSaved summary plot to {save_dir}/levee_summary.png")
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating summary plots: {str(e)}")
        return None

if __name__ == '__main__':
    import argparse
    import random
    from pathlib import Path
    
    parser = argparse.ArgumentParser(
        description='Create elevation comparison plots for levee systems'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('system_id', nargs='?', help='USACE system ID to plot')
    group.add_argument('-r', '--random', action='store_true', help='Plot a random levee system')
    group.add_argument('-s', '--summary', action='store_true', help='Create summary plots for all processed levees')
    parser.add_argument('--save_dir', type=str, default='plots',
                      help='Directory to save plots (default: plots)')
    args = parser.parse_args()
    
    if args.summary:
        plot_summary(save_dir=Path(args.save_dir))
    elif args.random:
        # Get list of processed levee files
        processed_dir = Path('data/processed')
        levee_files = list(processed_dir.glob('levee_*.parquet'))
        if not levee_files:
            print("No processed levee files found.")
            exit(1)
            
        # Pick a random file and extract system ID
        random_file = random.choice(levee_files)
        system_id = random_file.stem.replace('levee_', '')
        print(f"Randomly selected system ID: {system_id}")
        plot_levee_system(system_id, save_dir=Path(args.save_dir))
    else:
        plot_levee_system(args.system_id, save_dir=Path(args.save_dir))