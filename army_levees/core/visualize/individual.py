"""Functions for visualizing individual levee systems."""

import logging
from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def diagnose_elevation_differences(system_id: str) -> None:
    """Print diagnostic information about elevation differences for a levee system."""
    try:
        # Load data
        data = gpd.read_parquet(f"data/processed/levee_{system_id}.parquet")

        # Basic stats
        print(f"\nDiagnostics for system {system_id}:")
        print(f"Total points: {len(data)}")

        # NLD elevation distribution
        nld_unique = data["elevation"].value_counts().sort_index()
        print("\nNLD elevation counts:")
        print(nld_unique)

        print("\nNLD elevation stats:")
        print(f"min: {data['elevation'].min():.2f}m")
        print(f"max: {data['elevation'].max():.2f}m")
        print(f"mean: {data['elevation'].mean():.2f}m")
        print(f"zeros: {(data['elevation'] == 0).sum()}")
        print(f"nulls: {data['elevation'].isna().sum()}")

        print("\n3DEP elevation stats:")
        print(f"min: {data['dep_elevation'].min():.2f}m")
        print(f"max: {data['dep_elevation'].max():.2f}m")
        print(f"mean: {data['dep_elevation'].mean():.2f}m")
        print(f"zeros: {(data['dep_elevation'] == 0).sum()}")
        print(f"nulls: {data['dep_elevation'].isna().sum()}")

        # Look at first few rows
        print("\nFirst few rows:")
        print(data[["elevation", "dep_elevation", "distance_along_track"]].head())

    except Exception as e:
        print(f"Error diagnosing system {system_id}: {str(e)}")


def plot_elevation_profile(
    system_id: str, save_dir: Optional[Path] = None
) -> Optional[Figure]:
    """Create elevation comparison plots for a levee system."""
    try:
        # Load data
        data = gpd.read_parquet(f"data/processed/levee_{system_id}.parquet")
        logger.info(f"Loaded {len(data)} points for system {system_id}")

        # Sort by distance for consistent plotting
        data = data.sort_values("distance_along_track")

        # Create validity mask (only where NLD has data)
        valid_mask = data["elevation"] > 0
        valid_data = data[valid_mask].copy()

        # Check if we have any valid points
        if len(valid_data) == 0:
            logger.error(f"No valid NLD points found for system {system_id}")
            return None

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Plot 1: Elevation profiles (only where NLD has data)
        ax1.plot(
            valid_data["distance_along_track"],
            valid_data["elevation"],
            "b.-",
            label="NLD",
            alpha=0.7,
        )
        ax1.plot(
            valid_data["distance_along_track"],
            valid_data["dep_elevation"],
            "r.-",
            label="3DEP",
            alpha=0.7,
        )

        # Add gaps in data
        if len(data) > len(valid_data):
            invalid_data = data[~valid_mask]
            ax1.plot(
                invalid_data["distance_along_track"],
                invalid_data["dep_elevation"],
                "r.",
                alpha=0.2,
                label="3DEP (no NLD data)",
            )

        ax1.set_title(f"Elevation Profiles - System {system_id}")
        ax1.set_xlabel("Distance Along Levee (m)")
        ax1.set_ylabel("Elevation (m)")
        ax1.legend(loc="best", bbox_to_anchor=(1, 1))
        ax1.grid(True)

        # Plot 2: Elevation differences
        differences = valid_data["elevation"] - valid_data["dep_elevation"]
        ax2.plot(valid_data["distance_along_track"], differences, "k.-")
        ax2.axhline(y=0, color="r", linestyle="--", alpha=0.5)

        # Add error bands
        mean_diff = differences.mean()
        std_diff = differences.std()
        ax2.axhline(y=mean_diff, color="g", linestyle="--", alpha=0.5, label="Mean")
        ax2.axhspan(
            mean_diff - std_diff,
            mean_diff + std_diff,
            color="g",
            alpha=0.1,
            label="±1 Std Dev",
        )

        ax2.set_title("Elevation Differences (NLD - 3DEP)")
        ax2.set_xlabel("Distance Along Levee (m)")
        ax2.set_ylabel("Difference (m)")
        ax2.grid(True)
        ax2.legend(loc="best", bbox_to_anchor=(1, 1))

        # Add statistics
        stats_text = (
            f"Total points: {len(data)}\n"
            f"Valid points: {len(valid_data)} ({100*len(valid_data)/len(data):.1f}%)\n"
            f"Mean diff: {mean_diff:.2f}m\n"
            f"Std dev: {std_diff:.2f}m\n"
            f"Min diff: {differences.min():.2f}m\n"
            f"Max diff: {differences.max():.2f}m"
        )

        ax1.text(
            0.02,
            0.98,
            stats_text,
            transform=ax1.transAxes,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.8),
        )

        plt.tight_layout()

        # Save if directory provided
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                save_dir / f"levee_profile_{system_id}.png",
                dpi=300,
                bbox_inches="tight",
            )
            logger.info(f"Saved plot to {save_dir}/levee_profile_{system_id}.png")

        return fig

    except Exception as e:
        logger.error(f"Error plotting system {system_id}: {str(e)}")
        return None


def get_system_statistics(system_id: str) -> Optional[dict]:
    """Get summary statistics for a levee system."""
    try:
        # Load data directly
        data = gpd.read_parquet(f"data/processed/levee_{system_id}.parquet")

        # Filter out zeros
        valid_data = data[data["elevation"] > 0].copy()
        if len(valid_data) < 2:
            return None

        # Calculate difference
        valid_data["difference"] = valid_data["elevation"] - valid_data["dep_elevation"]

        return {
            "system_id": system_id,
            "total_length": valid_data["distance_along_track"].max(),
            "mean_diff": valid_data["difference"].mean(),
            "median_diff": valid_data["difference"].median(),
            "std_diff": valid_data["difference"].std(),
            "rmse": np.sqrt((valid_data["difference"] ** 2).mean()),
            "n_points": len(valid_data),
            "min_elevation": valid_data["elevation"].min(),
            "max_elevation": valid_data["elevation"].max(),
            "min_difference": valid_data["difference"].min(),
            "max_difference": valid_data["difference"].max(),
        }

    except Exception as e:
        logger.error(f"Error getting statistics for system {system_id}: {str(e)}")
        return None
