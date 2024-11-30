"""Utility functions for visualization."""

import logging
from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def filter_valid_segments(data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Filter out invalid elevation segments.

    Returns:
        GeoDataFrame with only valid points (non-zero NLD elevations).
        If no valid points exist, returns an empty GeoDataFrame.
    """
    try:
        logger.info("Starting to filter segments...")
        logger.info(f"Initial data points: {len(data)}")

        # Create a deep copy to avoid SettingWithCopyWarning
        data = data.copy(deep=True)

        # Check minimum number of points
        if len(data) < 20:
            logger.error(f"Too few points: {len(data)} (minimum 20 required)")
            return gpd.GeoDataFrame(
                geometry=gpd.GeoSeries(dtype="geometry"), crs=data.crs
            )

        # Check for near-zero NLD elevations when 3DEP shows significant elevation
        near_zero_mask = data["elevation"] < 0.01  # 1cm threshold for "near zero"
        if near_zero_mask.any():
            near_zero_sections = data[near_zero_mask]
            mean_dep_at_zeros = near_zero_sections["dep_elevation"].mean()
            if (
                mean_dep_at_zeros > 5
            ):  # If 3DEP shows >5m elevation where NLD is near zero
                logger.error(
                    f"Found near-zero NLD elevations while 3DEP shows significant elevation (mean: {mean_dep_at_zeros:.1f}m)"
                )
                return gpd.GeoDataFrame(
                    geometry=gpd.GeoSeries(dtype="geometry"), crs=data.crs
                )

        # Calculate elevation differences between consecutive points
        data["elevation_diff"] = data["elevation"].diff()
        data["elevation_diff_pct"] = data["elevation_diff"] / data["elevation"].shift()

        # Calculate elevation differences with 3DEP
        data["difference"] = data["elevation"] - data["dep_elevation"]
        mean_diff = data["difference"].mean()
        std_diff = data["difference"].std()

        # Debug logging for offset detection
        logger.info(
            f"Checking offset conditions: mean_diff={mean_diff:.1f}m, std_diff={std_diff:.1f}m"
        )

        # Check for offsets in order of severity:

        # 1. Basic threshold check (>10m mean difference)
        if abs(mean_diff) > 10:
            logger.error(f"Large mean elevation difference detected: {mean_diff:.1f}m")
            return gpd.GeoDataFrame(
                geometry=gpd.GeoSeries(dtype="geometry"), crs=data.crs
            )

        # 2. Very large offsets (>500m) - likely unit conversion errors
        if abs(mean_diff) > 500:
            logger.error(f"Extreme elevation difference detected: {mean_diff:.1f}m")
            return gpd.GeoDataFrame(
                geometry=gpd.GeoSeries(dtype="geometry"), crs=data.crs
            )

        # 3. Inconsistent offsets (high std dev)
        if std_diff > 25:
            logger.error(
                f"Inconsistent elevation differences detected (std: {std_diff:.1f}m)"
            )
            return gpd.GeoDataFrame(
                geometry=gpd.GeoSeries(dtype="geometry"), crs=data.crs
            )

        # 4. Check for unrealistic elevation jumps (>100m) in NLD data
        max_elev_jump = data["elevation_diff"].abs().max()
        if max_elev_jump > 100:
            logger.error(
                f"Unrealistic elevation jump detected in NLD data: {max_elev_jump:.1f}m"
            )
            return gpd.GeoDataFrame(
                geometry=gpd.GeoSeries(dtype="geometry"), crs=data.crs
            )

        # 5. Check for unrealistic absolute elevations (>1000m change from mean)
        mean_elev = data["elevation"].mean()
        max_elev_diff = (data["elevation"] - mean_elev).abs().max()
        if max_elev_diff > 1000:
            logger.error(
                f"Unrealistic elevation range detected in NLD data: {max_elev_diff:.1f}m from mean"
            )
            return gpd.GeoDataFrame(
                geometry=gpd.GeoSeries(dtype="geometry"), crs=data.crs
            )

        # Sort by distance and reset index
        data = data.sort_values("distance_along_track")
        data = gpd.GeoDataFrame(data, geometry="geometry", crs=data.crs)
        data = data.reset_index(drop=True)

        # Log statistics
        total_length = data["distance_along_track"].max()
        logger.info(f"Total levee length: {total_length:.1f}m")
        logger.info(f"Valid points: {len(data)}")
        logger.info(f"Mean elevation difference: {mean_diff:.1f}m")

        return data

    except Exception as e:
        logger.error(f"Error in filter_valid_segments: {str(e)}")
        return gpd.GeoDataFrame(geometry=gpd.GeoSeries(dtype="geometry"), crs=data.crs)


def load_levee_data(system_id: str) -> Optional[gpd.GeoDataFrame]:
    """Load and preprocess data for a levee system."""
    try:
        data = gpd.read_parquet(f"data/processed/levee_{system_id}.parquet")
        return filter_valid_segments(data)
    except Exception as e:
        logger.error(f"Error loading data for system {system_id}: {str(e)}")
        return None


def get_processed_systems() -> List[str]:
    """Get list of all processed levee system IDs."""
    processed_dir = Path("data/processed")
    if not processed_dir.exists():
        return []
    return [f.stem.replace("levee_", "") for f in processed_dir.glob("levee_*.parquet")]
