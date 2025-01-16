"""Functions for sampling levee data."""

import argparse
import asyncio
import concurrent.futures
import json
import logging
import os
import subprocess
import tempfile
import time
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple, Union
from enum import Enum
import csv
from datetime import datetime

import aiohttp
import geopandas as gpd
import nest_asyncio
import numpy as np
import pandas as pd
import py3dep
import pygeoutils as geoutils
import rasterio
import requests
import xarray as xr
from pygeoogc import WMS, ServiceURL
from rasterio.transform import rowcol
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from requests.adapters import HTTPAdapter
from shapely import box as shapely_box
from shapely.geometry import Point
from tqdm import tqdm
from urllib3.util import Retry

from army_levees.core.visualize.utils import (filter_valid_segments,
                                              get_processed_systems,
                                              get_utm_crs, load_system_data)

# Enable asyncio in Jupyter
nest_asyncio.apply()

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

TIMEOUT = 120  # Increase timeout further
MAX_RETRIES = 3  # Reduce retries but make them more effective
CONCURRENT_LIMIT = 1  # Keep single connection
BASE_WAIT = 5  # Increase base wait time


# Configure requests session with retries
def create_session() -> requests.Session:
    """Create requests session with retries and timeouts."""
    session = requests.Session()
    retries = Retry(
        total=MAX_RETRIES,
        backoff_factor=1.0,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(
        max_retries=retries,
        pool_connections=CONCURRENT_LIMIT,
        pool_maxsize=CONCURRENT_LIMIT,
    )
    session.mount("https://", adapter)
    return session


# Create global session and stats
session = create_session()
profile_cache: Dict[str, gpd.GeoDataFrame] = {}


# Create a dataclass for statistics
@dataclass
class ProcessingStats:
    total_attempts: int = 0
    success: int = 0
    floodwalls: int = 0
    no_data: int = 0
    all_zeros: int = 0
    all_nans: int = 0
    too_few_points: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.success / max(1, self.total_attempts)

    def __str__(self) -> str:
        """Format statistics for display."""
        return (
            f"\nProcessing Statistics:\n"
            f"  Total attempts: {self.total_attempts}\n"
            f"  Success rate: {self.success_rate:.1%}\n"
            f"  Successful: {self.success}\n"
            f"  Floodwalls skipped: {self.floodwalls}\n"
            f"  No data available: {self.no_data}\n"
            f"  Invalid data:\n"
            f"    All zeros: {self.all_zeros}\n"
            f"    All NaNs: {self.all_nans}\n"
            f"    Too few points: {self.too_few_points}\n"
        )


@lru_cache(maxsize=1)
def get_usace_system_ids() -> List[str]:
    """Get list of USACE system IDs from NLD API."""
    url = "https://levees.sec.usace.army.mil:443/api-local/system-categories/usace-nonusace"
    try:
        response = session.get(url, timeout=30)
        if response.status_code == 200 and "USACE" in response.json():
            return json.loads(response.json()["USACE"])
        else:
            logger.warning(
                f"API request failed with status code: {response.status_code}. "
                "Falling back to existing data."
            )
            return []
    except requests.exceptions.RequestException as e:
        logger.warning(f"API request failed: {e}\n" "Falling back to existing data.")
        return []


class SystemStatus(Enum):
    """Status codes for levee systems."""
    SUCCESS = "success"  # Successfully processed
    HAS_FLOODWALL = "has_floodwall"  # Contains significant floodwall
    NO_DATA = "no_data"  # No data available from NLD
    EMPTY_RESPONSE = "empty_response"  # Empty response from API
    NO_3DEP = "no_3dep"  # No 3DEP coverage
    INVALID_BOUNDS = "invalid_bounds"  # Invalid coordinate bounds
    TOO_FEW_POINTS = "too_few_points"  # Not enough valid points
    ERROR = "error"  # Other error

def update_system_status(system_id: str, status: SystemStatus, details: str = ""):
    """Update the status log for a system."""
    status_file = Path("data/system_status.csv")

    # Create header if file doesn't exist
    if not status_file.exists():
        status_file.parent.mkdir(parents=True, exist_ok=True)
        with open(status_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['system_id', 'status', 'details', 'timestamp'])
        existing_statuses = {}
    else:
        # Read existing statuses
        existing_statuses = {}
        with open(status_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_statuses[row['system_id']] = row['status']

    # Only write if status has changed or doesn't exist
    if system_id not in existing_statuses or existing_statuses[system_id] != status.value:
        with open(status_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                system_id,
                status.value,
                details,
                datetime.now().isoformat()
            ])


async def get_nld_profile_async(
    system_id: str, session: aiohttp.ClientSession, stats: ProcessingStats
) -> Optional[np.ndarray]:
    """Get profile data for a system ID from NLD API asynchronously."""
    max_retries = MAX_RETRIES
    base_wait = BASE_WAIT

    for attempt in range(max_retries):
        try:
            # Add delay between attempts
            if attempt > 0:
                wait_time = base_wait * (2**attempt)  # Exponential backoff
                logger.info(f"Waiting {wait_time} seconds before retry...")
                await asyncio.sleep(wait_time)

            # First check for floodwalls
            segments_url = f"https://levees.sec.usace.army.mil:443/api-local/segments?system_id={system_id}"
            async with session.get(segments_url, timeout=30) as response:
                if response.status != 200:
                    logger.error(
                        f"Error getting segments: Status code {response.status}"
                    )
                    if attempt < max_retries - 1:
                        wait_time = base_wait * (2**attempt)  # Exponential backoff
                        logger.info(f"Retrying in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                        continue
                    return None

                segments = await response.json()

                # Only skip if any segment has floodwall miles
                has_floodwall = any(
                    segment.get("floodwallMiles", 0) > 0.0  # Add 0.0 mile threshold
                    for segment in segments
                )
                if has_floodwall:
                    stats.floodwalls += 1
                    logger.info(
                        f"System {system_id} contains floodwall miles. Skipping."
                    )
                    update_system_status(system_id, SystemStatus.HAS_FLOODWALL)
                    return None

            # Get profile data if no floodwalls
            url = f"https://levees.sec.usace.army.mil:443/api-local/system/{system_id}/route"
            async with session.get(url, timeout=30) as response:
                if response.status != 200:
                    logger.error(f"Error: Status code {response.status}")
                    if attempt < max_retries - 1:
                        wait_time = base_wait * (2**attempt)
                        logger.info(f"Retrying in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                        continue
                    return None

                profile_data = await response.json()
                if not profile_data:
                    logger.error("Error: Empty response")
                    update_system_status(system_id, SystemStatus.EMPTY_RESPONSE)
                    stats.no_data += 1
                    return None

                # Extract coordinates from topology
                arcs = profile_data.get("geometry", {}).get("arcs", [[]])[0]
                if not arcs:
                    logger.error("Error: No arcs found in topology")
                    stats.no_data += 1
                    return None

                # Convert to numpy array directly
                coords = np.array(arcs, dtype=np.float32)
                if len(coords) == 0 or coords.shape[1] < 4:
                    logger.error("Error: Invalid coordinate data")
                    stats.too_few_points += 1
                    return None

                return coords

        except asyncio.TimeoutError:
            logger.error(
                f"Timeout error for system {system_id} (attempt {attempt + 1}/{max_retries})"
            )
            if attempt < max_retries - 1:
                wait_time = base_wait * (2**attempt)
                logger.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            continue

        except aiohttp.ClientError as e:
            logger.error(
                f"Connection error for system {system_id}: {str(e)} (attempt {attempt + 1}/{max_retries})"
            )
            if attempt < max_retries - 1:
                wait_time = base_wait * (2**attempt)
                logger.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            continue

        except Exception as e:
            logger.error(f"Error getting profile data for system {system_id}: {str(e)}")
            update_system_status(system_id, SystemStatus.ERROR, str(e))
            return None

    logger.error(
        f"Failed to get profile data for system {system_id} after {max_retries} attempts"
    )
    return None


def create_geodataframe(
    coords: np.ndarray, system_id: str, dep_elevations: Tuple[np.ndarray, np.ndarray]
) -> Optional[gpd.GeoDataFrame]:
    """Create GeoDataFrame from coordinate array efficiently."""
    try:
        # Basic data validation
        if dep_elevations is None:
            logger.error(f"System {system_id}: No 3DEP elevations available")
            return None

        # Extract point and buffer elevations
        point_elevs, buffer_elevs = dep_elevations

        # Create points from coordinates
        points = [Point(x, y) for x, y in coords[:, :2]]

        # Create data dictionary
        data = {
            "system_id": [system_id] * len(coords),
            "geometry": points,
            "elevation": coords[:, 2],  # NLD elevation
            "dep_elevation": point_elevs,
            "dep_max_elevation": buffer_elevs,
            "distance_along_track": coords[:, 3],
        }

        # Calculate elevation differences
        data["difference"] = data["elevation"] - data["dep_elevation"]

        # Create GeoDataFrame with proper CRS
        gdf = gpd.GeoDataFrame(
            data, crs="EPSG:3857"  # Use string format instead of gpd.CRS
        )

        return gdf

    except Exception as e:
        logger.error(f"Error creating GeoDataFrame for system {system_id}: {str(e)}")
        return None


async def get_3dep_elevations_async(
    coords_list: List[Tuple[float, float]],
    batch_size: int = 500,
    crs: int = 3857,
    buffer_size: int = 3,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Get 3DEP elevations for coordinates."""
    try:
        # Create GeoDataFrame from points
        points = [Point(x, y) for x, y in coords_list]
        gdf = gpd.GeoDataFrame(
            {"geometry": points}, crs=f"EPSG:{crs}"  # Use string format for CRS
        )

        # Convert to WGS84 for DEM query
        gdf_4326 = gdf.to_crs(4326)
        bounds = gdf_4326.total_bounds

        # Add buffer to bounds
        buffer_deg = 0.001  # ~100m buffer
        bbox = (
            bounds[0] - buffer_deg,
            bounds[1] - buffer_deg,
            bounds[2] + buffer_deg,
            bounds[3] + buffer_deg,
        )

        # Create bounding box polygon for WMS request
        bbox_poly = shapely_box(*bbox)

        # Validate bounds
        if not (-180 <= bbox[0] <= 180 and -180 <= bbox[2] <= 180):
            logger.error(f"Invalid longitude values in bounds: {bbox}")
            return None
        if not (-90 <= bbox[1] <= 90 and -90 <= bbox[3] <= 90):
            logger.error(f"Invalid latitude values in bounds: {bbox}")
            return None

        # Initialize WMS client with each attempt to avoid DB locks
        max_retries = 3
        db_lock_retries = 5  # Specific retries for DB locks
        base_wait = 1

        # Create a temporary directory for this request
        with tempfile.TemporaryDirectory() as temp_dir:
            for attempt in range(max_retries):
                try:
                    # Create new WMS instance for each attempt
                    wms = WMS(
                        ServiceURL().wms.nm_3dep,
                        layers="3DEPElevation:None",
                        outformat="image/tiff",
                        crs=4326,
                        version="1.3.0",
                    )

                    # Handle potential DB locks with inner retry loop
                    for db_attempt in range(db_lock_retries):
                        try:
                            # Request map with reasonable resolution
                            resolution = 1  # Set to 1 meter resolution
                            tiff_path = wms.getmap_bybox(
                                bbox,
                                resolution,
                                box_crs=4326,
                                max_px=1000000,
                                tiff_dir=temp_dir,  # Use temporary directory
                            )

                            # Handle tiff_path being a list or single path
                            if isinstance(tiff_path, list):
                                tiff_path = tiff_path[0]

                            # Read the file into bytes
                            with open(tiff_path, "rb") as f:
                                tiff_bytes = f.read()

                            # Create the expected dictionary format
                            r_dict = {"name": tiff_bytes}

                            break  # Success, exit DB retry loop

                        except Exception as db_error:
                            if "database is locked" in str(db_error).lower():
                                if db_attempt < db_lock_retries - 1:
                                    wait_time = base_wait * (2**db_attempt)
                                    logger.warning(
                                        f"Database locked, retrying in {wait_time}s (attempt {db_attempt + 1}/{db_lock_retries})"
                                    )
                                    await asyncio.sleep(wait_time)
                                    continue
                            raise  # Re-raise if not a DB lock or out of retries

                    # Convert to xarray Dataset
                    dem = geoutils.gtiff2xarray(r_dict, bbox_poly, 4326)

                    if dem is None or dem.rio.nodata is None:
                        raise ValueError("Invalid DEM data returned")

                    # Get the DEM data array first
                    if isinstance(dem, xr.DataArray):
                        if dem.ndim == 3:
                            dem_data = dem.values[0]  # Handle 3D array (band, y, x)
                        else:
                            dem_data = dem.values  # Handle 2D array (y, x)
                    else:
                        logger.error("DEM data is not an xarray DataArray")
                        return None

                    # Ensure dem_data is 2D
                    if dem_data.ndim != 2:
                        logger.error(
                            f"Unexpected DEM data dimensions: {dem_data.shape}"
                        )
                        return None

                    # Log DEM details
                    transform = dem.rio.transform()
                    pixel_width = abs(transform[0])
                    pixel_height = abs(transform[4])
                    logger.info(
                        f"DEM details:\n"
                        f"  Shape: {dem_data.shape}\n"
                        f"  Resolution: {pixel_width:.2f}x{pixel_height:.2f} degrees\n"
                        f"  (~{pixel_width * 111000:.1f}x{pixel_height * 111000:.1f} meters at equator)\n"
                        f"  Requested resolution: {resolution}m"
                    )

                    # Initialize arrays for all points
                    point_elevations = np.full(
                        len(coords_list), np.nan, dtype=np.float32
                    )
                    max_buffer_elevations = np.full(
                        len(coords_list), np.nan, dtype=np.float32
                    )

                    for i in range(0, len(coords_list), batch_size):
                        batch_coords = coords_list[i : i + batch_size]
                        batch_gdf = gpd.GeoDataFrame(
                            {"geometry": [Point(x, y) for x, y in batch_coords]},
                            crs=crs,
                        ).to_crs(4326)

                        # Get point values using xarray
                        point_values = dem.sel(
                            x=xr.DataArray(batch_gdf.geometry.x),
                            y=xr.DataArray(batch_gdf.geometry.y),
                            method="nearest",
                        ).values

                        # Convert coordinates to pixel indices
                        xs = batch_gdf.geometry.x.values
                        ys = batch_gdf.geometry.y.values
                        transform = dem.rio.transform()

                        # Get DEM bounds from transform and shape
                        height, width = dem_data.shape
                        bounds_left = transform[2]
                        bounds_right = bounds_left + width * transform[0]
                        bounds_top = transform[5]
                        bounds_bottom = bounds_top + height * transform[4]

                        # Check coordinates against bounds
                        valid_x = (xs >= bounds_left) & (xs <= bounds_right)
                        valid_y = (ys >= bounds_bottom) & (ys <= bounds_top)
                        valid_mask = valid_x & valid_y

                        if not valid_mask.all():
                            n_invalid = (~valid_mask).sum()
                            logger.warning(
                                f"{n_invalid} points outside DEM bounds in batch {i//batch_size + 1}. "
                                f"Bounds: [{bounds_left:.6f}, {bounds_bottom:.6f}, {bounds_right:.6f}, {bounds_top:.6f}]"
                            )

                        # Process only valid points
                        valid_indices = np.where(valid_mask)[0]
                        row_indices, col_indices = rowcol(
                            transform, xs[valid_mask], ys[valid_mask]
                        )
                        for idx, row, col, point_val in zip(
                            valid_indices,
                            row_indices,
                            col_indices,
                            point_values.flatten()[valid_mask],
                        ):
                            try:
                                row, col = int(row), int(col)

                                # Calculate buffer window indices
                                row_start = max(0, row - buffer_size)
                                row_end = min(dem_data.shape[0], row + buffer_size + 1)
                                col_start = max(0, col - buffer_size)
                                col_end = min(dem_data.shape[1], col + buffer_size + 1)

                                if (
                                    row_start >= dem_data.shape[0]
                                    or row_end <= 0
                                    or col_start >= dem_data.shape[1]
                                    or col_end <= 0
                                ):
                                    logger.warning(
                                        f"Invalid buffer window indices at ({row}, {col}), using point value"
                                    )
                                    max_buffer = point_val
                                else:
                                    buffer_data = dem_data[
                                        row_start:row_end, col_start:col_end
                                    ]
                                    if buffer_data.size == 0:
                                        max_buffer = point_val
                                    else:
                                        max_buffer = np.nanmax(buffer_data)

                                # Store values in the full arrays
                                batch_idx = i + idx
                                point_elevations[batch_idx] = point_val
                                max_buffer_elevations[batch_idx] = max_buffer

                            except Exception as e:
                                logger.warning(
                                    f"Error processing point {i + idx}: {str(e)}"
                                )
                                # Leave as NaN

                    # Check if we have any valid elevations
                    valid_mask = ~np.isnan(point_elevations)
                    if not valid_mask.any():
                        logger.error("No valid elevations found")
                        return None

                    # Track valid elevations and their indices
                    valid_elevations = point_elevations[valid_mask]
                    valid_buffer_elevations = max_buffer_elevations[valid_mask]

                    logger.info(
                        f"Processed {len(valid_elevations)}/{len(coords_list)} points "
                        f"({len(valid_elevations)/len(coords_list):.1%} coverage)"
                    )

                    # Coverage check
                    if len(valid_elevations) / len(coords_list) >= 0.25:
                        logger.info(
                            f"3DEP coverage: {len(valid_elevations)/len(coords_list):.1%} "
                            f"({len(valid_elevations)}/{len(coords_list)} points)"
                        )
                        return point_elevations, max_buffer_elevations
                    else:
                        logger.error(
                            f"Too many missing 3DEP values "
                            f"({100 - len(valid_elevations)/len(coords_list)*100:.1f}%)"
                        )
                        return None

                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(
                            f"Failed to get DEM after {max_retries} attempts: {str(e)}"
                        )
                        return None
                    wait_time = base_wait * (2**attempt)
                    logger.warning(
                        f"WMS request failed (attempt {attempt + 1}/{max_retries}): {str(e)}"
                    )
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)

    except Exception as e:
        logger.error(f"Error getting elevation data: {str(e)}")
        return None


async def process_system(
    system_id: str, session: aiohttp.ClientSession, stats: ProcessingStats
) -> Optional[Dict]:
    """Process a single system asynchronously."""
    try:
        # Check if system has already been processed
        processed_path = Path("data/processed") / f"levee_{system_id}.parquet"
        if processed_path.exists():
            try:
                # Attempt to load existing data
                existing_data = gpd.read_parquet(processed_path)
                if len(existing_data) > 0:
                    logger.info(f"System {system_id}: Using existing data ({len(existing_data)} points)")
                    stats.success += 1  # Count existing data as success
                    update_system_status(system_id, SystemStatus.SUCCESS, f"Loaded {len(existing_data)} points")
                    return existing_data
                else:
                    logger.warning(f"System {system_id}: Existing data empty, reprocessing")
            except Exception as e:
                logger.warning(f"System {system_id}: Error loading existing data, reprocessing: {e}")
        else:
            logger.info(f"System {system_id}: No existing data found, downloading...")

        # Check cache next
        if system_id in profile_cache:
            return profile_cache[system_id]

        # Get profile data
        coords = await get_nld_profile_async(system_id, session, stats)
        if coords is None:
            stats.no_data += 1
            logger.warning(f"System {system_id}: No profile data available")
            return None

        # Check 3DEP coverage before proceeding
        bounds = [
            coords[:, 0].min() - 0.001,  # Add small buffer
            coords[:, 1].min() - 0.001,
            coords[:, 0].max() + 0.001,
            coords[:, 1].max() + 0.001,
        ]

        # Convert bounds if needed
        if bounds[0] > 180 or bounds[0] < -180:  # Likely in web mercator
            gdf = gpd.GeoDataFrame(
                geometry=[shapely_box(*bounds)],
                crs=3857
            ).to_crs(4326)
            bounds = gdf.total_bounds.tolist()

        # Check 3DEP coverage
        try:
            sources = py3dep.query_3dep_sources(bounds, crs=4326, res="1m")
            if sources is None or sources.empty:
                update_system_status(system_id, SystemStatus.NO_3DEP)
                return None
        except Exception as e:
            logger.debug(f"Error checking 3DEP coverage: {e}")
            update_system_status(system_id, SystemStatus.NO_3DEP)
            return None

        # Convert numpy array coordinates to list of tuples
        coords_list = [(x, y) for x, y in coords[:, :2]]

        # Get 3DEP elevations with increased buffer size
        dep_elevations = await get_3dep_elevations_async(
            coords_list, crs=3857, buffer_size=2
        )

        # Create GeoDataFrame with validation
        gdf = create_geodataframe(coords, system_id, dep_elevations)
        if gdf is None:
            return None

        # Apply filtering with relaxed constraints
        filtered_gdf = filter_valid_segments(
            gdf,
            min_points=3,  # Reduce minimum points requirement
            min_coverage=0.0  # Accept any coverage above zero
        )
        if len(filtered_gdf) == 0:
            logger.error(f"System {system_id}: No valid points after filtering")
            update_system_status(system_id, SystemStatus.TOO_FEW_POINTS)
            return None

        # Cache and save
        profile_cache[system_id] = filtered_gdf
        save_dir = Path("data/processed")
        save_dir.mkdir(parents=True, exist_ok=True)
        filtered_gdf.to_parquet(save_dir / f"levee_{system_id}.parquet")

        # Record successful processing
        update_system_status(system_id, SystemStatus.SUCCESS, f"Downloaded {len(filtered_gdf)} points")
        stats.success += 1

        return filtered_gdf

    except Exception as e:
        logger.error(f"Error processing system {system_id}: {str(e)}")
        update_system_status(system_id, SystemStatus.ERROR, str(e))
        return None


async def get_random_levees_async(n_samples: int = 10, max_concurrent: int = 1) -> list:
    """Get random sample of NEW levee systems using async/await."""
    stats = ProcessingStats()

    # Get existing and attempted systems
    existing_systems = set()
    attempted_systems = set()

    # Get systems from parquet files
    processed_dir = Path("data/processed")
    if processed_dir.exists():
        for f in processed_dir.glob("levee_*.parquet"):
            system_id = f.stem.replace("levee_", "")
            existing_systems.add(system_id)

    # Get systems from status.csv
    status_file = Path("data/system_status.csv")
    if status_file.exists():
        status_df = pd.read_csv(status_file)
        attempted_systems.update(status_df['system_id'].unique())

    logger.info(f"Found {len(existing_systems)} existing processed systems")
    logger.info(f"Found {len(attempted_systems)} previously attempted systems")

    # Get all system IDs
    usace_ids = get_usace_system_ids()
    if not usace_ids:
        logger.error("\nCannot get levee data: USACE API is unavailable.")
        return []

    # Filter out already processed AND attempted systems
    new_system_ids = [sid for sid in usace_ids if sid not in existing_systems and sid not in attempted_systems]
    logger.info(f"Found {len(new_system_ids)} completely new systems to try")

    if not new_system_ids:
        logger.error("\nNo new systems available to sample.")
        return []

    # Sample more than we need to account for failures
    sample_size = min(n_samples * 3, len(new_system_ids))
    sample_ids = np.random.choice(new_system_ids, size=sample_size, replace=False)

    # Process samples
    results = []
    async with aiohttp.ClientSession() as session:
        for system_id in tqdm(sample_ids, desc="Processing samples"):
            try:
                result = await process_system(system_id, session, stats)
                if result is not None:
                    results.append(result)
                    if len(results) >= n_samples:
                        break
            except Exception as e:
                logger.error(f"Error processing system {system_id}: {e}")
                continue

    return results


def get_random_levees(n_samples: int = 10, max_concurrent: int = 4) -> list:
    """Synchronous wrapper for async function."""
    return asyncio.run(
        get_random_levees_async(
            n_samples=n_samples,
            max_concurrent=max_concurrent,
        )
    )


def validate_1m_coverage(
    bbox: tuple[float, float, float, float], crs: int = 3857
) -> bool:
    """Validate that the entire bbox is covered by 1m 3DEP data."""
    try:
        # Convert bbox to 4326 if needed
        bbox_poly = shapely_box(*bbox)
        if crs != 4326:
            gdf = gpd.GeoDataFrame(
                {"geometry": [bbox_poly]},
                crs=f"EPSG:{crs}",  # Use string format for CRS
            )
            gdf_4326 = gdf.to_crs(4326)
            # Buffer slightly to avoid topology errors
            bbox_poly = gdf_4326.geometry.iloc[0].buffer(0.0001)
            bbox_4326 = bbox_poly.bounds
        else:
            bbox_4326 = bbox

        logger.debug(f"Querying 3DEP with WGS84 bbox: {bbox_4326}")

        # Query available 1m sources
        try:
            sources = py3dep.query_3dep_sources(bbox_4326, crs=4326, res="1m")
            if sources is None or sources.empty:
                logger.info(f"No 1m 3DEP coverage found for bbox {bbox_4326}")
                return False

            # Check if bbox is completely within coverage
            coverage = sources.geometry.unary_union
            contains = coverage.contains(bbox_poly)

            if not contains:
                logger.info("Bbox not completely covered by 1m 3DEP data")
                # Calculate percent coverage
                intersection = coverage.intersection(bbox_poly)
                coverage_pct = intersection.area / bbox_poly.area * 100
                logger.info(f"Coverage percentage: {coverage_pct:.1f}%")

            return contains

        except Exception as e:
            logger.error(f"3DEP query failed: {str(e)}")
            logger.error(f"Query bbox (WGS84): {bbox_4326}")
            return False

    except Exception as e:
        logger.error(f"Error in validate_1m_coverage: {str(e)}")
        logger.error(f"Input bbox: {bbox}")
        return False


def get_dem_vrt(
    bbox: tuple[float, float, float, float],
    resolution: int,
    vrt_path: Path | str,
    tiff_dir: Path | str = "cache",
    crs: int = 3857,
    require_1m: bool = True,
) -> None:
    """Get DEM data at best available resolution from 3DEP and save it as a VRT file."""
    try:
        # Convert paths to Path objects
        tiff_dir = Path(tiff_dir)
        vrt_path = Path(vrt_path)

        # Ensure cache directory exists
        tiff_dir.mkdir(parents=True, exist_ok=True)

        # Convert bbox to WGS84 if needed
        bbox_poly = shapely_box(*bbox)
        if crs != 4326:
            gdf = gpd.GeoDataFrame({"geometry": [bbox_poly]}, crs=crs)
            gdf_4326 = gdf.to_crs(4326)
            bbox_poly = gdf_4326.geometry.iloc[0].buffer(0.0001)
            bbox_4326 = bbox_poly.bounds
        else:
            bbox_4326 = bbox

        # Validate bounds
        if not all(-180 <= x <= 180 for x in (bbox_4326[0], bbox_4326[2])):
            raise ValueError("Invalid longitude values in bbox")
        if not all(-90 <= y <= 90 for y in (bbox_4326[1], bbox_4326[3])):
            raise ValueError("Invalid latitude values in bbox")

        # Check coverage if 1m required
        if require_1m:
            has_1m = validate_1m_coverage(bbox_4326, crs=4326)
            if not has_1m:
                if not require_1m:
                    logger.warning(
                        "1m data not available, falling back to lower resolution"
                    )
                else:
                    raise ValueError(
                        "1m 3DEP data is not available for the entire bounding box. "
                        "Use require_1m=False to override this check."
                    )

        # Get DEM with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                wms_url = ServiceURL().wms.nm_3dep
                wms = WMS(
                    wms_url,
                    layers="3DEPElevation:None",
                    outformat="image/tiff",
                    crs=4326,
                    validation=False,
                )

                # Get the DEM data
                fname = wms.getmap_bybox(
                    bbox_4326,
                    resolution,
                    max_px=8000000,
                    tiff_dir=str(tiff_dir),  # Convert to string for WMS
                )

                # Handle fname being a list
                if isinstance(fname, list):
                    fname = fname[0]

                # Convert to Path and check existence
                tiff_path = Path(fname)
                if not tiff_path.exists():
                    logger.error(f"DEM file not found: {tiff_path}")
                    continue

                # Create VRT file
                try:
                    # Use subprocess directly to control argument handling
                    cmd = ["gdalbuildvrt", str(vrt_path), str(tiff_path)]
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, check=True
                    )

                    if vrt_path.exists():
                        return None  # Success
                    else:
                        logger.error("VRT file not created")
                        continue

                except subprocess.CalledProcessError as e:
                    logger.error(f"gdalbuildvrt failed: {e.stderr}")
                    continue

            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(
                    f"WMS request failed (attempt {attempt + 1}/{max_retries}): {str(e)}"
                )
                time.sleep(1 * (attempt + 1))  # Exponential backoff

        raise RuntimeError("Failed to get DEM after all retries")

    except Exception as e:
        logger.error(f"Error in get_dem_vrt: {str(e)}")
        raise


def filter_valid_segments(
    gdf: gpd.GeoDataFrame,
    min_points: int = 3,  # Reduce from 10 to 3
    min_coverage: float = 0.0,  # Reduce from 0.25 to 0.0
) -> gpd.GeoDataFrame:
    """Filter a single levee profile."""
    try:
        # Sort by distance first
        gdf_sorted = gdf.sort_values("distance_along_track").copy()

        # Only filter out NaN values
        nan_mask = gdf_sorted["elevation"].isna() | gdf_sorted["dep_elevation"].isna()
        if nan_mask.any():
            gdf_sorted = gdf_sorted[~nan_mask].copy()

        # Check minimum points
        if len(gdf_sorted) < min_points:
            logger.error(f"Too few points after filtering ({len(gdf_sorted)})")
            return gpd.GeoDataFrame(columns=gdf_sorted.columns, crs=gdf_sorted.crs)

        return gdf_sorted

    except Exception as e:
        logger.error(f"Error in filter_valid_segments: {str(e)}")
        return gpd.GeoDataFrame(columns=gdf.columns, crs=gdf.crs)


def log_progress_summary(stats: ProcessingStats) -> None:
    """Log detailed progress summary."""
    logger.info(
        f"\nProgress Summary:\n"
        f"  Success Rate: {stats.success_rate:.1%}\n"
        f"  Systems Processed: {stats.success}/{stats.total_attempts}\n"
        f"  Failures:\n"
        f"    Floodwalls: {stats.floodwalls}\n"
        f"    No Data: {stats.no_data}\n"
        f"    Invalid Data:\n"
        f"      All zeros: {stats.all_zeros}\n"
        f"      All NaNs: {stats.all_nans}\n"
        f"      Too few points: {stats.too_few_points}\n"
    )


async def check_3dep_coverage(system_ids: List[str], session: aiohttp.ClientSession) -> Dict[str, str]:
    """Check 3DEP coverage for all levee systems."""
    coverage = {}
    all_bounds = []

    # First get bounds for all systems
    logger.info("Getting system bounds...")
    with tqdm(total=len(system_ids), desc="Collecting bounds") as pbar:
        for system_id in system_ids:
            try:
                # Get profile data
                coords = await get_nld_profile_async(system_id, session, ProcessingStats())
                if coords is not None:
                    bounds = [
                        coords[:, 0].min() - 0.001,  # Add small buffer
                        coords[:, 1].min() - 0.001,
                        coords[:, 0].max() + 0.001,
                        coords[:, 1].max() + 0.001,
                    ]
                    # Convert bounds to WGS84 if they're not already
                    if bounds[0] > 180 or bounds[0] < -180:  # Likely in web mercator
                        gdf = gpd.GeoDataFrame(
                            geometry=[shapely_box(*bounds)],
                            crs=3857
                        ).to_crs(4326)
                        bounds = gdf.total_bounds.tolist()
                    all_bounds.append((system_id, bounds))
            except Exception as e:
                logger.warning(f"Error getting bounds for system {system_id}: {e}")
            finally:
                pbar.update(1)

    # Now check 3DEP coverage for collected bounds in smaller batches
    logger.info(f"\nChecking 3DEP coverage for {len(all_bounds)} systems...")
    batch_size = 50  # Process in smaller batches

    for i in range(0, len(all_bounds), batch_size):
        batch = all_bounds[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1} of {(len(all_bounds) + batch_size - 1)//batch_size}")

        with tqdm(total=len(batch), desc="Checking 3DEP coverage") as pbar:
            for system_id, bbox in batch:
                try:
                    # Add error handling and retries for py3dep queries
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            sources = py3dep.query_3dep_sources(bbox, crs=4326, res="1m")
                            if sources is not None and not sources.empty:
                                coverage[system_id] = "1m"
                                break
                        except Exception as e:
                            if attempt == max_retries - 1:
                                logger.debug(f"Failed to check 1m coverage after {max_retries} attempts: {e}")
                            else:
                                await asyncio.sleep(1)  # Brief pause between retries

                    if system_id not in coverage:
                        coverage[system_id] = None
                        update_system_status(system_id, SystemStatus.NO_3DEP)

                except Exception as e:
                    logger.warning(f"Error checking coverage for system {system_id}: {e}")
                    coverage[system_id] = None
                    update_system_status(system_id, SystemStatus.ERROR, str(e))

                pbar.update(1)

        # Save progress after each batch
        save_3dep_coverage(coverage)
        logger.info(f"Saved progress for {len(coverage)} systems")

        # Brief pause between batches
        await asyncio.sleep(2)

    return coverage


def add_existing_systems_to_status():
    """Add existing processed systems to the status CSV."""
    processed_dir = Path("data/processed")
    if not processed_dir.exists():
        logger.error("No processed directory found")
        return

    # Get all existing parquet files
    existing_files = list(processed_dir.glob("levee_*.parquet"))
    logger.info(f"Found {len(existing_files)} existing processed files")

    for file_path in tqdm(existing_files, desc="Adding existing systems to status"):
        try:
            # Extract system ID from filename
            system_id = file_path.stem.replace("levee_", "")

            # Load the parquet file to get point count
            gdf = gpd.read_parquet(file_path)
            if len(gdf) > 0:
                update_system_status(
                    system_id,
                    SystemStatus.SUCCESS,
                    f"Existing data with {len(gdf)} points"
                )
            else:
                update_system_status(
                    system_id,
                    SystemStatus.TOO_FEW_POINTS,
                    "Empty file"
                )
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")


def get_all_system_ids() -> List[str]:
    """Get list of ALL system IDs (USACE and non-USACE) from NLD API."""
    url = "https://levees.sec.usace.army.mil:443/api-local/system-categories/usace-nonusace"
    try:
        response = session.get(url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            # Combine USACE and non-USACE systems
            usace_systems = json.loads(data.get("USACE", "[]"))
            nonusace_systems = json.loads(data.get("NON-USACE", "[]"))
            return usace_systems + nonusace_systems
        else:
            logger.warning(f"API request failed with status code: {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        logger.warning(f"API request failed: {e}")
        return []

def save_3dep_coverage(coverage: Dict[str, str]):
    """Save 3DEP coverage information to CSV."""
    coverage_file = Path("data/3dep_coverage.csv")
    coverage_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([
        {"system_id": sid, "resolution": res}
        for sid, res in coverage.items()
    ])
    df.to_csv(coverage_file, index=False)
    logger.info(f"Saved 3DEP coverage information for {len(df)} systems")

def load_3dep_coverage() -> Dict[str, str]:
    """Load existing 3DEP coverage information."""
    coverage_file = Path("data/3dep_coverage.csv")
    if not coverage_file.exists():
        return {}

    df = pd.read_csv(coverage_file)
    return dict(zip(df.system_id, df.resolution))

async def check_all_3dep_coverage(session: aiohttp.ClientSession):
    """Check and save 3DEP coverage for all levee systems."""
    # Load existing coverage
    existing_coverage = load_3dep_coverage()
    logger.info(f"Found existing 3DEP coverage for {len(existing_coverage)} systems")

    # Get all system IDs
    all_systems = get_all_system_ids()
    new_systems = [sid for sid in all_systems if sid not in existing_coverage]
    logger.info(f"Found {len(new_systems)} systems needing 3DEP coverage check")

    if new_systems:
        # Check coverage for new systems
        new_coverage = await check_3dep_coverage(new_systems, session)

        # Combine with existing coverage
        combined_coverage = {**existing_coverage, **new_coverage}

        # Save updated coverage
        save_3dep_coverage(combined_coverage)
        return combined_coverage
    return existing_coverage

async def main_async():
    """Async main function."""
    parser = argparse.ArgumentParser(
        description="Sample levee systems and compare NLD vs 3DEP elevations"
    )
    parser.add_argument(
        "-n", "--n_samples", type=int, default=10, help="Number of systems to sample"
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=1,
        help="Maximum number of concurrent connections",
    )
    parser.add_argument(
        "--check-3dep",
        action="store_true",
        help="Check 3DEP coverage for all systems"
    )
    args = parser.parse_args()

    # Create output directories
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    # Add existing systems to status first
    add_existing_systems_to_status()

    # Get samples
    logger.info(f"\nGetting {args.n_samples} random levee samples...")
    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        if args.check_3dep:
            await check_all_3dep_coverage(session)
        else:
            results = await get_random_levees_async(
                n_samples=args.n_samples,
                max_concurrent=args.max_concurrent,
            )

            # Log summary
            elapsed = time.time() - start_time
            logger.info(
                f"\nSummary:\n"
                f"  Systems processed: {len(results)}\n"
                f"  Total time: {elapsed:.1f} seconds\n"
                f"  Average time per system: {elapsed/max(1,len(results)):.1f} seconds"
            )

def main():
    """Command-line interface."""
    asyncio.run(main_async())

def calculate_system_difference(system_id: str) -> float:
    """Calculate mean elevation difference for a system."""
    try:
        # Load the parquet file
        file_path = Path(f"data/processed/levee_{system_id}.parquet")
        if not file_path.exists():
            logger.error(f"No data found for system {system_id}")
            return None

        # Read the data
        gdf = gpd.read_parquet(file_path)

        # Calculate difference (NLD - 3DEP)
        diff = gdf['elevation'] - gdf['dep_elevation']
        mean_diff = diff.mean()
        std_diff = diff.std()

        logger.info(f"\nSystem {system_id} Statistics:")
        logger.info(f"Mean difference (NLD - 3DEP): {mean_diff:.2f} meters")
        logger.info(f"Standard deviation: {std_diff:.2f} meters")
        logger.info(f"Number of points: {len(gdf)}")

        return mean_diff

    except Exception as e:
        logger.error(f"Error calculating difference for system {system_id}: {e}")
        return None

if __name__ == "__main__":
    main()
