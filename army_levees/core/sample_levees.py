"""Functions for collecting levee elevation data from NLD and 3DEP.

This module provides the core functionality for:
1. Getting levee system IDs from the NLD API
2. Getting elevation profiles for each system
3. Getting matching 3DEP elevations
4. Saving the processed data
"""

import asyncio
import concurrent.futures
import json
import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import aiohttp
import geopandas as gpd
import nest_asyncio
import numpy as np
import pandas as pd
import py3dep
import requests
from requests.adapters import HTTPAdapter
from shapely.geometry import Point
from tqdm import tqdm
from urllib3.util import Retry
from shapely import box as shapely_box
from pygeoogc import WMS, ServiceURL
import pygeoutils as geoutils
from typing import Union, Literal
from army_levees.core.visualize.utils import filter_valid_segments, get_utm_crs
from rasterio.warp import transform_bounds
import rasterio
import tempfile
import subprocess
import warnings
import xarray as xr
from rasterio.windows import Window
import os
from rasterio.transform import rowcol

# Enable asyncio in Jupyter
nest_asyncio.apply()

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Configure requests session with retries
def create_session() -> requests.Session:
    """Create requests session with retries and timeouts."""
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# Create global session and stats
session = create_session()
profile_cache: Dict[str, gpd.GeoDataFrame] = {}

# Global statistics tracker
stats = {
    "floodwalls": 0,
    "no_data": 0,
    "all_zeros": 0,
    "all_nans": 0,
    "too_few_points": 0,
    "success": 0,
}


@lru_cache(maxsize=1)
def get_usace_system_ids() -> List[str]:
    """Get list of USACE system IDs from NLD API."""
    url = "https://levees.sec.usace.army.mil:443/api-local/system-categories/usace-nonusace"
    try:
        response = session.get(url, timeout=30)
        if response.status_code == 200 and "USACE" in response.json():
            return json.loads(response.json()["USACE"])
        else:
            logger.error(
                f"Error: 'USACE' key issue or request failed with status code: {response.status_code}"
            )
            return []
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return []


async def get_nld_profile_async(
    system_id: str, session: aiohttp.ClientSession, stats: dict
) -> Optional[np.ndarray]:
    """Get profile data for a system ID from NLD API asynchronously."""
    max_retries = 3
    base_wait = 1  # Base wait time in seconds
    
    for attempt in range(max_retries):
        try:
            # First check for floodwalls
            segments_url = f"https://levees.sec.usace.army.mil:443/api-local/segments?system_id={system_id}"
            async with session.get(segments_url, timeout=30) as response:
                if response.status != 200:
                    logger.error(f"Error getting segments: Status code {response.status}")
                    if attempt < max_retries - 1:
                        wait_time = base_wait * (2 ** attempt)  # Exponential backoff
                        logger.info(f"Retrying in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                        continue
                    return None

                segments = await response.json()
                
                # Check if there are any floodwall miles
                has_floodwall = any(
                    segment.get("floodwallMiles", 0) > 0 for segment in segments
                )
                if has_floodwall:
                    stats["floodwalls"] += 1
                    logger.info(f"System {system_id} contains floodwall miles. Skipping.")
                    return None

            # Get profile data if no floodwalls
            url = f"https://levees.sec.usace.army.mil:443/api-local/system/{system_id}/route"
            async with session.get(url, timeout=30) as response:
                if response.status != 200:
                    logger.error(f"Error: Status code {response.status}")
                    if attempt < max_retries - 1:
                        wait_time = base_wait * (2 ** attempt)
                        logger.info(f"Retrying in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                        continue
                    return None

                profile_data = await response.json()
                if not profile_data:
                    logger.error("Error: Empty response")
                    return None

                # Extract coordinates from topology
                arcs = profile_data.get("geometry", {}).get("arcs", [[]])[0]
                if not arcs:
                    logger.error("Error: No arcs found in topology")
                    return None

                # Convert to numpy array directly
                coords = np.array(arcs, dtype=np.float32)
                if len(coords) == 0 or coords.shape[1] < 4:
                    logger.error("Error: Invalid coordinate data")
                    return None

                return coords

        except asyncio.TimeoutError:
            logger.error(f"Timeout error for system {system_id} (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                wait_time = base_wait * (2 ** attempt)
                logger.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            continue
            
        except aiohttp.ClientError as e:
            logger.error(f"Connection error for system {system_id}: {str(e)} (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                wait_time = base_wait * (2 ** attempt)
                logger.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            continue
            
        except Exception as e:
            logger.error(f"Error getting profile data for system {system_id}: {str(e)}")
            return None

    logger.error(f"Failed to get profile data for system {system_id} after {max_retries} attempts")
    return None


def create_geodataframe(
    coords: np.ndarray, 
    system_id: str,
    dep_elevations: Tuple[np.ndarray, np.ndarray]
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
            'system_id': [system_id] * len(coords),
            'geometry': points,
            'elevation': coords[:, 2],  # NLD elevation
            'dep_elevation': point_elevs,
            'dep_max_elevation': buffer_elevs,
            'distance_along_track': coords[:, 3]
        }
        
        # Calculate elevation differences
        data['difference'] = data['elevation'] - data['dep_elevation']
        
        # Create GeoDataFrame with proper CRS
        gdf = gpd.GeoDataFrame(
            data,
            crs="EPSG:3857"  # Use string format instead of gpd.CRS
        )
        
        return gdf

    except Exception as e:
        logger.error(f"Error creating GeoDataFrame for system {system_id}: {str(e)}")
        return None


async def get_3dep_elevations_async(
    coords_list: List[Tuple[float, float]], 
    batch_size: int = 1000,
    crs: int = 3857,
    buffer_size: int = 5
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Get 3DEP elevations using WMS with improved error handling."""
    try:
        # Create GeoDataFrame from points
        points = [Point(x, y) for x, y in coords_list]
        gdf = gpd.GeoDataFrame(
            {"geometry": points}, 
            crs=f"EPSG:{crs}"  # Use string format for CRS
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
            bounds[3] + buffer_deg
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
                        version="1.3.0"
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
                                tiff_dir=temp_dir  # Use temporary directory
                            )
                            
                            # Handle tiff_path being a list or single path
                            if isinstance(tiff_path, list):
                                tiff_path = tiff_path[0]
                                
                            # Read the file into bytes
                            with open(tiff_path, 'rb') as f:
                                tiff_bytes = f.read()
                                
                            # Create the expected dictionary format
                            r_dict = {'name': tiff_bytes}
                            
                            break  # Success, exit DB retry loop
                            
                        except Exception as db_error:
                            if "database is locked" in str(db_error).lower():
                                if db_attempt < db_lock_retries - 1:
                                    wait_time = base_wait * (2 ** db_attempt)
                                    logger.warning(f"Database locked, retrying in {wait_time}s (attempt {db_attempt + 1}/{db_lock_retries})")
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
                        logger.error(f"Unexpected DEM data dimensions: {dem_data.shape}")
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
                    point_elevations = np.full(len(coords_list), np.nan, dtype=np.float32)
                    max_buffer_elevations = np.full(len(coords_list), np.nan, dtype=np.float32)
                    
                    for i in range(0, len(coords_list), batch_size):
                        batch_coords = coords_list[i:i + batch_size]
                        batch_gdf = gpd.GeoDataFrame(
                            {"geometry": [Point(x, y) for x, y in batch_coords]},
                            crs=crs
                        ).to_crs(4326)
                        
                        # Get point values using xarray
                        point_values = dem.sel(
                            x=xr.DataArray(batch_gdf.geometry.x),
                            y=xr.DataArray(batch_gdf.geometry.y),
                            method="nearest"
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
                        row_indices, col_indices = rowcol(transform, xs[valid_mask], ys[valid_mask])
                        for idx, row, col, point_val in zip(
                            valid_indices,
                            row_indices,
                            col_indices,
                            point_values.flatten()[valid_mask]
                        ):
                            try:
                                row, col = int(row), int(col)
                                
                                # Calculate buffer window indices
                                row_start = max(0, row - buffer_size)
                                row_end = min(dem_data.shape[0], row + buffer_size + 1)
                                col_start = max(0, col - buffer_size)
                                col_end = min(dem_data.shape[1], col + buffer_size + 1)
                                
                                if (row_start >= dem_data.shape[0] or row_end <= 0 or 
                                    col_start >= dem_data.shape[1] or col_end <= 0):
                                    logger.warning(f"Invalid buffer window indices at ({row}, {col}), using point value")
                                    max_buffer = point_val
                                else:
                                    buffer_data = dem_data[row_start:row_end, col_start:col_end]
                                    if buffer_data.size == 0:
                                        max_buffer = point_val
                                    else:
                                        max_buffer = np.nanmax(buffer_data)
                                
                                # Store values in the full arrays
                                batch_idx = i + idx
                                point_elevations[batch_idx] = point_val
                                max_buffer_elevations[batch_idx] = max_buffer
                                
                            except Exception as e:
                                logger.warning(f"Error processing point {i + idx}: {str(e)}")
                                # Leave as NaN
                    
                    # Check if we have any valid elevations
                    valid_mask = ~np.isnan(point_elevations)
                    if not valid_mask.any():
                        logger.error("No valid elevations found")
                        return None
                    
                    logger.info(
                        f"Processed {valid_mask.sum()}/{len(coords_list)} points "
                        f"({valid_mask.sum()/len(coords_list):.1%} coverage)"
                    )
                    
                    # Additional validation of point and buffer elevations
                    if point_elevations is not None and max_buffer_elevations is not None:
                        # Check for too many missing values
                        nan_ratio = np.isnan(point_elevations).mean()
                        if nan_ratio > 0.5:  # Increased from 0.2 to 0.5
                            logger.error(f"Too many missing 3DEP values ({nan_ratio:.1%})")
                            return None
                            
                        # Only check for completely missing data, not zeros
                        if np.all(np.isnan(point_elevations)):
                            logger.error("No valid 3DEP elevations found")
                            return None
                            
                        # Log coverage statistics
                        valid_points = ~np.isnan(point_elevations)
                        logger.info(
                            f"3DEP coverage: {valid_points.mean():.1%} "
                            f"({valid_points.sum()}/{len(point_elevations)} points)"
                        )
                        
                        return point_elevations, max_buffer_elevations
                        
                    return None
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to get DEM after {max_retries} attempts: {str(e)}")
                        return None
                    wait_time = base_wait * (2 ** attempt)
                    logger.warning(f"WMS request failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    
    except Exception as e:
        logger.error(f"Error getting elevation data: {str(e)}")
        return None


async def process_system_async(
    system_id: str, session: aiohttp.ClientSession, stats: dict
) -> Optional[gpd.GeoDataFrame]:
    """Process a single system asynchronously."""
    try:
        # Check if system has already been processed
        processed_path = Path("data/processed") / f"levee_{system_id}.parquet"
        if processed_path.exists():
            try:
                # Attempt to load existing data
                existing_data = gpd.read_parquet(processed_path)
                if len(existing_data) > 0:
                    logger.info(f"System {system_id}: Loading existing data")
                    return existing_data
                else:
                    logger.warning(f"System {system_id}: Existing data empty, reprocessing")
            except Exception as e:
                logger.warning(f"System {system_id}: Error loading existing data, reprocessing: {e}")
        
        # Check cache next
        if system_id in profile_cache:
            return profile_cache[system_id]

        # Get NLD profile
        coords = await get_nld_profile_async(system_id, session, stats)
        if coords is None:
            stats["no_data"] += 1
            logger.warning(f"System {system_id}: No profile data available")
            return None

        # Convert numpy array coordinates to list of tuples
        coords_list = [(x, y) for x, y in coords[:, :2]]
        
        # Get 3DEP elevations with increased buffer size
        dep_elevations = await get_3dep_elevations_async(
            coords_list, 
            crs=3857,
            buffer_size=2  # 5x5 window (2 pixels on each side)
        )
        
        # Create GeoDataFrame with validation
        gdf = create_geodataframe(coords, system_id, dep_elevations)
        if gdf is None:
            return None

        # Apply filtering
        filtered_gdf = filter_valid_segments(gdf)
        if len(filtered_gdf) == 0:
            logger.error(f"System {system_id}: No valid points after filtering")
            return None

        # Cache and save
        profile_cache[system_id] = filtered_gdf
        save_dir = Path("data/processed")
        save_dir.mkdir(parents=True, exist_ok=True)
        filtered_gdf.to_parquet(save_dir / f"levee_{system_id}.parquet")

        return filtered_gdf

    except Exception as e:
        logger.error(f"Error processing system {system_id}: {str(e)}")
        return None


def get_processed_systems() -> Set[str]:
    """Get set of system IDs that have already been processed."""
    processed_dir = Path("data/processed")
    if not processed_dir.exists():
        return set()

    return {p.stem.replace("levee_", "") for p in processed_dir.glob("levee_*.parquet")}


async def get_random_levees_async(
    n_samples: int = 10, skip_existing: bool = True, max_concurrent: int = 4
) -> list:
    """Get random sample of n levee systems using async/await."""
    # Get list of all USACE system IDs
    usace_ids = get_usace_system_ids()

    if skip_existing:
        # Remove already processed systems
        processed = get_processed_systems()
        usace_ids = [id for id in usace_ids if id not in processed]
        logger.info(
            f"Found {len(processed)} existing systems, sampling from remaining {len(usace_ids)}"
        )

    if not usace_ids:
        logger.error("No systems available to sample")
        return []

    results = []
    attempted_ids = set()
    # Increase batch size for larger samples
    batch_size = min(100, max(n_samples * 2, 50))

    # Reset statistics
    global stats
    stats.update({k: 0 for k in stats})

    while len(results) < n_samples and len(attempted_ids) < len(usace_ids):
        # Sample new batch of IDs that haven't been attempted
        available_ids = [id for id in usace_ids if id not in attempted_ids]
        if not available_ids:
            break

        sample_ids = np.random.choice(
            available_ids, min(batch_size, len(available_ids)), replace=False
        )
        attempted_ids.update(sample_ids)

        # Process systems concurrently
        connector = aiohttp.TCPConnector(limit=max_concurrent)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [process_system_async(sid, session, stats) for sid in sample_ids]
            for task in tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc=f"Processing systems ({len(results)}/{n_samples})",
            ):
                try:
                    result = await task
                    if result is not None:
                        results.append(result)
                        stats["success"] += 1
                    if len(results) >= n_samples:
                        break
                except Exception as e:
                    logger.error(f"Task failed: {str(e)}")

        # Log progress after each batch
        logger.info(f"\nProgress after {len(attempted_ids)} attempts:")
        logger.info(f"Successfully processed: {len(results)}/{n_samples}")
        logger.info(f"Success rate: {len(results)/max(1,len(attempted_ids)):.1%}")

        if len(results) >= n_samples:
            break

    # Log final statistics
    logger.info("\nFinal Statistics:")
    logger.info(f"Total attempts: {len(attempted_ids)}")
    logger.info(f"Success rate: {len(results)/max(1,len(attempted_ids)):.1%}")
    logger.info(f"Systems with floodwalls: {stats['floodwalls']}")
    logger.info(f"Systems with no data: {stats['no_data']}")
    logger.info(f"Systems with all zeros: {stats['all_zeros']}")
    logger.info(f"Systems with all NaNs: {stats['all_nans']}")
    logger.info(f"Systems with too few points: {stats['too_few_points']}")
    logger.info(f"Successfully processed: {stats['success']}")

    return results[:n_samples]  # Ensure we return exactly n_samples


def get_random_levees(
    n_samples: int = 10, skip_existing: bool = True, max_concurrent: int = 4
) -> list:
    """Synchronous wrapper for async function."""
    return asyncio.run(
        get_random_levees_async(
            n_samples=n_samples,
            skip_existing=skip_existing,
            max_concurrent=max_concurrent,
        )
    )


def validate_1m_coverage(bbox: tuple[float, float, float, float], crs: int = 3857) -> bool:
    """Validate that the entire bbox is covered by 1m 3DEP data."""
    try:
        # Convert bbox to 4326 if needed
        bbox_poly = shapely_box(*bbox)
        if crs != 4326:
            gdf = gpd.GeoDataFrame(
                {"geometry": [bbox_poly]}, 
                crs=f"EPSG:{crs}"  # Use string format for CRS
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
            gdf = gpd.GeoDataFrame(
                {"geometry": [bbox_poly]}, 
                crs=crs
            )
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
                    logger.warning("1m data not available, falling back to lower resolution")
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
                    validation=False
                )
                
                # Get the DEM data
                fname = wms.getmap_bybox(
                    bbox_4326, 
                    resolution,
                    max_px=8000000,
                    tiff_dir=str(tiff_dir)  # Convert to string for WMS
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
                    cmd = ['gdalbuildvrt', str(vrt_path), str(tiff_path)]
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        check=True
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
                logger.warning(f"WMS request failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                time.sleep(1 * (attempt + 1))  # Exponential backoff
                
        raise RuntimeError("Failed to get DEM after all retries")
        
    except Exception as e:
        logger.error(f"Error in get_dem_vrt: {str(e)}")
        raise


def filter_valid_segments(gdf: gpd.GeoDataFrame, 
                         max_diff_std: Optional[float] = None,
                         max_mean_diff: float = 20.0,
                         outlier_threshold: float = 3.0,
                         min_points: int = 10) -> gpd.GeoDataFrame:
    """Filter a single levee profile.
    
    This is a light filter - just removes obvious errors.
    Detailed filtering happens in filter_levees.py.
    """
    try:
        # Sort by distance first
        gdf_sorted = gdf.sort_values('distance_along_track').copy()
        
        # Only filter out NaN values
        nan_mask = gdf_sorted['elevation'].isna() | gdf_sorted['dep_elevation'].isna()
        if nan_mask.any():
            gdf_sorted = gdf_sorted[~nan_mask].copy()
            
        # Check minimum points
        if len(gdf_sorted) < min_points:
            logger.error(f"Too few points after filtering ({len(gdf_sorted)})")
            return gpd.GeoDataFrame(columns=gdf_sorted.columns, crs=gdf_sorted.crs)
            
        # Check if differences are too large
        diff = gdf_sorted['elevation'] - gdf_sorted['dep_elevation']
        if abs(diff.mean()) > max_mean_diff:
            logger.error(
                f"Large mean elevation difference ({diff.mean():.1f}m) "
                f"This suggests a systematic offset between NLD and 3DEP data."
            )
            return gpd.GeoDataFrame(columns=gdf_sorted.columns, crs=gdf_sorted.crs)
            
        return gdf_sorted
        
    except Exception as e:
        logger.error(f"Error in filter_valid_segments: {str(e)}")
        return gpd.GeoDataFrame(columns=gdf.columns, crs=gdf.crs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sample levee systems and compare NLD vs 3DEP elevations"
    )
    parser.add_argument(
        "-n", "--n_samples", type=int, default=10, help="Number of systems to sample"
    )
    parser.add_argument(
        "--include_existing",
        action="store_true",
        help="Include already processed systems in sampling",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=4,
        help="Maximum number of concurrent connections",
    )
    args = parser.parse_args()

    # Create output directories
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    # Get samples
    print(f"\nGetting {args.n_samples} random levee samples...")
    start_time = time.time()

    results = get_random_levees(
        n_samples=args.n_samples,
        skip_existing=not args.include_existing,
        max_concurrent=args.max_concurrent,
    )

    elapsed = time.time() - start_time
    print(f"Successfully processed {len(results)} systems in {elapsed:.1f} seconds")
    print(f"Average time per system: {elapsed/max(1,len(results)):.1f} seconds\n")

    # Show total processed
    processed = get_processed_systems()
    print(f"Total systems processed: {len(processed)}")
