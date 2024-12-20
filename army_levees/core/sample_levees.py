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
    try:
        # First check for floodwalls
        segments_url = f"https://levees.sec.usace.army.mil:443/api-local/segments?system_id={system_id}"
        async with session.get(segments_url) as response:
            if response.status != 200:
                logger.error(f"Error getting segments: Status code {response.status}")
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
        url = (
            f"https://levees.sec.usace.army.mil:443/api-local/system/{system_id}/route"
        )
        async with session.get(url) as response:
            if response.status != 200:
                logger.error(f"Error: Status code {response.status}")
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

    except Exception as e:
        logger.error(f"Error getting profile data for system {system_id}: {str(e)}")
        return None


def create_geodataframe(
    coords: np.ndarray, system_id: str
) -> Optional[gpd.GeoDataFrame]:
    """Create GeoDataFrame from coordinate array efficiently.

    Validates and filters NLD elevation data, converting from feet to meters.
    Performs several quality checks:
    1. Checks for NaN or zero values
    2. Validates elevation ranges
    3. Ensures minimum number of valid points
    """
    try:
        # Basic data validation
        if np.isnan(coords[:, 2]).all():
            logger.error(f"System {system_id}: All NLD elevations are NaN")
            return None

        if (coords[:, 2] == 0).all():
            logger.error(f"System {system_id}: All NLD elevations are zero")
            return None

        # Log original elevation range
        valid_ft = coords[~np.isnan(coords[:, 2]) & (coords[:, 2] != 0), 2]
        if len(valid_ft) == 0:
            logger.error(
                f"System {system_id}: No valid elevations after filtering zeros and NaNs"
            )
            return None

        logger.info(
            f"System {system_id} NLD elevations (feet): {valid_ft.min():.1f}ft to {valid_ft.max():.1f}ft"
        )

        # Convert to meters
        nld_elevations = valid_ft * 0.3048
        logger.info(
            f"System {system_id} NLD elevations (meters): {nld_elevations.min():.1f}m to {nld_elevations.max():.1f}m"
        )

        # Validate elevation range (above sea level and below Mt. Everest)
        if nld_elevations.max() > 8848 or nld_elevations.min() < -420:
            logger.error(
                f"System {system_id}: Unreasonable elevations after conversion"
            )
            return None

        # Create points only for valid elevations
        valid_mask = ~np.isnan(coords[:, 2]) & (coords[:, 2] != 0)
        valid_coords = coords[valid_mask]
        points = [Point(x, y) for x, y in zip(valid_coords[:, 0], valid_coords[:, 1])]

        # Create DataFrame with valid points
        df = pd.DataFrame(
            {
                "system_id": system_id,
                "elevation": valid_coords[:, 2] * 0.3048,  # Convert feet to meters
                "elevation_ft": valid_coords[:, 2],  # Keep original feet values
                "distance_along_track": valid_coords[:, 3],
                "geometry": points,
            }
        )

        # Check minimum number of points (at least 10 valid points)
        if len(df) < 10:
            logger.warning(f"System {system_id}: Too few valid points ({len(df)})")
            return None

        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:3857")

        # Convert to 4326 for elevation sampling
        gdf = gdf.to_crs("EPSG:4326")

        # Log final stats
        logger.info(
            f"System {system_id}: {len(df)} valid points out of {len(coords)} total"
        )

        return gdf

    except Exception as e:
        logger.error(f"Error creating GeoDataFrame for system {system_id}: {str(e)}")
        return None


async def get_3dep_elevations_async(
    coords_batch: List[Tuple[float, float]], batch_size: int = 1000
) -> Optional[np.ndarray]:
    """Get 3DEP elevations asynchronously in batches."""
    try:
        elevations = []
        for i in range(0, len(coords_batch), batch_size):
            batch = coords_batch[i : i + batch_size]
            # Run in executor to prevent blocking
            batch_elevs = await asyncio.get_event_loop().run_in_executor(
                None, py3dep.elevation_bycoords, batch, 4326
            )
            elevations.extend(batch_elevs)
        return np.array(elevations, dtype=np.float32)
    except Exception as e:
        logger.error(f"Error getting batch elevation data: {str(e)}")
        return None


async def process_system_async(
    system_id: str, session: aiohttp.ClientSession, stats: dict
) -> Optional[gpd.GeoDataFrame]:
    """Process a single system asynchronously.

    Handles data collection, validation, and filtering for a levee system.
    Returns None if any validation checks fail.
    """
    try:
        # Check cache first
        if system_id in profile_cache:
            return profile_cache[system_id]

        # Get NLD profile
        coords = await get_nld_profile_async(system_id, session, stats)
        if coords is None:
            stats["no_data"] += 1
            logger.warning(f"System {system_id}: No profile data available")
            return None

        # Create GeoDataFrame with validation
        gdf = create_geodataframe(coords, system_id)
        if gdf is None:
            # Error already logged in create_geodataframe
            return None

        # Get coordinates for 3DEP
        coords_list = list(zip(gdf.geometry.x, gdf.geometry.y))

        # Get 3DEP elevations
        dep_elevations = await get_3dep_elevations_async(coords_list)
        if dep_elevations is None:
            stats["no_data"] += 1
            logger.error(f"System {system_id}: Failed to get 3DEP elevations")
            return None

        # Add elevations and calculate difference
        gdf["dep_elevation"] = dep_elevations
        gdf["difference"] = gdf["elevation"] - gdf["dep_elevation"]

        # Log elevation comparison
        logger.info(
            f"System {system_id} 3DEP elevations: {gdf['dep_elevation'].min():.1f}m to {gdf['dep_elevation'].max():.1f}m"
        )
        logger.info(
            f"System {system_id} mean difference: {gdf['difference'].mean():.1f}m"
        )

        # Apply filtering
        from .visualize.utils import filter_valid_segments

        filtered_gdf = filter_valid_segments(gdf)
        if len(filtered_gdf) == 0:
            logger.error(f"System {system_id}: No valid points after filtering")
            return None

        # Cache filtered result
        profile_cache[system_id] = filtered_gdf

        # Save filtered system
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
