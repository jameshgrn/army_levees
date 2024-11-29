"""Functions for collecting levee elevation data from NLD and 3DEP.

This module provides the core functionality for:
1. Getting levee system IDs from the NLD API
2. Getting elevation profiles for each system
3. Getting matching 3DEP elevations
4. Saving the processed data
"""

import geopandas as gpd
import numpy as np
import requests
import json
from pathlib import Path
import logging
from typing import List, Optional, Set, Dict, Tuple, Union
import py3dep
import concurrent.futures
from functools import lru_cache
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import pandas as pd
from tqdm import tqdm
from shapely.geometry import Point
import asyncio
import aiohttp
import nest_asyncio

# Enable asyncio in Jupyter
nest_asyncio.apply()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure requests session with retries
def create_session() -> requests.Session:
    """Create requests session with retries and timeouts."""
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session

# Create global session
session = create_session()

# Cache for NLD profiles
profile_cache: Dict[str, gpd.GeoDataFrame] = {}

@lru_cache(maxsize=1)
def get_usace_system_ids() -> List[str]:
    """Get list of USACE system IDs from NLD API."""
    url = 'https://levees.sec.usace.army.mil:443/api-local/system-categories/usace-nonusace'
    try:
        response = session.get(url, timeout=30)
        if response.status_code == 200 and 'USACE' in response.json():
            return json.loads(response.json()['USACE'])
        else:
            logger.error(f"Error: 'USACE' key issue or request failed with status code: {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return []

async def get_nld_profile_async(system_id: str, session: aiohttp.ClientSession) -> Optional[np.ndarray]:
    """Get profile data for a system ID from NLD API asynchronously."""
    try:
        # First check for floodwalls
        segments_url = f'https://levees.sec.usace.army.mil:443/api-local/segments?system_id={system_id}'
        async with session.get(segments_url) as response:
            if response.status != 200:
                logger.error(f"Error getting segments: Status code {response.status}")
                return None
                
            segments = await response.json()
            
            # Check if there are any floodwall miles
            has_floodwall = any(segment.get('floodwallMiles', 0) > 0 for segment in segments)
            if has_floodwall:
                logger.info(f"System {system_id} contains floodwall miles. Skipping.")
                return None

        # Get profile data if no floodwalls
        url = f'https://levees.sec.usace.army.mil:443/api-local/system/{system_id}/route'
        async with session.get(url) as response:
            if response.status != 200:
                logger.error(f"Error: Status code {response.status}")
                return None
                
            profile_data = await response.json()
            if not profile_data:
                logger.error("Error: Empty response")
                return None
                
            # Extract coordinates from topology
            arcs = profile_data.get('geometry', {}).get('arcs', [[]])[0]
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

def create_geodataframe(coords: np.ndarray, system_id: str) -> Optional[gpd.GeoDataFrame]:
    """Create GeoDataFrame from coordinate array efficiently."""
    try:
        # Create points using numpy operations
        points = [Point(x, y) for x, y in zip(coords[:, 0], coords[:, 1])]
        
        # Create DataFrame first (faster than direct GeoDataFrame creation)
        df = pd.DataFrame({
            'system_id': system_id,
            'elevation': coords[:, 2] * 0.3048,  # Convert feet to meters
            'distance_along_track': coords[:, 3],
            'geometry': points
        })
        
        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:3857")
        
        # Convert to 4326 for elevation sampling
        gdf = gdf.to_crs("EPSG:4326")
        
        return gdf
        
    except Exception as e:
        logger.error(f"Error creating GeoDataFrame: {str(e)}")
        return None

async def get_3dep_elevations_async(coords_batch: List[Tuple[float, float]], 
                                  batch_size: int = 1000) -> Optional[np.ndarray]:
    """Get 3DEP elevations asynchronously in batches."""
    try:
        elevations = []
        for i in range(0, len(coords_batch), batch_size):
            batch = coords_batch[i:i + batch_size]
            # Run in executor to prevent blocking
            batch_elevs = await asyncio.get_event_loop().run_in_executor(
                None, py3dep.elevation_bycoords, batch, 4326
            )
            elevations.extend(batch_elevs)
        return np.array(elevations, dtype=np.float32)
    except Exception as e:
        logger.error(f"Error getting batch elevation data: {str(e)}")
        return None

async def process_system_async(system_id: str, session: aiohttp.ClientSession) -> Optional[gpd.GeoDataFrame]:
    """Process a single system asynchronously."""
    try:
        # Check cache first
        if system_id in profile_cache:
            return profile_cache[system_id]
        
        # Get NLD profile
        coords = await get_nld_profile_async(system_id, session)
        if coords is None:
            return None
            
        # Create GeoDataFrame
        gdf = create_geodataframe(coords, system_id)
        if gdf is None:
            return None
        
        # Get coordinates for 3DEP
        coords_list = list(zip(gdf.geometry.x, gdf.geometry.y))
        
        # Get 3DEP elevations
        dep_elevations = await get_3dep_elevations_async(coords_list)
        if dep_elevations is None:
            return None
        
        # Add elevations and calculate difference
        gdf['dep_elevation'] = dep_elevations
        gdf['difference'] = gdf['elevation'] - gdf['dep_elevation']
        
        # Cache result
        profile_cache[system_id] = gdf
        
        # Save individual system
        save_dir = Path('data/processed')
        save_dir.mkdir(parents=True, exist_ok=True)
        gdf.to_parquet(save_dir / f"levee_{system_id}.parquet")
        
        return gdf
        
    except Exception as e:
        logger.error(f"Error processing system {system_id}: {str(e)}")
        return None

def get_processed_systems() -> Set[str]:
    """Get set of system IDs that have already been processed."""
    processed_dir = Path('data/processed')
    if not processed_dir.exists():
        return set()
    
    return {p.stem.replace('levee_', '') for p in processed_dir.glob('levee_*.parquet')}

async def get_random_levees_async(n_samples: int = 10, skip_existing: bool = True, 
                                max_concurrent: int = 4) -> list:
    """Get random sample of n levee systems using async/await."""
    # Get list of all USACE system IDs
    usace_ids = get_usace_system_ids()
    
    if skip_existing:
        # Remove already processed systems
        processed = get_processed_systems()
        usace_ids = [id for id in usace_ids if id not in processed]
        logger.info(f"Found {len(processed)} existing systems, sampling from remaining {len(usace_ids)}")
    
    if not usace_ids:
        logger.error("No systems available to sample")
        return []
    
    results = []
    attempted_ids = set()
    batch_size = min(50, max(n_samples * 2, 25))  # Sample more than needed to account for failures
    
    while len(results) < n_samples and len(attempted_ids) < len(usace_ids):
        # Sample new batch of IDs that haven't been attempted
        available_ids = [id for id in usace_ids if id not in attempted_ids]
        if not available_ids:
            break
            
        sample_ids = np.random.choice(available_ids, 
                                    min(batch_size, len(available_ids)), 
                                    replace=False)
        attempted_ids.update(sample_ids)
        
        # Process systems concurrently
        connector = aiohttp.TCPConnector(limit=max_concurrent)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [process_system_async(sid, session) for sid in sample_ids]
            for task in tqdm(asyncio.as_completed(tasks), total=len(tasks),
                           desc="Processing systems"):
                try:
                    result = await task
                    if result is not None:
                        results.append(result)
                        if len(results) >= n_samples:
                            break
                except Exception as e:
                    logger.error(f"Task failed: {str(e)}")
        
        if len(results) >= n_samples:
            break
            
    return results[:n_samples]  # Ensure we return exactly n_samples

def get_random_levees(n_samples: int = 10, skip_existing: bool = True,
                     max_concurrent: int = 4) -> list:
    """Synchronous wrapper for async function."""
    return asyncio.run(get_random_levees_async(
        n_samples=n_samples,
        skip_existing=skip_existing,
        max_concurrent=max_concurrent
    ))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Sample levee systems and compare NLD vs 3DEP elevations'
    )
    parser.add_argument('-n', '--n_samples', type=int, default=10,
                      help='Number of systems to sample')
    parser.add_argument('--include_existing', action='store_true',
                      help='Include already processed systems in sampling')
    parser.add_argument('--max_concurrent', type=int, default=4,
                      help='Maximum number of concurrent connections')
    args = parser.parse_args()
    
    # Create output directories
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    # Get samples
    print(f'\nGetting {args.n_samples} random levee samples...')
    start_time = time.time()
    
    results = get_random_levees(
        n_samples=args.n_samples, 
        skip_existing=not args.include_existing,
        max_concurrent=args.max_concurrent
    )
    
    elapsed = time.time() - start_time
    print(f'Successfully processed {len(results)} systems in {elapsed:.1f} seconds')
    print(f'Average time per system: {elapsed/max(1,len(results)):.1f} seconds\n')
    
    # Show total processed
    processed = get_processed_systems()
    print(f'Total systems processed: {len(processed)}')