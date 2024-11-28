"""Functions for collecting levee elevation data from NLD and 3DEP.

This module provides the core functionality for:
1. Getting levee system IDs from the NLD API
2. Getting elevation profiles for each system
3. Getting matching 3DEP elevations
4. Saving the processed data

Typical usage:
    >>> from army_levees import get_random_levees
    >>> results = get_random_levees(n_samples=10)  # Get 10 new systems
    
The data is saved in parquet files with this structure:
    data/processed/
    └── levee_SYSTEMID.parquet

Each parquet file contains:
    - system_id: USACE system ID
    - nld_elevation: Elevation from NLD (meters)
    - dep_elevation: Elevation from 3DEP (meters)
    - difference: NLD - 3DEP (meters)
    - distance_along_track: Distance along levee (meters)
    - geometry: Point geometry (EPSG:4326)
"""

import geopandas as gpd
import numpy as np
import requests
import json
from pathlib import Path
import logging
from typing import List, Optional, Set
import py3dep

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_usace_system_ids() -> List[str]:
    """Get list of USACE system IDs from NLD API.
    
    Returns:
        List[str]: List of system IDs (e.g. ["5205000591", "5205000592"])
        Empty list if API call fails
    
    Note:
        The API endpoint returns both USACE and non-USACE systems.
        We only want USACE systems for this analysis.
    """
    url = 'https://levees.sec.usace.army.mil:443/api-local/system-categories/usace-nonusace'
    try:
        response = requests.get(url)
        if response.status_code == 200 and 'USACE' in response.json():
            return json.loads(response.json()['USACE'])
        else:
            logger.error(f"Error: 'USACE' key issue or request failed with status code: {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return []

def get_nld_profile(system_id: str) -> Optional[gpd.GeoDataFrame]:
    """Get profile data for a system ID from NLD API.
    
    Args:
        system_id: USACE system ID (e.g. "5205000591")
    
    Returns:
        GeoDataFrame with columns:
            - elevation: NLD elevation (meters)
            - distance_along_track: Distance along levee (meters)
            - geometry: Point geometry (EPSG:4326)
        None if API call fails or data is invalid
    
    Note:
        1. NLD elevations are in feet, converted to meters
        2. Coordinates are in EPSG:3857 (web mercator), converted to 4326
    """
    try:
        url = f'https://levees.sec.usace.army.mil:443/api-local/system/{system_id}/route'
        logger.info(f"Fetching profile data from: {url}")
        
        response = requests.get(url)
        if response.status_code != 200:
            logger.error(f"Error: Status code {response.status_code}")
            return None
            
        profile_data = response.json()
        if not profile_data:
            logger.error("Error: Empty response")
            return None
            
        # Extract coordinates from topology
        arcs = profile_data.get('geometry', {}).get('arcs', [[]])[0]
        if not arcs:
            logger.error("Error: No arcs found in topology")
            return None
            
        # Create points from coordinates
        # Each coord is [x, y, z, distance]
        points = []
        elevations = []
        distances = []
        
        for coord in arcs:
            if len(coord) >= 4:  # [x, y, z, distance]
                points.append([coord[0], coord[1]])
                elevations.append(coord[2])
                distances.append(coord[3])
                
        if not points:
            logger.error("Error: No valid points found")
            return None
            
        logger.info(f"Extracted {len(points)} points")
            
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            {
                'elevation': elevations,
                'distance_along_track': distances,
                'geometry': gpd.points_from_xy([p[0] for p in points], [p[1] for p in points])
            },
            crs="EPSG:3857"  # Route endpoint returns web mercator coordinates
        )
        
        # Convert NLD elevations from feet to meters
        gdf['elevation'] = gdf['elevation'] * 0.3048  # 1 foot = 0.3048 meters
        
        # Convert to 4326 for elevation sampling
        gdf = gdf.to_crs("EPSG:4326")
        
        # Check elevations
        if gdf['elevation'].eq(0).all():
            logger.error("Error: All elevations are zero")
            return None
            
        logger.info(f"Profile stats:")
        logger.info(f"Length: {gdf['distance_along_track'].max():.1f}m")
        logger.info(f"Elevation range: {gdf['elevation'].min():.1f}m to {gdf['elevation'].max():.1f}m")
        
        return gdf
            
    except Exception as e:
        logger.error(f"Error getting profile data for system {system_id}: {str(e)}")
        return None

def get_3dep_elevations(profile_gdf: gpd.GeoDataFrame) -> Optional[gpd.GeoDataFrame]:
    """Get 3DEP elevation data for points in profile_gdf.
    
    Args:
        profile_gdf: GeoDataFrame with point geometries in EPSG:4326
    
    Returns:
        Same GeoDataFrame with new column:
            - elevation_3dep: 3DEP elevation (meters)
        None if API call fails
    
    Note:
        3DEP elevations are in meters (no conversion needed)
    """
    try:
        # Get coordinates
        coords_list = list(zip(profile_gdf.geometry.x, profile_gdf.geometry.y))
        
        # Get elevations using py3dep
        logger.info("Getting 3DEP elevations...")
        elevations = py3dep.elevation_bycoords(coords_list, crs=4326)
        
        # Add to GeoDataFrame
        profile_gdf = profile_gdf.copy()
        profile_gdf['elevation_3dep'] = elevations
        
        return profile_gdf
        
    except Exception as e:
        logger.error(f"Error getting elevation data: {str(e)}")
        return None

def get_processed_systems() -> Set[str]:
    """Get set of system IDs that have already been processed.
    
    Returns:
        Set of system IDs that have parquet files in data/processed/
    """
    processed_dir = Path('data/processed')
    if not processed_dir.exists():
        return set()
    
    return {p.stem.replace('levee_', '') for p in processed_dir.glob('levee_*.parquet')}

def get_random_levees(n_samples: int = 10, skip_existing: bool = True) -> list:
    """Get random sample of n levee systems.
    
    Args:
        n_samples: Number of systems to sample
        skip_existing: If True, won't resample already processed systems
    
    Returns:
        List of GeoDataFrames, each containing:
            - system_id: USACE system ID
            - nld_elevation: NLD elevation (meters)
            - dep_elevation: 3DEP elevation (meters)
            - difference: NLD - 3DEP (meters)
            - distance_along_track: Distance along levee (meters)
            - geometry: Point geometry (EPSG:4326)
    
    Note:
        1. Data is automatically saved to data/processed/levee_SYSTEMID.parquet
        2. If skip_existing=True, only samples from systems not in data/processed/
    """
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
    
    # Randomly sample n systems
    sample_ids = np.random.choice(usace_ids, min(n_samples, len(usace_ids)))
    
    results = []
    for system_id in sample_ids:
        # Get NLD and 3DEP data
        data = analyze_levee_system(system_id)
        if data is not None:
            results.append(data)
            
    return results

def analyze_levee_system(system_id: str) -> Optional[gpd.GeoDataFrame]:
    """Get and compare elevations for one system.
    
    Args:
        system_id: USACE system ID (e.g. "5205000591")
    
    Returns:
        GeoDataFrame with columns:
            - system_id: USACE system ID
            - nld_elevation: NLD elevation (meters)
            - dep_elevation: 3DEP elevation (meters)
            - difference: NLD - 3DEP (meters)
            - distance_along_track: Distance along levee (meters)
            - geometry: Point geometry (EPSG:4326)
        None if either NLD or 3DEP data collection fails
    
    Note:
        Data is automatically saved to data/processed/levee_SYSTEMID.parquet
    """
    # Get NLD profile
    nld_data = get_nld_profile(system_id)
    if nld_data is None:
        return None
    
    # Get matching 3DEP elevations
    dep_data = get_3dep_elevations(nld_data)
    if dep_data is None:
        return None
    
    # Combine and save
    combined = gpd.GeoDataFrame({
        'system_id': system_id,
        'nld_elevation': nld_data['elevation'],
        'dep_elevation': dep_data['elevation_3dep'],
        'difference': nld_data['elevation'] - dep_data['elevation_3dep'],
        'distance_along_track': nld_data['distance_along_track'],
        'geometry': nld_data['geometry']
    })
    
    # Save individual system
    save_dir = Path('data/processed')
    save_dir.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(save_dir / f"levee_{system_id}.parquet")
    
    return combined

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Sample levee systems and compare NLD vs 3DEP elevations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__  # Use module docstring as extended help
    )
    parser.add_argument('-n', '--n_samples', type=int, default=10,
                      help='Number of systems to sample')
    parser.add_argument('--include_existing', action='store_true',
                      help='Include already processed systems in sampling')
    args = parser.parse_args()
    
    # Create output directories
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    # Get samples
    print(f'\nGetting {args.n_samples} random levee samples...')
    results = get_random_levees(
        n_samples=args.n_samples, 
        skip_existing=not args.include_existing
    )
    print(f'Successfully processed {len(results)} systems\n')
    
    # Show total processed
    processed = get_processed_systems()
    print(f'Total systems processed: {len(processed)}')