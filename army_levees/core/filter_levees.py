"""Filter existing levee data to remove problematic profiles."""

import logging
from pathlib import Path
import geopandas as gpd
import numpy as np
from tqdm import tqdm
from typing import Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def extract_valid_segments(gdf: gpd.GeoDataFrame,
                         zero_threshold: float = 0.01,
                         min_points: int = 3,
                         max_elev_diff: float = 50.0) -> list[gpd.GeoDataFrame]:
    """Extract valid segments from a levee profile.
    
    Returns segments where we have valid data in both NLD and 3DEP.
    A valid segment must have:
    1. At least min_points with data in both datasets
    2. Non-zero/non-missing elevations (> zero_threshold)
    3. Elevation differences less than max_elev_diff
    """
    try:
        # Sort by distance first
        gdf = gdf.sort_values('distance_along_track').copy()
        
        # Find near-zero points and log them
        near_zero_mask = (
            (np.abs(gdf['elevation']) < zero_threshold) | 
            (np.abs(gdf['dep_elevation']) < zero_threshold)
        )
        if near_zero_mask.any():
            near_zero_points = gdf[near_zero_mask]
            logger.info(f"Found {len(near_zero_points)} points with near-zero elevations:")
            logger.info(f"NLD range: {near_zero_points['elevation'].min():.6f}m to {near_zero_points['elevation'].max():.6f}m")
            logger.info(f"3DEP range: {near_zero_points['dep_elevation'].min():.6f}m to {near_zero_points['dep_elevation'].max():.6f}m")
        
        # Find points where both datasets have valid data
        valid_mask = (
            (np.abs(gdf['elevation']) >= zero_threshold) & 
            (np.abs(gdf['dep_elevation']) >= zero_threshold) & 
            gdf['elevation'].notna() &
            gdf['dep_elevation'].notna()
        )
        
        # Get indices of valid points
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) < min_points:
            logger.info(f"Only {len(valid_indices)} points with valid data in both datasets")
            return []
            
        # Check elevation differences
        elev_diff = np.abs(gdf.loc[valid_mask, 'elevation'] - gdf.loc[valid_mask, 'dep_elevation'])
        if elev_diff.max() > max_elev_diff:
            logger.info(f"Maximum elevation difference too large: {elev_diff.max():.1f}m")
            return []
            
        # Find gaps in valid data
        gaps = np.where(np.diff(valid_indices) > 1)[0]
        
        # Split into segments
        segments = []
        start_idx = 0
        
        # Process each segment
        for gap_idx in gaps:
            segment_indices = valid_indices[start_idx:gap_idx + 1]
            if len(segment_indices) >= min_points:
                segment_data = gdf.iloc[segment_indices].copy()
                # Reset distance to start at 0
                segment_data['distance_along_track'] -= segment_data['distance_along_track'].min()
                segments.append(segment_data)
            start_idx = gap_idx + 1
        
        # Don't forget last segment
        if start_idx < len(valid_indices):
            segment_indices = valid_indices[start_idx:]
            if len(segment_indices) >= min_points:
                segment_data = gdf.iloc[segment_indices].copy()
                segment_data['distance_along_track'] -= segment_data['distance_along_track'].min()
                segments.append(segment_data)
        
        if segments:
            total_points = sum(len(s) for s in segments)
            logger.info(f"Found {len(segments)} segments with {total_points} valid points")
        else:
            logger.info("No valid segments found")
            
        return segments
        
    except Exception as e:
        logger.error(f"Error extracting segments: {str(e)}")
        return []

def process_all_levees(
    input_dir: Path | str = "data/processed",
    output_dir: Path | str = "data/segments",
    **extract_kwargs
) -> None:
    """Process all levee profiles and save valid segments."""
    # Convert paths
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track statistics
    stats = {
        "total_profiles": 0,
        "profiles_with_segments": 0,
        "total_segments": 0,
        "total_valid_points": 0,
        "rejected": {
            "no_3dep_data": 0,
            "too_few_points": 0,
            "large_diff": 0,
            "errors": 0
        }
    }
    
    # Process each file
    for file in tqdm(list(input_dir.glob("levee_*.parquet")), desc="Processing levees"):
        try:
            system_id = file.stem.replace("levee_", "")
            gdf = gpd.read_parquet(file)
            
            # Log initial stats
            logger.info(f"\nProcessing {system_id}:")
            logger.info(f"  Total points: {len(gdf)}")
            
            # Extract valid segments
            segments = extract_valid_segments(gdf, **extract_kwargs)
            
            if segments:
                # Save valid segments
                for i, segment in enumerate(segments):
                    segment_file = output_dir / f"levee_{system_id}_segment_{i:02d}.parquet"
                    segment.to_parquet(segment_file)
                
                stats["profiles_with_segments"] += 1
                stats["total_segments"] += len(segments)
                stats["total_valid_points"] += sum(len(s) for s in segments)
                
                # Log success
                total_points = sum(len(s) for s in segments)
                logger.info(f"  Success: {len(segments)} segments with {total_points} valid points")
                logger.info(f"  Valid point percentage: {total_points/len(gdf):.1%}")
            
        except Exception as e:
            logger.error(f"Error processing {file.name}: {str(e)}")
            stats["rejected"]["errors"] += 1
            
        stats["total_profiles"] += 1
            
    # Log final statistics
    logger.info("\nProcessing complete!")
    logger.info(f"Total profiles processed: {stats['total_profiles']}")
    logger.info(f"Profiles with valid segments: {stats['profiles_with_segments']}")
    logger.info(f"Total segments extracted: {stats['total_segments']}")
    logger.info(f"Total valid points: {stats['total_valid_points']}")
    logger.info(f"Average segments per profile: {stats['total_segments']/max(1,stats['profiles_with_segments']):.1f}")
    logger.info("\nRejection reasons:")
    logger.info(f"  No 3DEP data: {stats['rejected']['no_3dep_data']}")
    logger.info(f"  Too few points: {stats['rejected']['too_few_points']}")
    logger.info(f"  Large elevation differences: {stats['rejected']['large_diff']}")
    logger.info(f"  Errors: {stats['rejected']['errors']}")
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Filter problematic levee profiles")
    parser.add_argument(
        "--input-dir", 
        type=str,
        default="data/processed",
        help="Directory containing parquet files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/segments",
        help="Directory to save filtered files"
    )
    parser.add_argument(
        "--zero-threshold",
        type=float,
        default=0.01,
        help="Threshold for considering elevation as zero"
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=3,
        help="Minimum number of points required per segment"
    )
    parser.add_argument(
        "--max-elev-diff",
        type=float,
        default=50.0,
        help="Maximum allowed elevation difference between NLD and 3DEP"
    )
    
    args = parser.parse_args()
    
    process_all_levees(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        zero_threshold=args.zero_threshold,
        min_points=args.min_points,
        max_elev_diff=args.max_elev_diff
    ) 