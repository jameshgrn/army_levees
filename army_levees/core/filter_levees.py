"""Filter existing levee data to remove problematic profiles."""

import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_valid_segments(
    gdf: gpd.GeoDataFrame,
    zero_threshold: float = 0.01,
    min_points: int = 3,
    max_elev_diff: float = 50.0,
    min_elevation: float = -100,
    max_elevation: float = 5000,
    ft_to_m_ratio_threshold: float = 0.1,
) -> list[gpd.GeoDataFrame]:
    """Extract valid segments from a levee profile.

    Args:
        gdf: GeoDataFrame with elevation data
        zero_threshold: Values below this are considered zero
        min_points: Minimum points needed for valid segment
        max_elev_diff: Maximum allowed elevation difference
        min_elevation: Minimum valid elevation
        max_elevation: Maximum valid elevation
        ft_to_m_ratio_threshold: How close ratio needs to be to 3.28084 to be considered feet
    """
    try:
        # Sort by distance first
        gdf = gdf.sort_values("distance_along_track").copy()

        # Check for potential feet-to-meters conversion issues
        mask = (gdf["elevation"] != 0) & (gdf["dep_elevation"] != 0)
        if mask.any():
            ratios = gdf.loc[mask, "elevation"] / gdf.loc[mask, "dep_elevation"]
            mean_ratio = ratios.mean()

            # If ratio is close to feet-to-meters conversion factor (3.28084)
            if abs(mean_ratio - 3.28084) < ft_to_m_ratio_threshold:
                logger.info(
                    "Detected likely feet-to-meters conversion issue - converting NLD to meters"
                )
                gdf["elevation"] = gdf["elevation"] / 3.28084

        # Track filtering statistics
        stats = {
            "total_points": len(gdf),
            "missing_data": {"nld": 0, "dep": 0, "both": 0},
            "near_zero": 0,
            "unreasonable_values": 0,
            "large_diff": 0,
            "valid_points": 0,
        }

        # Check for missing data
        nld_missing = gdf["elevation"].isna()
        dep_missing = gdf["dep_elevation"].isna()
        both_missing = nld_missing & dep_missing

        # Check for invalid values
        nld_invalid = (
            (np.abs(gdf["elevation"]) < zero_threshold)
            | (gdf["elevation"] < min_elevation)  # Near zero
            | (gdf["elevation"] > max_elevation)  # Too low  # Too high
        )
        dep_invalid = (
            (np.abs(gdf["dep_elevation"]) < zero_threshold)
            | (gdf["dep_elevation"] < min_elevation)
            | (gdf["dep_elevation"] > max_elevation)
        )

        # Check if ALL points in either dataset are near zero
        if (np.abs(gdf["elevation"]) < zero_threshold).all() or (
            np.abs(gdf["dep_elevation"]) < zero_threshold
        ).all():
            logger.info("Rejecting segment - all points near zero in one dataset")
            return []

        # Check elevation differences
        diff_mask = np.abs(gdf["elevation"] - gdf["dep_elevation"]) > max_elev_diff

        # Combined invalid mask
        invalid_mask = (
            nld_missing
            | dep_missing
            | nld_invalid  # Missing data
            | dep_invalid
            | diff_mask  # Invalid values  # Large differences
        )

        # Update statistics
        stats["missing_data"]["nld"] = nld_missing.sum()
        stats["missing_data"]["dep"] = dep_missing.sum()
        stats["missing_data"]["both"] = both_missing.sum()
        stats["near_zero"] = (nld_invalid | dep_invalid).sum()
        stats["large_diff"] = diff_mask.sum()

        # Get valid points
        valid_mask = ~invalid_mask
        valid_indices = np.where(valid_mask)[0]
        stats["valid_points"] = len(valid_indices)

        if len(valid_indices) < min_points:
            logger.info(
                f"\nRejecting segment - too few valid points ({len(valid_indices)} < {min_points})"
            )
            logger.info(str(stats))
            return []

        # Find gaps in valid data
        gaps = np.where(np.diff(valid_indices) > 1)[0]

        # Split into segments
        segments = []
        start_idx = 0

        for gap_idx in gaps:
            segment_indices = valid_indices[start_idx : gap_idx + 1]
            if len(segment_indices) >= min_points:
                segment_data = gdf.iloc[segment_indices].copy()
                segment_data["distance_along_track"] -= segment_data[
                    "distance_along_track"
                ].min()
                segments.append(segment_data)
            start_idx = gap_idx + 1

        # Don't forget last segment
        if start_idx < len(valid_indices):
            segment_indices = valid_indices[start_idx:]
            if len(segment_indices) >= min_points:
                segment_data = gdf.iloc[segment_indices].copy()
                segment_data["distance_along_track"] -= segment_data[
                    "distance_along_track"
                ].min()
                segments.append(segment_data)

        logger.info(f"\nExtracted {len(segments)} valid segments")
        logger.info(str(stats))

        return segments

    except Exception as e:
        logger.error(f"Error extracting segments: {str(e)}")
        return []


def process_all_levees(
    input_dir: Path | str = "data/processed",
    output_dir: Path | str = "data/segments",
    **extract_kwargs,
) -> None:
    """Process all levee profiles and save valid segments."""
    # Convert paths
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track filtering statistics
    stats = {
        "total_systems": 0,
        "total_points": 0,
        "points_rejected": {
            "missing_data": {"nld": 0, "dep": 0, "both": 0},
            "near_zero": 0,
            "large_diff": 0,
        },
        "systems_rejected": {"no_valid_points": 0, "too_few_points": 0, "errors": 0},
        "unit_conversions": {
            "count": 0,
            "systems": [],  # Track which systems needed conversion
            "mean_ratio": [],  # Track the ratios we found
            "points_affected": 0,  # Track how many points were converted
        },
        "valid_systems": 0,
        "valid_segments": 0,
        "valid_points": 0,
    }

    # Process each file
    files = list(input_dir.glob("levee_*.parquet"))
    for file in tqdm(files, desc="Processing levees"):
        try:
            stats["total_systems"] += 1
            system_id = file.stem.replace("levee_", "")
            gdf = gpd.read_parquet(file)

            # Check if system needs conversion before filtering
            mask = (gdf["elevation"] != 0) & (gdf["dep_elevation"] != 0)
            if mask.any():
                ratios = gdf.loc[mask, "elevation"] / gdf.loc[mask, "dep_elevation"]
                mean_ratio = ratios.mean()
                if abs(mean_ratio - 3.28084) < extract_kwargs.get(
                    "ft_to_m_ratio_threshold", 0.1
                ):
                    stats["unit_conversions"]["count"] += 1
                    stats["unit_conversions"]["systems"].append(system_id)
                    stats["unit_conversions"]["mean_ratio"].append(mean_ratio)
                    stats["unit_conversions"]["points_affected"] += len(gdf)

            stats["total_points"] += len(gdf)

            # Track points rejected by each condition with more detail
            nld_missing = gdf["elevation"].isna()
            dep_missing = gdf["dep_elevation"].isna()
            both_missing = nld_missing & dep_missing

            stats["points_rejected"]["missing_data"]["nld"] += nld_missing.sum()
            stats["points_rejected"]["missing_data"]["dep"] += dep_missing.sum()
            stats["points_rejected"]["missing_data"]["both"] += both_missing.sum()

            near_zero_mask = (
                np.abs(gdf["elevation"]) < extract_kwargs.get("zero_threshold", 0.01)
            ) | (
                np.abs(gdf["dep_elevation"])
                < extract_kwargs.get("zero_threshold", 0.01)
            )
            stats["points_rejected"]["near_zero"] += near_zero_mask.sum()

            diff_mask = np.abs(
                gdf["elevation"] - gdf["dep_elevation"]
            ) > extract_kwargs.get("max_elev_diff", 50.0)
            stats["points_rejected"]["large_diff"] += diff_mask.sum()

            # Extract segments
            segments = extract_valid_segments(gdf, **extract_kwargs)

            if segments:
                stats["valid_systems"] += 1
                stats["valid_segments"] += len(segments)
                total_valid_points = sum(len(s) for s in segments)
                stats["valid_points"] += total_valid_points

                # Save valid segments
                for i, segment in enumerate(segments):
                    segment_file = (
                        output_dir / f"levee_{system_id}_segment_{i:02d}.parquet"
                    )
                    segment.to_parquet(segment_file)
            else:
                stats["systems_rejected"]["no_valid_points"] += 1

        except Exception as e:
            logger.error(f"Error processing {file.name}: {str(e)}")
            stats["systems_rejected"]["errors"] += 1

    # Log final statistics with more detail
    logger.info("\nFinal Processing Summary:")
    logger.info(f"Systems processed: {stats['total_systems']}")
    logger.info(f"Total points: {stats['total_points']}")
    logger.info("\nPoints rejected:")

    # Calculate total missing (accounting for overlap)
    total_missing = (
        stats["points_rejected"]["missing_data"]["nld"]
        + stats["points_rejected"]["missing_data"]["dep"]
        - stats["points_rejected"]["missing_data"]["both"]
    )

    logger.info(
        f"  missing_data: {total_missing:,} ({total_missing/stats['total_points']*100:.1f}%)"
    )
    logger.info(
        f"    NLD missing: {stats['points_rejected']['missing_data']['nld']:,} "
        f"({stats['points_rejected']['missing_data']['nld']/stats['total_points']*100:.1f}%)"
    )
    logger.info(
        f"    3DEP missing: {stats['points_rejected']['missing_data']['dep']:,} "
        f"({stats['points_rejected']['missing_data']['dep']/stats['total_points']*100:.1f}%)"
    )
    logger.info(
        f"    Both missing: {stats['points_rejected']['missing_data']['both']:,} "
        f"({stats['points_rejected']['missing_data']['both']/stats['total_points']*100:.1f}%)"
    )

    logger.info("\nSystems rejected:")
    for reason, count in stats["systems_rejected"].items():
        pct = count / stats["total_systems"] * 100
        logger.info(f"  {reason}: {count} ({pct:.1f}%)")

    logger.info("\nValid data:")
    logger.info(
        f"  Systems: {stats['valid_systems']} ({stats['valid_systems']/stats['total_systems']*100:.1f}%)"
    )
    logger.info(
        f"  Segments: {stats['valid_segments']} (avg {stats['valid_segments']/max(1,stats['valid_systems']):.1f} per valid system)"
    )
    logger.info(
        f"  Points: {stats['valid_points']} ({stats['valid_points']/stats['total_points']*100:.1f}%)"
    )

    # Add detailed unit conversion stats to output
    logger.info("\nUnit conversions:")
    pct = stats["unit_conversions"]["count"] / stats["total_systems"] * 100
    logger.info(
        f"  Systems needing conversion: {stats['unit_conversions']['count']} ({pct:.1f}%)"
    )
    if stats["unit_conversions"]["count"] > 0:
        avg_ratio = sum(stats["unit_conversions"]["mean_ratio"]) / len(
            stats["unit_conversions"]["mean_ratio"]
        )
        logger.info(f"  Average ratio found: {avg_ratio:.4f}")
        points_pct = (
            stats["unit_conversions"]["points_affected"] / stats["total_points"] * 100
        )
        logger.info(
            f"  Points affected: {stats['unit_conversions']['points_affected']} ({points_pct:.1f}%)"
        )

        # Save list of converted systems
        conversion_file = output_dir / "unit_converted_systems.txt"
        with open(conversion_file, "w") as f:
            f.write("\n".join(stats["unit_conversions"]["systems"]))
        logger.info(f"  Saved list of converted systems to {conversion_file}")


def filter_levees():
    """Filter levee data for quality."""
    # Load all processed levee data
    all_data = []
    for file in Path("data/processed").glob("levee_*.parquet"):
        gdf = gpd.read_parquet(file)
        all_data.append(gdf)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter problematic levee profiles")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/processed",
        help="Directory containing parquet files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/segments",
        help="Directory to save filtered files",
    )
    parser.add_argument(
        "--zero-threshold",
        type=float,
        default=0.01,
        help="Threshold for considering elevation as zero",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=3,
        help="Minimum number of points required per segment",
    )
    parser.add_argument(
        "--max-elev-diff",
        type=float,
        default=50.0,
        help="Maximum allowed elevation difference between NLD and 3DEP",
    )

    args = parser.parse_args()

    process_all_levees(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        zero_threshold=args.zero_threshold,
        min_points=args.min_points,
        max_elev_diff=args.max_elev_diff,
    )
