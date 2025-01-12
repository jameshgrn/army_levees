"""Command-line interface for visualizing levee elevation data."""

import argparse
import logging
import random
from pathlib import Path
from typing import Literal, Optional

import folium
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import Normalize, to_hex
from tqdm import tqdm

from army_levees.core.visualize.individual import (
    _print_segment_stats, diagnose_elevation_differences,
    plot_elevation_profile)
from army_levees.core.visualize.interactive import create_interactive_dashboard

logger = logging.getLogger(__name__)


def plot_levee_profile(
    gdf: gpd.GeoDataFrame, title: Optional[str] = None, was_converted: bool = False
) -> None:
    """Plot elevation profile comparing NLD and 3DEP data.

    Args:
        gdf: GeoDataFrame with columns:
            - distance_along_track: Distance along levee
            - elevation: NLD elevation
            - dep_elevation: 3DEP elevation
        title: Optional plot title
        was_converted: Whether NLD data was converted from feet to meters
    """
    plt.figure(figsize=(12, 6))

    # Plot both elevation profiles
    plt.plot(
        gdf.distance_along_track, gdf.elevation, "b-", label="NLD Elevation", alpha=0.7
    )
    plt.plot(
        gdf.distance_along_track,
        gdf.dep_elevation,
        "r-",
        label="3DEP Elevation",
        alpha=0.7,
    )

    # Plot elevation differences
    plt.fill_between(
        gdf.distance_along_track,
        gdf.elevation,
        gdf.dep_elevation,
        alpha=0.2,
        color="gray",
        label="Difference",
    )

    plt.grid(True, alpha=0.3)
    plt.xlabel("Distance Along Levee (m)")
    plt.ylabel("Elevation (m)")

    # Add conversion note to title if applicable
    plot_title = title or ""
    if was_converted:
        plot_title = (
            f"{plot_title}\n(NLD converted from feet to meters)"
            if plot_title
            else "NLD converted from feet to meters"
        )
    if plot_title:
        plt.title(plot_title)

    plt.legend()


def plot_levee_map(
    gdf: gpd.GeoDataFrame, background: Literal["satellite", "terrain"] = "terrain"
) -> Optional[folium.Map]:
    """Create interactive map of levee with elevation data."""
    try:
        # Create map centered on levee
        center = [gdf.geometry.y.mean(), gdf.geometry.x.mean()]
        m = folium.Map(
            location=center, zoom_start=13, tiles=f"Stamen {background.title()}"
        )

        # Add elevation points colored by difference
        diffs = gdf.elevation - gdf.dep_elevation
        norm = Normalize(vmin=diffs.min(), vmax=diffs.max())
        cmap = colormaps["RdYlBu"]

        for idx, row in gdf.iterrows():
            diff = row.elevation - row.dep_elevation
            color = cmap(norm(diff))
            hex_color = to_hex(color)
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5,
                color=hex_color,
                popup=f"NLD: {row.elevation:.1f}m<br>"
                f"3DEP: {row.dep_elevation:.1f}m<br>"
                f"Diff: {diff:.1f}m",
            ).add_to(m)

        return m

    except Exception as e:
        logger.error(f"Error creating map: {str(e)}")
        return None


def plot_levee_stats(gdf: gpd.GeoDataFrame) -> None:
    """Plot statistical summary of levee data.

    Shows:
    - Elevation distributions
    - Difference histogram
    - Spatial patterns
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Elevation distributions
    gdf.elevation.hist(ax=axes[0, 0], bins=30, alpha=0.5, label="NLD")
    gdf.dep_elevation.hist(ax=axes[0, 0], bins=30, alpha=0.5, label="3DEP")
    axes[0, 0].set_title("Elevation Distributions")
    axes[0, 0].legend()

    # Difference histogram
    diffs = gdf.elevation - gdf.dep_elevation
    diffs.hist(ax=axes[0, 1], bins=30)
    axes[0, 1].set_title("Elevation Differences")

    # Spatial patterns
    scatter = axes[1, 0].scatter(gdf.geometry.x, gdf.geometry.y, c=diffs, cmap="RdYlBu")
    axes[1, 0].set_title("Spatial Pattern of Differences")
    plt.colorbar(scatter, ax=axes[1, 0])

    # Distance vs difference
    axes[1, 1].scatter(gdf.distance_along_track, diffs)
    axes[1, 1].set_title("Differences vs Distance")

    plt.tight_layout()


def get_segments_for_system(
    system_id: str, data_dir: Path | str
) -> list[gpd.GeoDataFrame]:
    """Get all segments for a given system."""
    segments = []
    data_dir = Path(data_dir)  # Convert to Path if string
    segment_pattern = f"levee_{system_id}_segment_*.parquet"
    segment_files = list(data_dir.glob(segment_pattern))

    if not segment_files:
        logger.warning(f"Could not find profile data for system {system_id}")
        return []

    for file in sorted(segment_files):  # Sort to keep segments in order
        try:
            gdf = gpd.read_parquet(file)
            segments.append(gdf)
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")

    return segments


def plot_system(
    system_id: str,
    data_dir: Path | str = "data/segments",
    save_dir: Path | str = "plots",
) -> None:
    """Plot all segments for a system on one plot."""
    data_dir = Path(data_dir)
    save_dir = Path(save_dir)

    # Check if system was converted
    conversion_file = data_dir / "unit_converted_systems.txt"
    was_converted = False
    if conversion_file.exists():
        with open(conversion_file) as f:
            converted_systems = f.read().splitlines()
            was_converted = system_id in converted_systems

    segments = get_segments_for_system(system_id, data_dir)
    if not segments:
        return

    # Create one plot with all segments
    plt.figure(figsize=(15, 8))

    # Track cumulative distance for proper spacing
    cumulative_distance = 0

    # Plot each segment
    for segment in segments:
        # Adjust distances to account for gaps between segments
        adjusted_distances = segment.distance_along_track + cumulative_distance

        # Plot NLD in blue, 3DEP in red
        plt.plot(adjusted_distances, segment.elevation, "b-", alpha=0.7, label="NLD")
        plt.plot(
            adjusted_distances, segment.dep_elevation, "r-", alpha=0.7, label="3DEP"
        )

        # Plot differences in gray
        plt.fill_between(
            adjusted_distances,
            segment.elevation,
            segment.dep_elevation,
            alpha=0.1,
            color="gray",
        )

        # Update cumulative distance for next segment
        # Add a small gap between segments (e.g., 100m)
        cumulative_distance = adjusted_distances.max() + 100

    plt.grid(True, alpha=0.3)
    plt.xlabel("Distance Along Levee (m)")
    plt.ylabel("Elevation (m)")

    # Add conversion note to title if applicable
    title = f"System {system_id}"
    if was_converted:
        title += "\n(NLD converted from feet to meters)"
    plt.title(title)

    # Only show legend once
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(
        by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc="upper left"
    )

    # Save with extra width for legend
    plt.savefig(save_dir / f"system_{system_id}.png", bbox_inches="tight", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Create elevation comparison plots for levee systems"
    )

    # System ID selection group
    id_group = parser.add_mutually_exclusive_group()
    id_group.add_argument("system_id", nargs="?", help="USACE system ID to plot")
    id_group.add_argument(
        "-r", "--random", action="store_true", help="Use a random levee system"
    )
    id_group.add_argument(
        "-a", "--all", action="store_true", help="Process all available systems"
    )

    # Action group
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "-p", "--plot", action="store_true", help="Create plots for the system(s)"
    )
    action_group.add_argument(
        "-s",
        "--summary",
        action="store_true",
        help="Create summary plots for all processed levees",
    )
    action_group.add_argument(
        "-d", "--diagnose", action="store_true", help="Run diagnostics on the system(s)"
    )
    action_group.add_argument(
        "-i", "--interactive", action="store_true", help="Create interactive dashboard"
    )

    # Data source selection
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/segments"),
        help="Directory containing levee data (default: data/segments)",
    )
    parser.add_argument(
        "--raw-data",
        action="store_true",
        help="Use raw data from data/processed instead of filtered segments",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("plots"),
        help="Directory to save plots (default: plots)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the interactive dashboard in browser",
    )

    args = parser.parse_args()
    data_dir = Path("data/processed") if args.raw_data else args.data_dir

    # Get unique system IDs (strip off _segment_XX from filenames)
    system_ids = set()
    for file in Path(data_dir).glob("levee_*_segment_*.parquet"):
        system_id = file.stem.split("_segment_")[0].replace("levee_", "")
        system_ids.add(system_id)

    if not system_ids:
        logger.error(f"No levee data found in {data_dir}")
        return

    logger.info(f"Found {len(system_ids)} systems")

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Execute requested action
    if args.plot:
        for system_id in tqdm(sorted(system_ids), desc="Creating plots"):
            plot_system(system_id, data_dir, save_dir)
    elif args.diagnose:
        for sys_id in sorted(system_ids):
            segments = get_segments_for_system(sys_id, data_dir)
            if segments:
                for i, segment in enumerate(segments):
                    print(f"\nAnalyzing {sys_id} - Segment {i}:")
                    _print_segment_stats(segment)
    elif args.summary:
        # Create summary plots
        from army_levees.core.visualize.summary import (
            plot_elevation_differences, plot_point_counts,
            plot_segment_lengths)

        # Get all segments
        all_segments = []
        for system_id in tqdm(sorted(system_ids), desc="Loading segments"):
            segments = get_segments_for_system(system_id, data_dir)
            all_segments.extend(segments)

        if all_segments:
            # Create summary plots
            plot_elevation_differences(all_segments, save_dir)
            plot_segment_lengths(all_segments, save_dir)
            plot_point_counts(all_segments, save_dir)
            logger.info(f"Saved summary plots to {save_dir}")
    elif args.interactive:
        # Create interactive dashboard
        dashboard = create_interactive_dashboard(
            save_path=save_dir / "levee_dashboard.html",
            data_dir=data_dir,
            raw_data=args.raw_data,
            show=args.show,
        )
        if dashboard:
            logger.info(
                f"Saved interactive dashboard to {save_dir}/levee_dashboard.html"
            )
            if args.show:
                logger.info("Opening dashboard in browser...")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()
