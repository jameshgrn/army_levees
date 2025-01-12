"""Tools for investigating discrepancies between NLD and 3DEP elevation data."""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Point
from tqdm import tqdm

from army_levees.core.sample_levees import (ProcessingStats,
                                            create_geodataframe,
                                            get_3dep_elevations_async,
                                            get_nld_profile_async,
                                            get_usace_system_ids)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def investigate_system(system_id: str) -> Optional[gpd.GeoDataFrame]:
    """Detailed investigation of a specific levee system."""
    try:
        # Load existing data
        parquet_path = Path("data/processed") / f"levee_{system_id}.parquet"
        if not parquet_path.exists():
            logger.error(f"No existing data found for system {system_id}")
            return None

        gdf = gpd.read_parquet(parquet_path)
        if not isinstance(gdf, gpd.GeoDataFrame):
            gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=gdf.crs)

        # Add missing data analysis here
        logger.info("\nMissing Data Analysis:")
        nld_missing = gdf["elevation"].isna()
        dep_missing = gdf["dep_elevation"].isna()
        both_missing = nld_missing & dep_missing

        logger.info(f"Total points: {len(gdf)}")
        logger.info(f"NLD missing: {nld_missing.sum()} ({nld_missing.mean():.1%})")
        logger.info(f"3DEP missing: {dep_missing.sum()} ({dep_missing.mean():.1%})")
        logger.info(f"Both missing: {both_missing.sum()} ({both_missing.mean():.1%})")

        if dep_missing.any():
            missing_points = gdf[dep_missing]
            logger.info("\nPoints with missing 3DEP data:")
            logger.info(f"  Count: {len(missing_points)}")
            logger.info(
                f"  NLD elevation range: {missing_points['elevation'].min():.1f}m to {missing_points['elevation'].max():.1f}m"
            )
            logger.info(f"  Geographic bounds:")
            logger.info(
                f"    Longitude: {missing_points.geometry.x.min():.2f} to {missing_points.geometry.x.max():.2f}"
            )
            logger.info(
                f"    Latitude: {missing_points.geometry.y.min():.2f} to {missing_points.geometry.y.max():.2f}"
            )

        # Filter out rows where either elevation is null
        gdf = gdf.dropna(subset=["elevation", "dep_elevation"])

        if len(gdf) == 0:
            logger.error("No overlapping elevation data found")
            return None

        # Recalculate distance along track after filtering
        gdf["distance_along_track"] = gdf.geometry.distance(
            gdf.geometry.iloc[0]
        ).cumsum()

        # Analyze elevation differences
        diffs = gdf["difference"]
        logger.info("\nElevation Difference Analysis:")
        logger.info(f"Number of comparison points: {len(gdf)}")
        logger.info(f"Mean difference: {diffs.mean():.2f}m")
        logger.info(f"Std deviation: {diffs.std():.2f}m")
        logger.info(f"Max difference: {diffs.max():.2f}m")
        logger.info(f"Min difference: {diffs.min():.2f}m")

        # Find locations of large differences
        large_diffs = gdf[abs(diffs) > 2.0]
        if len(large_diffs) > 0:
            logger.info("\nLocations with large differences (>2m):")
            for idx, row in large_diffs.iterrows():
                logger.info(
                    f"Distance: {row['distance_along_track']:.0f}m, "
                    f"NLD: {row['elevation']:.2f}m, "
                    f"3DEP: {row['dep_elevation']:.2f}m, "
                    f"Diff: {row['difference']:.2f}m"
                )

        # Create interactive map
        gdf_4326 = gdf.to_crs(4326)
        center_lat = gdf_4326.geometry.y.mean()
        center_lon = gdf_4326.geometry.x.mean()

        # Create map with satellite imagery
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=14,
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery",
        )

        # Add OpenStreetMap as a layer
        folium.TileLayer("OpenStreetMap").add_to(m)

        # Add line of levee system
        points = [(p.y, p.x) for p in gdf_4326.geometry]
        line = LineString(points)
        style = {"color": "blue", "weight": 3, "opacity": 0.8}
        folium.GeoJson(
            line.__geo_interface__,
            name="Levee Alignment",
            style_function=lambda x: style,
        ).add_to(m)

        # Add markers for large differences
        if len(large_diffs) > 0:
            large_diffs_4326 = large_diffs.to_crs(4326)
            for _, row in large_diffs_4326.iterrows():
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=8,
                    color="red",
                    fill=True,
                    popup=f"Diff: {row['difference']:.1f}m<br>"
                    f"NLD: {row['elevation']:.1f}m<br>"
                    f"3DEP: {row['dep_elevation']:.1f}m<br>"
                    f"Distance: {row['distance_along_track']:.0f}m",
                ).add_to(m)

        # Add layer control
        folium.LayerControl().add_to(m)

        # Save map
        output_dir = Path("investigation")
        output_dir.mkdir(exist_ok=True)
        map_path = output_dir / f"system_{system_id}_map.html"
        m.save(str(map_path))
        logger.info(f"Map saved to {map_path}")

        # Create elevation profile plot
        plt.figure(figsize=(15, 10))

        # Convert series to numpy arrays for plotting
        distances = gdf["distance_along_track"].to_numpy()
        nld_elevs = gdf["elevation"].to_numpy()
        dep_elevs = gdf["dep_elevation"].to_numpy()
        dep_max_elevs = gdf["dep_elevation_max"].to_numpy()
        differences = gdf["difference"].to_numpy()

        # Plot elevations
        plt.subplot(2, 1, 1)
        plt.plot(distances, nld_elevs, "b-", label="NLD")
        plt.plot(distances, dep_elevs, "r-", label="3DEP")
        plt.plot(distances, dep_max_elevs, "g--", label="3DEP (max)")
        plt.grid(True)
        plt.legend()
        plt.title(f"Elevation Profiles - System {system_id}")
        plt.ylabel("Elevation (m)")

        # Plot differences
        plt.subplot(2, 1, 2)
        plt.plot(distances, differences, "k-")
        plt.axhline(y=0, color="r", linestyle=":")
        plt.axhline(
            y=float(differences.mean()), color="g", linestyle="--", label="Mean"
        )
        plt.fill_between(
            distances,
            float(differences.mean() - differences.std()),
            float(differences.mean() + differences.std()),
            alpha=0.2,
            color="g",
            label="±1 Std Dev",
        )
        plt.grid(True)
        plt.legend()
        plt.title("Elevation Differences (NLD - 3DEP)")
        plt.xlabel("Distance Along Levee (m)")
        plt.ylabel("Difference (m)")

        # Save plot
        plot_path = output_dir / f"system_{system_id}_profile.png"
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Profile plot saved to {plot_path}")

        # Add elevation change consistency analysis
        changes = gdf["dep_elevation"] - gdf["elevation"]
        mean_change = changes.mean()
        points_agreeing = (changes * mean_change > 0).sum()
        consistency = (points_agreeing / len(changes)) * 100

        logger.info("\nChange Consistency Analysis:")
        logger.info(f"Mean change: {mean_change:.2f}m")
        logger.info(f"Change consistency: {consistency:.1f}%")

        # Categorize the change
        if abs(mean_change) <= 0.5:
            category = "Stable"
        elif mean_change < -0.5:
            category = "Degradation"
        else:
            category = "Aggradation"
        logger.info(f"Change category: {category}")

        # Add elevation-based analysis
        mean_elev = gdf["elevation"].mean()
        if mean_elev > 1000:
            logger.info("\nNote: This is a high-elevation system (>1000m)")
            logger.info("These systems tend to show more variable changes")

        return gdf

    except Exception as e:
        logger.error(f"Error investigating system: {str(e)}")
        return None


async def investigate_systems(
    system_ids: List[str], output_dir: Path = Path("investigation")
) -> None:
    """Investigate multiple levee systems.

    Args:
        system_ids: List of system IDs to investigate
        output_dir: Directory to save outputs
    """
    output_dir.mkdir(exist_ok=True)

    # Individual system analysis
    for system_id in tqdm(system_ids, desc="Analyzing systems"):
        gdf = await investigate_system(system_id)
        if gdf is not None:
            logger.info(f"Successfully analyzed system {system_id}")
        else:
            logger.warning(f"Failed to analyze system {system_id}")


async def main():
    """Investigate systems with high 3DEP failure rates."""
    # Let's look at systems with most missing data
    systems = [
        "4705000014",  # Pick a system with lots of missing 3DEP
        "6005000020",  # Another problematic one
    ]

    for system_id in systems:
        logger.info(f"\nInvestigating system {system_id}")
        gdf = await investigate_system(system_id)

        if gdf is not None:
            # Look specifically at where 3DEP is missing
            dep_missing = gdf["dep_elevation"].isna()
            if dep_missing.any():
                missing_points = gdf[dep_missing]
                logger.info(f"3DEP missing at {len(missing_points)} points")
                logger.info("Sample of missing points:")
                logger.info(missing_points[["elevation", "dep_elevation"]].head())


if __name__ == "__main__":
    asyncio.run(main())
