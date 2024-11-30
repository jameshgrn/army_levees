"""Functions for creating interactive maps of levee systems."""

import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Optional

import folium
import matplotlib.pyplot as plt

from .individual import get_system_statistics
from .utils import get_processed_systems, load_levee_data

logger = logging.getLogger(__name__)


def create_system_map(
    system_id: str, save_path: Optional[str] = None
) -> Optional[folium.Map]:
    """Create an interactive folium map for a single levee system."""
    try:
        # Load and filter data
        data = load_levee_data(system_id)
        if data is None:
            return None

        # Convert to EPSG:4326 for mapping
        data_4326 = data.to_crs(epsg=4326)

        # Calculate center point
        center_point = data_4326.unary_union.centroid
        center_lat, center_lon = center_point.y, center_point.x

        # Create base map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

        # Add satellite imagery
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="Satellite",
            overlay=False,
            control=True,
        ).add_to(m)

        # Add OpenStreetMap
        folium.TileLayer(
            tiles="OpenStreetMap", name="OpenStreetMap", overlay=False, control=True
        ).add_to(m)

        # Create feature groups
        fg_all = folium.FeatureGroup(name="All Points")
        fg_high = folium.FeatureGroup(name="High Difference (>0.5m)")
        fg_low = folium.FeatureGroup(name="Low Difference (<-0.5m)")

        # Add points to map
        for idx, row in data_4326.iterrows():
            # Create popup content
            popup_content = f"""
                <b>System ID:</b> {system_id}<br>
                <b>NLD Elevation:</b> {row['elevation']:.2f}m<br>
                <b>3DEP Elevation:</b> {row['dep_elevation']:.2f}m<br>
                <b>Difference:</b> {row['difference']:.2f}m<br>
                <b>Distance:</b> {row['distance_along_track']:.1f}m
            """

            # Create circle marker
            marker = folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5,
                popup=folium.Popup(popup_content, max_width=300),
                color="blue",
                fill=True,
                fillOpacity=0.7,
            )

            # Add to appropriate groups
            marker.add_to(fg_all)
            if row["difference"] > 0.5:
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=5,
                    popup=folium.Popup(popup_content, max_width=300),
                    color="red",
                    fill=True,
                    fillOpacity=0.7,
                ).add_to(fg_high)
            elif row["difference"] < -0.5:
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=5,
                    popup=folium.Popup(popup_content, max_width=300),
                    color="green",
                    fill=True,
                    fillOpacity=0.7,
                ).add_to(fg_low)

        # Add feature groups to map
        fg_all.add_to(m)
        fg_high.add_to(m)
        fg_low.add_to(m)

        # Add layer control
        folium.LayerControl().add_to(m)

        # Save map if path provided
        if save_path:
            m.save(save_path)
            logger.info(f"Saved interactive map to {save_path}")

        return m

    except Exception as e:
        logger.error(f"Error creating folium map: {str(e)}")
        return None


def create_summary_map(save_path: Optional[str] = None) -> Optional[folium.Map]:
    """Create an interactive folium map showing all levee systems."""
    try:
        systems = get_processed_systems()
        if not systems:
            logger.error("No processed levee files found")
            return None

        # Load first system to center map
        first_data = load_levee_data(systems[0])
        if first_data is None:
            return None

        first_data_4326 = first_data.to_crs(epsg=4326)
        center_point = first_data_4326.unary_union.centroid
        center_lat, center_lon = center_point.y, center_point.x

        # Create base map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

        # Add satellite imagery
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="Satellite",
            overlay=False,
            control=True,
        ).add_to(m)

        # Add OpenStreetMap
        folium.TileLayer(
            tiles="OpenStreetMap", name="OpenStreetMap", overlay=False, control=True
        ).add_to(m)

        # Create feature groups
        fg_all = folium.FeatureGroup(name="All Systems")
        fg_high = folium.FeatureGroup(name="High Difference (>0.5m)")
        fg_low = folium.FeatureGroup(name="Low Difference (<-0.5m)")

        # Process each system
        for system_id in systems:
            data = load_levee_data(system_id)
            if data is None:
                continue

            data_4326 = data.to_crs(epsg=4326)
            stats = get_system_statistics(system_id)
            if stats is None:
                continue

            # Create profile plot
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(
                data_4326["distance_along_track"],
                data_4326["elevation"],
                "b-",
                label="NLD",
            )
            ax.plot(
                data_4326["distance_along_track"],
                data_4326["dep_elevation"],
                "r--",
                label="3DEP",
            )
            ax.set_title(f"Elevation Profile - System {system_id}")
            ax.set_xlabel("Distance Along Levee (m)")
            ax.set_ylabel("Elevation (m)")
            ax.legend()
            ax.grid(True)
            plt.tight_layout()

            # Convert plot to base64 string
            buf = BytesIO()
            plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close()
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode()

            # Create popup content
            popup_content = f"""
                <h3>System {system_id}</h3>
                <b>Mean Difference:</b> {stats['mean_diff']:.2f}m<br>
                <b>Total Length:</b> {stats['total_length']:.0f}m<br>
                <b>Valid Points:</b> {stats['n_points']}<br>
                <img src="data:image/png;base64,{img_str}" alt="Elevation Profile" width="400">
            """

            # Create line feature
            line_coords = [[p.y, p.x] for p in data_4326.geometry]
            line = folium.PolyLine(
                locations=line_coords,
                popup=folium.Popup(popup_content, max_width=500),
                color="blue",
                weight=2,
                opacity=0.8,
            )

            # Add to appropriate groups
            line.add_to(fg_all)
            if stats["mean_diff"] > 0.5:
                folium.PolyLine(
                    locations=line_coords,
                    popup=folium.Popup(popup_content, max_width=500),
                    color="red",
                    weight=2,
                    opacity=0.8,
                ).add_to(fg_high)
            elif stats["mean_diff"] < -0.5:
                folium.PolyLine(
                    locations=line_coords,
                    popup=folium.Popup(popup_content, max_width=500),
                    color="green",
                    weight=2,
                    opacity=0.8,
                ).add_to(fg_low)

        # Add feature groups to map
        fg_all.add_to(m)
        fg_high.add_to(m)
        fg_low.add_to(m)

        # Add layer control
        folium.LayerControl().add_to(m)

        # Save map if path provided
        if save_path:
            m.save(save_path)
            logger.info(f"Saved interactive summary map to {save_path}")

        return m

    except Exception as e:
        logger.error(f"Error creating summary map: {str(e)}")
        return None
