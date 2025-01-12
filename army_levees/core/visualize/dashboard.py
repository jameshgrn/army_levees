"""Functions for creating a dashboard visualization of levee data."""

from pathlib import Path
from typing import List, Optional

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
from matplotlib.widgets import Button
from shapely.geometry import LineString, Point

from .utils import get_processed_systems, load_system_data


class DashboardPlotter:
    """Dashboard plotter for visualizing levee data.

    This class creates a matplotlib-based dashboard with three main components:
    1. Elevation profile comparing NLD and 3DEP data
    2. Location map with satellite imagery
    3. State-level context map

    The dashboard includes navigation buttons to browse through different levee systems.
    """

    def __init__(
        self,
        data_dir: str | Path = "data/segments",
        raw_data: bool = False,
        cache_size: int = 5,
    ):
        """Initialize the dashboard plotter.

        Args:
            data_dir: Directory containing levee data
            raw_data: Whether to use raw data
            cache_size: Number of systems to cache in each direction (prev/next)
        """
        self.data_dir = Path(data_dir)
        self.raw_data = raw_data
        self.cache_size = cache_size
        self.system_ids = sorted(get_processed_systems(data_dir=data_dir))
        self.current_idx = 0
        self.data_cache = {}  # Cache for loaded data

        if not self.system_ids:
            raise ValueError("No levee systems found")

        # Load state boundaries from Natural Earth Data
        url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_1_states_provinces_lakes.geojson"
        self.states = gpd.read_file(url)
        # Filter for US states
        self.states = self.states[self.states["admin"] == "United States of America"]
        self.states = self.states.to_crs(epsg=3857)

        # Create figure
        self.fig = plt.figure(figsize=(15, 10))
        self.gs = GridSpec(2, 2, figure=self.fig)

        # Add navigation buttons
        prev_ax = plt.axes((0.2, 0.01, 0.1, 0.04))
        next_ax = plt.axes((0.7, 0.01, 0.1, 0.04))
        self.prev_button = Button(prev_ax, "Previous")
        self.next_button = Button(next_ax, "Next")
        self.prev_button.on_clicked(self.prev_system)
        self.next_button.on_clicked(self.next_system)

        # Initialize cache
        self._update_cache()

        # Plot initial system
        self.update_plot()

    def _update_cache(self):
        """Update the data cache with nearby systems."""
        n = len(self.system_ids)
        for i in range(-self.cache_size, self.cache_size + 1):
            idx = (self.current_idx + i) % n
            system_id = self.system_ids[idx]
            if system_id not in self.data_cache:
                data = load_system_data(
                    system_id, data_dir=self.data_dir, raw_data=self.raw_data
                )
                if data is not None:
                    self.data_cache[system_id] = data

        # Remove old cache entries
        current_system = self.system_ids[self.current_idx]
        cache_range = set(
            self.system_ids[(self.current_idx + i) % n]
            for i in range(-self.cache_size, self.cache_size + 1)
        )
        self.data_cache = {k: v for k, v in self.data_cache.items() if k in cache_range}

    def prev_system(self, event):
        """Show previous system."""
        self.current_idx = (self.current_idx - 1) % len(self.system_ids)
        self._update_cache()
        self.update_plot()

    def next_system(self, event):
        """Show next system."""
        self.current_idx = (self.current_idx + 1) % len(self.system_ids)
        self._update_cache()
        self.update_plot()

    def update_plot(self):
        """Update the plot with current system."""
        # Clear previous plots
        self.fig.clf()
        self.gs = GridSpec(2, 2, figure=self.fig)

        # Get current system data
        system_id = self.system_ids[self.current_idx]
        data = self.data_cache.get(system_id)
        if data is None:
            data = load_system_data(
                system_id, data_dir=self.data_dir, raw_data=self.raw_data
            )
            if data is None:
                print(f"No data found for system {system_id}")
                return
            self.data_cache[system_id] = data

        # Plot elevation profile
        self.plot_elevation_profile(data, system_id)

        # Plot location map
        self.plot_location_map(data)

        # Plot state map
        self.plot_state_map(data)

        # Add navigation buttons
        prev_ax = plt.axes((0.2, 0.01, 0.1, 0.04))
        next_ax = plt.axes((0.7, 0.01, 0.1, 0.04))
        self.prev_button = Button(prev_ax, "Previous")
        self.next_button = Button(next_ax, "Next")
        self.prev_button.on_clicked(self.prev_system)
        self.next_button.on_clicked(self.next_system)

        # Update layout
        plt.tight_layout()
        plt.draw()

    def plot_elevation_profile(self, profile: gpd.GeoDataFrame, system_id: str):
        """Plot elevation profile."""
        ax_profile = self.fig.add_subplot(self.gs[0, :])

        # Plot lines
        ax_profile.plot(
            profile["distance_along_track"],
            profile["elevation"],
            "b-",
            label="NLD",
            alpha=0.8,
        )
        ax_profile.plot(
            profile["distance_along_track"],
            profile["dep_elevation"],
            "r-",
            label="3DEP",
            alpha=0.8,
        )

        # Add points
        ax_profile.scatter(
            profile["distance_along_track"],
            profile["elevation"],
            c="blue",
            s=20,
            alpha=0.5,
        )
        ax_profile.scatter(
            profile["distance_along_track"],
            profile["dep_elevation"],
            c="red",
            s=20,
            alpha=0.5,
        )

        # Customize plot
        ax_profile.set_title(
            f"System ID: {system_id} ({self.current_idx + 1}/{len(self.system_ids)})"
        )
        ax_profile.set_xlabel("Distance Along Track (m)")
        ax_profile.set_ylabel("Elevation (m)")
        ax_profile.grid(True, alpha=0.3)
        ax_profile.legend()

    def plot_location_map(self, profile: gpd.GeoDataFrame):
        """Plot location map with satellite imagery."""
        ax_map = self.fig.add_subplot(self.gs[1, 0])

        # Convert to Web Mercator for contextily
        gdf_3857 = profile.to_crs(epsg=3857)

        # Plot the points
        gdf_3857.plot(ax=ax_map, color="blue", alpha=0.5, markersize=3)

        # Create and plot the line connecting points
        points = [(p.x, p.y) for p in gdf_3857.geometry]
        line = LineString(points)
        ax_map.plot(
            [p[0] for p in points], [p[1] for p in points], "r-", linewidth=2, alpha=0.8
        )

        # Add satellite imagery basemap
        ctx.add_basemap(
            ax_map,
            source="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            zoom="auto",
        )

        # Add OpenStreetMap overlay with partial transparency
        ctx.add_basemap(ax_map, alpha=0.5, zoom="auto")

        # Customize map
        ax_map.set_title("Location")
        ax_map.set_xlabel("Longitude")
        ax_map.set_ylabel("Latitude")

    def plot_state_map(self, profile: gpd.GeoDataFrame):
        """Plot state map with levee location."""
        ax_state = self.fig.add_subplot(self.gs[1, 1])

        # Convert profile to same CRS as states
        gdf_3857 = profile.to_crs(epsg=3857)

        # Get state containing the levee
        center_point = gdf_3857.geometry.unary_union.centroid

        # Find the state containing the levee
        state_mask = self.states.intersects(gdf_3857.unary_union)
        current_state = self.states[state_mask]

        if not current_state.empty:
            # Get bounds of current state
            state_bounds = current_state.total_bounds

            # Find neighboring states
            buffer_distance = 10000  # 10 km buffer (reduced from 50km)
            buffered_state = current_state.geometry.buffer(buffer_distance)
            neighbor_mask = self.states.intersects(buffered_state.iloc[0])
            nearby_states = self.states[neighbor_mask]

            # Plot nearby states
            nearby_states.plot(
                ax=ax_state,
                color="lightgray",
                edgecolor="black",
                linewidth=0.5,
                alpha=0.3,
            )

            # Highlight current state
            current_state.plot(
                ax=ax_state, color="white", edgecolor="black", linewidth=1, alpha=0.5
            )

            # Calculate bounds centered on the levee with a small buffer
            levee_bounds = gdf_3857.total_bounds
            levee_center_x = (levee_bounds[0] + levee_bounds[2]) / 2
            levee_center_y = (levee_bounds[1] + levee_bounds[3]) / 2

            # Use a fixed width and height for the view (about 20km)
            view_width = 20000  # meters
            view_height = 20000  # meters

            # Set view bounds centered on levee
            ax_state.set_xlim(
                levee_center_x - view_width / 2, levee_center_x + view_width / 2
            )
            ax_state.set_ylim(
                levee_center_y - view_height / 2, levee_center_y + view_height / 2
            )

            # Create a mask for the states
            from shapely.geometry import MultiPolygon

            geom = nearby_states.unary_union
            if isinstance(geom, MultiPolygon):
                # For MultiPolygon, combine all exterior coordinates
                coords = []
                for polygon in geom.geoms:
                    coords.extend(polygon.exterior.coords)
                coords = np.array(coords)
            else:
                # For single Polygon
                coords = np.array(geom.exterior.coords)

            mask_patch = Polygon(coords, facecolor="none", edgecolor="none")
            ax_state.add_patch(mask_patch)

            # Add satellite imagery basemap
            ctx.add_basemap(
                ax_state,
                source="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                alpha=0.8,
            )

            # Add OpenStreetMap overlay with higher transparency
            ctx.add_basemap(
                ax_state,
                source="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
                alpha=0.4,
            )

            # Set clip path for the basemap
            for img in ax_state.get_images():
                img.set_clip_path(mask_patch)

            # Plot levee location
            gdf_3857.plot(
                ax=ax_state, color="red", markersize=20, alpha=0.8  # Reduced from 50
            )
        else:
            # Fallback to showing continental US
            conus_mask = ~self.states["name"].isin(["Alaska", "Hawaii"])
            conus_states = self.states[conus_mask]
            conus_states.plot(
                ax=ax_state,
                color="lightgray",
                edgecolor="black",
                linewidth=0.5,
                alpha=0.3,
            )

            # Create a mask for CONUS
            from shapely.geometry import MultiPolygon

            geom = conus_states.unary_union
            if isinstance(geom, MultiPolygon):
                # For MultiPolygon, combine all exterior coordinates
                coords = []
                for polygon in geom.geoms:
                    coords.extend(polygon.exterior.coords)
                coords = np.array(coords)
            else:
                # For single Polygon
                coords = np.array(geom.exterior.coords)

            mask_patch = Polygon(coords, facecolor="none", edgecolor="none")
            ax_state.add_patch(mask_patch)

            # Add satellite imagery basemap
            ctx.add_basemap(
                ax_state,
                source="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                alpha=0.8,
            )

            # Add OpenStreetMap overlay with higher transparency
            ctx.add_basemap(
                ax_state,
                source="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
                alpha=0.4,
            )

            # Set clip path for the basemap
            for img in ax_state.get_images():
                img.set_clip_path(mask_patch)

            # Plot levee location
            gdf_3857.plot(
                ax=ax_state, color="red", markersize=20, alpha=0.8  # Reduced from 50
            )

            # Set bounds to continental US with small buffer
            bounds = conus_states.total_bounds
            bounds_width = bounds[2] - bounds[0]
            bounds_height = bounds[3] - bounds[1]
            center_x = (bounds[0] + bounds[2]) / 2
            center_y = (bounds[1] + bounds[3]) / 2
            buffer_ratio = 0.01  # Reduced from 0.02
            ax_state.set_xlim(
                center_x - bounds_width * (1 + buffer_ratio) / 2,
                center_x + bounds_width * (1 + buffer_ratio) / 2,
            )
            ax_state.set_ylim(
                center_y - bounds_height * (1 + buffer_ratio) / 2,
                center_y + bounds_height * (1 + buffer_ratio) / 2,
            )

        # Customize map
        ax_state.set_title("State Location")
        ax_state.set_xlabel("Longitude")
        ax_state.set_ylabel("Latitude")
        ax_state.axis("equal")


def create_dashboard(
    system_id: str,
    save_dir: str | Path = "plots",
    data_dir: str | Path = "data/segments",
    raw_data: bool = False,
    show: bool = False,
) -> None:
    """Create a dashboard visualization for a levee system.

    The dashboard consists of:
    - Top: Elevation profile (NLD vs 3DEP)
    - Bottom left: Contextily map with levee location
    - Bottom right: State map showing levee location
    """
    # Create plotter
    plotter = DashboardPlotter(data_dir=data_dir, raw_data=raw_data)

    # Find index of requested system
    try:
        idx = plotter.system_ids.index(system_id)
        plotter.current_idx = idx
        plotter.update_plot()
    except ValueError:
        print(f"System {system_id} not found")
        return

    # Save plot
    save_path = Path(save_dir) / f"dashboard_{system_id}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()
