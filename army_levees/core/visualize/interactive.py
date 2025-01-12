"""Functions for creating interactive visualizations."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .utils import get_processed_systems, load_system_data


def get_segment_files(data_dir: Path) -> Dict[str, List[Path]]:
    """Get all segment files grouped by system ID."""
    segments = {}
    # First check if directory exists
    if not data_dir.exists():
        raise ValueError(f"Data directory {data_dir} does not exist")

    # Look for segment files
    for file in sorted(data_dir.glob("*.parquet")):
        if not file.stem.startswith("levee_"):
            continue

        # Extract system ID from filename
        # Format: levee_SYSTEMID_segment_XX.parquet
        parts = file.stem.split("_segment_")
        if len(parts) != 2:
            continue

        system_id = parts[0].replace("levee_", "")
        if system_id not in segments:
            segments[system_id] = []
        segments[system_id].append(file)

    if not segments:
        raise ValueError(f"No segment files found in {data_dir}")

    return segments


class InteractiveDashboard:
    """Interactive dashboard for visualizing levee elevation data.

    This class creates a Plotly-based dashboard with three main components:
    1. Elevation profile comparing NLD and 3DEP data
    2. Overview map showing all levee systems
    3. Detailed view of the current system

    The dashboard supports interactive features like zooming, panning,
    and hovering for detailed information.
    """

    def __init__(
        self,
        data_dir: str | Path = "data/segments",
        raw_data: bool = False,
    ):
        """Initialize the interactive dashboard.

        Args:
            data_dir: Directory containing levee segment data
            raw_data: Whether to use raw data
        """
        self.data_dir = Path(data_dir)
        self.raw_data = raw_data

        # Get all system IDs and their segments
        self.segments = get_segment_files(self.data_dir)
        self.system_ids = sorted(self.segments.keys())

        if not self.system_ids:
            raise ValueError("No levee systems found")

        # Load initial data
        self.current_system = self.system_ids[0]
        self.data_cache: Dict[str, List[gpd.GeoDataFrame]] = {}
        self._update_cache()

        # Create figure
        self.fig = self._create_dashboard()

    def _update_cache(self):
        """Update the data cache with current system."""
        if self.current_system not in self.data_cache:
            segments = []
            for file in sorted(self.segments[self.current_system]):
                try:
                    segment = gpd.read_parquet(file)
                    segments.append(segment)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
            self.data_cache[self.current_system] = segments

    def _create_dashboard(self) -> go.Figure:
        """Create the interactive dashboard."""
        # Create subplot layout
        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[[{"colspan": 2}, None], [{"type": "mapbox"}, {"type": "mapbox"}]],
            subplot_titles=(
                f"Elevation Profile - System {self.current_system}",
                "All Systems Overview",
                "System Detail",
            ),
            vertical_spacing=0.1,
            horizontal_spacing=0.05,
        )

        # Add elevation profile
        self._add_elevation_profile(fig)

        # Add overview map
        self._add_overview_map(fig)

        # Add system detail map
        self._add_system_detail(fig)

        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            plot_bgcolor="white",
            paper_bgcolor="white",
            title=dict(
                text=f"Levee System Dashboard - System {self.current_system}",
                x=0.5,
                xanchor="center",
            ),
            mapbox=dict(style="open-street-map", zoom=3),
            mapbox2=dict(style="open-street-map", zoom=10),
            clickmode='event+select',
        )

        return fig

    def _add_elevation_profile(self, fig: go.Figure):
        """Add elevation profile plot."""
        segments = self.data_cache[self.current_system]

        for i, segment in enumerate(segments):
            # Add NLD elevation
            fig.add_trace(
                go.Scatter(
                    x=segment["distance_along_track"],
                    y=segment["elevation"],
                    mode="lines+markers",
                    name=f"NLD Elevation (Segment {i+1})",
                    line=dict(color="blue"),
                    showlegend=True,
                    hovertemplate=(
                        "Segment %d<br>" % (i + 1)
                        + "Distance: %{x:.1f}m<br>"
                        + "NLD Elevation: %{y:.1f}m<br>"
                        + "<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )

            # Add 3DEP elevation
            fig.add_trace(
                go.Scatter(
                    x=segment["distance_along_track"],
                    y=segment["dep_elevation"],
                    mode="lines+markers",
                    name=f"3DEP Elevation (Segment {i+1})",
                    line=dict(color="red"),
                    showlegend=True,
                    hovertemplate=(
                        "Segment %d<br>" % (i + 1)
                        + "Distance: %{x:.1f}m<br>"
                        + "3DEP Elevation: %{y:.1f}m<br>"
                        + "<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )

        # Update axes
        fig.update_xaxes(
            title_text="Distance Along Track (m)",
            row=1,
            col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray",
        )
        fig.update_yaxes(
            title_text="Elevation (m)",
            row=1,
            col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray",
        )

    def _add_overview_map(self, fig: go.Figure):
        """Add overview map of all systems."""
        all_lats = []
        all_lons = []
        colors = []
        hover_texts = []

        for system_id in self.system_ids:
            if system_id not in self.data_cache:
                segments = []
                for file in sorted(self.segments[system_id]):
                    try:
                        segment = gpd.read_parquet(file)
                        segments.append(segment)
                    except Exception as e:
                        print(f"Error loading {file}: {e}")
                self.data_cache[system_id] = segments

            segments = self.data_cache[system_id]
            if not segments:
                continue

            # Process each segment
            for i, segment in enumerate(segments):
                # Convert to WGS84
                segment_4326 = segment.to_crs(4326)

                # Calculate statistics
                diff = segment["elevation"] - segment["dep_elevation"]
                mean_diff = diff.mean()
                std_diff = diff.std()
                max_diff = diff.abs().max()

                # Get center point of segment
                center_lat = segment_4326.geometry.y.mean()
                center_lon = segment_4326.geometry.x.mean()
                all_lats.append(center_lat)
                all_lons.append(center_lon)

                # Add color based on difference
                colors.append("red" if abs(mean_diff) > 5 else "blue")

                # Add hover text
                hover_text = (
                    f"System: {system_id}<br>"
                    + f"Segment: {i+1}/{len(segments)}<br>"
                    + f"Mean Diff: {mean_diff:.1f}m<br>"
                    + f"Max Diff: {max_diff:.1f}m<br>"
                    + f"Std Diff: {std_diff:.1f}m"
                )
                hover_texts.append(hover_text)

        if all_lats:
            fig.add_trace(
                go.Scattermapbox(
                    lat=all_lats,
                    lon=all_lons,
                    mode="markers",
                    marker=dict(color=colors, size=8),
                    name="All Systems",
                    text=hover_texts,
                    hoverinfo="text",
                    customdata=[[system_id] for system_id in self.system_ids],
                    showlegend=True,
                ),
                row=2,
                col=1,
            )

            # Center map
            center_lat = np.mean(all_lats)
            center_lon = np.mean(all_lons)
            fig.update_layout(mapbox=dict(center=dict(lat=center_lat, lon=center_lon)))

    def _add_system_detail(self, fig: go.Figure):
        """Add detailed map of current system."""
        segments = self.data_cache[self.current_system]

        for i, segment in enumerate(segments):
            # Convert to WGS84
            segment_4326 = segment.to_crs(4326)

            # Calculate elevation difference
            diff = segment["elevation"] - segment["dep_elevation"]
            mean_diff = diff.mean()
            std_diff = diff.std()
            max_diff = diff.abs().max()
            color = "red" if abs(mean_diff) > 5 else "blue"

            # Add system trace
            coords = [[p.y, p.x] for p in segment_4326.geometry]
            coords = np.array(coords)

            hover_text = [
                (
                    f"Segment {i+1}<br>"
                    + f"Distance: {d:.1f}m<br>"
                    + f"NLD: {e:.1f}m<br>"
                    + f"3DEP: {d3:.1f}m<br>"
                    + f"Diff: {diff:.1f}m"
                )
                for d, e, d3, diff in zip(
                    segment["distance_along_track"],
                    segment["elevation"],
                    segment["dep_elevation"],
                    segment["elevation"] - segment["dep_elevation"],
                )
            ]

            fig.add_trace(
                go.Scattermapbox(
                    lat=coords[:, 0],
                    lon=coords[:, 1],
                    mode="lines+markers",
                    marker=dict(color=color, size=8),
                    name=f"Segment {i+1}",
                    text=hover_text,
                    hoverinfo="text",
                    showlegend=True,
                ),
                row=2,
                col=2,
            )

        # Center map on first segment
        if segments:
            segment_4326 = segments[0].to_crs(4326)
            center_lat = segment_4326.geometry.y.mean()
            center_lon = segment_4326.geometry.x.mean()
            fig.update_layout(mapbox2=dict(center=dict(lat=center_lat, lon=center_lon)))

    def update_system(self, system_id: str):
        """Update the dashboard to show a different system."""
        if system_id not in self.system_ids:
            raise ValueError(f"System {system_id} not found")

        self.current_system = system_id
        self._update_cache()

        # Create new figure
        self.fig = self._create_dashboard()

    def show(self):
        """Display the dashboard."""
        self.fig.show()

    def save(self, save_path: str | Path):
        """Save the dashboard to HTML file with click functionality."""
        # Add JavaScript callback for click events
        self.fig.write_html(
            str(save_path),
            config={'responsive': True},
            full_html=True,
            include_plotlyjs=True,
            post_script="""
            <script>
                var plot = document.getElementsByClassName('plotly-graph-div')[0];
                plot.on('plotly_click', function(data) {
                    if (data.points.length > 0 && data.points[0].customdata) {
                        var system_id = data.points[0].customdata[0];
                        // Update all plots with new system
                        Plotly.react(plot, updateSystem(system_id));
                    }
                });

                function updateSystem(system_id) {
                    // Make AJAX request to get new system data
                    fetch(`/update_system/${system_id}`)
                        .then(response => response.json())
                        .then(newFigure => {
                            Plotly.react(plot, newFigure.data, newFigure.layout);
                        });
                }
            </script>
            """
        )


def create_interactive_dashboard(
    save_path: str | Path = "plots/levee_dashboard.html",
    data_dir: str | Path = "data/segments",
    raw_data: bool = False,
    show: bool = False,
) -> Optional[InteractiveDashboard]:
    """Create an interactive dashboard visualization.

    The dashboard consists of:
    - Top: Interactive elevation profile (NLD vs 3DEP) with segments
    - Bottom left: Overview map of all levee systems
    - Bottom right: Detailed satellite view of current system

    Args:
        save_path: Path to save the dashboard HTML
        data_dir: Directory containing levee data
        raw_data: Whether to use raw data
        show: Whether to display the dashboard
    """
    try:
        dashboard = InteractiveDashboard(data_dir=data_dir, raw_data=raw_data)

        # Save dashboard
        dashboard.save(save_path)

        if show:
            dashboard.show()

        return dashboard

    except Exception as e:
        print(f"Error creating dashboard: {e}")
        return None


from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/update_system/<system_id>')
def update_system(system_id):
    dashboard = InteractiveDashboard()  # You'll need to handle state management
    dashboard.update_system(system_id)
    return jsonify({
        'data': dashboard.fig.data,
        'layout': dashboard.fig.layout
    })
