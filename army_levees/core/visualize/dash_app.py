"""Dash app for interactive levee visualization."""

from pathlib import Path
from typing import Dict, List, Tuple

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import geopandas as gpd
import numpy as np
from matplotlib.colors import Normalize, to_hex, SymLogNorm
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

from .interactive import get_segment_files

def get_color_scale(values: np.ndarray) -> Tuple[List[str], List[float]]:
    """Create a diverging color scale for elevation differences using symmetric log normalization.

    Args:
        values: Array of values to create color scale for

    Returns:
        Tuple of (colors, values) for plotly color scale
    """
    # Find max absolute value for scaling
    vmax = max(abs(values.min()), abs(values.max()))

    # Use symmetric log normalization
    # Linear scaling near zero (within ±1m), log scaling beyond that
    norm = SymLogNorm(linthresh=1.0, linscale=1.0, vmin=-vmax, vmax=vmax)

    # Get RdYlBu colormap
    cmap = plt.get_cmap('RdYlBu')

    # Create color scale with more points near zero for better resolution
    # Use a combination of linear and log spacing
    linear_values = np.linspace(-1, 1, 5)  # More points in linear region
    pos_log_values = np.logspace(0, np.log10(vmax), 4)[1:]  # Log spacing for positives
    neg_log_values = -np.logspace(0, np.log10(vmax), 4)[1:][::-1]  # Log spacing for negatives
    scale_values = np.concatenate([neg_log_values, linear_values, pos_log_values])

    # Get colors
    colors = [to_hex(tuple(cmap(norm(v)))) for v in scale_values]

    # Convert to plotly format
    return colors, scale_values.tolist()

class LeveeDashboard:
    """Dash app for interactive levee visualization."""

    def __init__(
        self,
        data_dir: str | Path = "data/segments",
        raw_data: bool = False,
    ):
        """Initialize the dashboard.

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

        # Initialize data cache
        self.current_system = self.system_ids[0]
        self.data_cache: Dict[str, List[gpd.GeoDataFrame]] = {}
        self._update_cache()

        # Create Dash app
        self.app = dash.Dash(__name__)
        self.app.layout = self._create_layout()
        self._setup_callbacks()

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

    def _create_layout(self):
        """Create the Dash app layout."""
        return html.Div([
            html.H1("Levee System Dashboard", style={'textAlign': 'center'}),

            # Top row - Elevation Profile
            html.Div([
                dcc.Graph(
                    id='elevation-profile',
                    style={'height': '400px'},
                    config={'scrollZoom': True},
                ),
            ], style={'marginBottom': '20px'}),

            # Bottom row - Maps
            html.Div([
                # Overview Map
                html.Div([
                    html.Div([
                        html.Label("Overview Map Layer:"),
                        dcc.RadioItems(
                            id='overview-map-layer',
                            options=[
                                {'label': 'Street Map', 'value': 'street'},
                                {'label': 'USGS Satellite', 'value': 'satellite'},
                                {'label': 'Google Hybrid', 'value': 'google'}
                            ],
                            value='google',
                            inline=True,
                            style={'marginLeft': '10px'}
                        ),
                    ], style={'marginBottom': '5px'}),
                    dcc.Graph(
                        id='overview-map',
                        style={'height': '400px'},
                        config={'scrollZoom': True},
                    ),
                ], style={'width': '48%', 'display': 'inline-block'}),

                # System Detail Map
                html.Div([
                    html.Div([
                        html.Label("Detail Map Layer:"),
                        dcc.RadioItems(
                            id='detail-map-layer',
                            options=[
                                {'label': 'Street Map', 'value': 'street'},
                                {'label': 'USGS Satellite', 'value': 'satellite'},
                                {'label': 'Google Hybrid', 'value': 'google'}
                            ],
                            value='google',
                            inline=True,
                            style={'marginLeft': '10px'}
                        ),
                    ], style={'marginBottom': '5px'}),
                    dcc.Graph(
                        id='system-detail',
                        style={'height': '400px'},
                        config={'scrollZoom': True},
                    ),
                ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
            ]),

            # Store for current system ID
            dcc.Store(id='current-system-id'),
        ])

    def _get_base_map_layout(self, style: str) -> dict:
        """Get the base map layout configuration.

        Args:
            style: One of 'street', 'satellite', or 'google'
        """
        if style == 'satellite':
            return dict(
                style="white-bg",
                layers=[{
                    "below": 'traces',
                    "sourcetype": "raster",
                    "sourceattribution": "United States Geological Survey",
                    "source": [
                        "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
                    ]
                }]
            )
        elif style == 'google':
            return dict(
                style="white-bg",
                layers=[{
                    "below": 'traces',
                    "sourcetype": "raster",
                    "sourceattribution": "Google",
                    "source": [
                        "http://mt0.google.com/vt/lyrs=y&hl=en&x={x}&y={y}&z={z}&s=Ga"  # y for hybrid mode
                    ]
                }]
            )
        else:
            return dict(style="open-street-map")

    def _create_elevation_profile(self, system_id: str) -> go.Figure:
        """Create elevation profile plot."""
        segments = self.data_cache[system_id]

        fig = go.Figure()

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
                )
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
                )
            )

        fig.update_layout(
            title=f"Elevation Profile - System {system_id}",
            xaxis_title="Distance Along Track (m)",
            yaxis_title="Elevation (m)",
            showlegend=True,
            plot_bgcolor="white",
            paper_bgcolor="white",
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
            ),
        )

        return fig

    def _create_overview_map(self, base_map_style: str = 'street') -> go.Figure:
        """Create overview map of all systems."""
        all_lats = []
        all_lons = []
        mean_diffs = []
        hover_texts = []
        system_ids = []
        current_idx = None  # Track the current system index

        for i, system_id in enumerate(self.system_ids):
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

            # Track current system index
            if system_id == self.current_system:
                current_idx = len(all_lats)  # Store index before adding new points

            for j, segment in enumerate(segments):
                segment_4326 = segment.to_crs(4326)

                # Calculate statistics
                diff = segment["elevation"] - segment["dep_elevation"]
                mean_diff = diff.mean()
                std_diff = diff.std()
                max_diff = diff.abs().max()

                # Get center point
                center_lat = segment_4326.geometry.y.mean()
                center_lon = segment_4326.geometry.x.mean()
                all_lats.append(center_lat)
                all_lons.append(center_lon)
                system_ids.append(system_id)
                mean_diffs.append(mean_diff)

                hover_text = (
                    f"System: {system_id}<br>"
                    + f"Segment: {j+1}/{len(segments)}<br>"
                    + f"Mean Diff: {mean_diff:.1f}m<br>"
                    + f"Max Diff: {max_diff:.1f}m<br>"
                    + f"Std Diff: {std_diff:.1f}m"
                )
                hover_texts.append(hover_text)

        # Create color scale
        colors, values = get_color_scale(np.array(mean_diffs))
        colorscale = [[i/(len(colors)-1), color] for i, color in enumerate(colors)]

        fig = go.Figure(
            go.Scattermapbox(
                lat=all_lats,
                lon=all_lons,
                mode="markers",
                marker=dict(
                    color=mean_diffs,
                    colorscale=colorscale,
                    size=8,
                    colorbar=dict(
                        title="Mean Elevation<br>Difference (m)",
                        titleside="right",
                        tickmode="array",
                        ticktext=[f"{v:.1f}" for v in values],
                        tickvals=values,
                    ),
                ),
                text=hover_texts,
                hoverinfo="text",
                customdata=[[sid] for sid in system_ids],
                name="All Systems",
            )
        )

        # Add highlighted marker if we found the current system
        if current_idx is not None:
            fig.add_trace(
                go.Scattermapbox(
                    lat=[all_lats[current_idx]],
                    lon=[all_lons[current_idx]],
                    mode="markers",
                    marker=dict(
                        size=15,
                        color='yellow',
                        opacity=0.7,
                    ),
                    hoverinfo="skip",
                    name="Selected System",
                )
            )

        fig.update_layout(
            title="All Systems Overview",
            mapbox=dict(
                **self._get_base_map_layout(base_map_style),
                bounds=dict(
                    west=-125,
                    east=-65,
                    south=25,
                    north=49,
                ) if not hasattr(self, '_map_initialized') else None,
                center=dict(lat=39.8283, lon=-98.5795),
                zoom=3,
            ),
            showlegend=False,
            margin=dict(l=0, r=0, t=30, b=0),
            uirevision=True,
        )
        self._map_initialized = True
        return fig

    def _create_system_detail(self, system_id: str, base_map_style: str = 'street') -> go.Figure:
        """Create detailed map of current system."""
        segments = self.data_cache[system_id]

        fig = go.Figure()

        # Collect all differences to create consistent color scale
        all_diffs = []
        all_lats = []
        all_lons = []

        # First pass to collect all coordinates and differences
        for segment in segments:
            segment_4326 = segment.to_crs(4326)
            diff = segment["elevation"] - segment["dep_elevation"]
            all_diffs.extend(diff.values)

            coords = [[p.y, p.x] for p in segment_4326.geometry]
            all_lats.extend([c[0] for c in coords])
            all_lons.extend([c[1] for c in coords])

        # Calculate bounds with 2km buffer (roughly 0.018 degrees)
        buffer_size = 0.018
        lat_min, lat_max = min(all_lats), max(all_lats)
        lon_min, lon_max = min(all_lons), max(all_lons)

        # Calculate appropriate zoom level based on extent
        lat_range = lat_max - lat_min
        lon_range = lon_max - lon_min
        max_range = max(lat_range, lon_range)
        # Increased base zoom level for much closer default view
        zoom = max(14, min(20, int(-0.5 * np.log2(max_range))))  # Changed min to 14 and coefficient to -0.5

        # Create color scale and add traces
        colors, values = get_color_scale(np.array(all_diffs))
        colorscale = [[i/(len(colors)-1), color] for i, color in enumerate(colors)]

        # Add segment traces
        for i, segment in enumerate(segments):
            segment_4326 = segment.to_crs(4326)
            coords = [[p.y, p.x] for p in segment_4326.geometry]
            coords = np.array(coords)

            # Add white line for the segment
            fig.add_trace(
                go.Scattermapbox(
                    lat=coords[:, 0],
                    lon=coords[:, 1],
                    mode="lines",
                    line=dict(color="white", width=3),
                    name=f"Segment {i+1} Path",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

            # Calculate elevation difference for each point
            diffs = segment["elevation"] - segment["dep_elevation"]

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
                    diffs,
                )
            ]

            # Add colored points
            fig.add_trace(
                go.Scattermapbox(
                    lat=coords[:, 0],
                    lon=coords[:, 1],
                    mode="markers",
                    marker=dict(
                        color=diffs,
                        colorscale=colorscale,
                        size=8,
                        colorbar=dict(
                            title="Elevation<br>Difference (m)",
                            titleside="right",
                            tickmode="array",
                            ticktext=[f"{v:.1f}" for v in values],
                            tickvals=values,
                        ),
                        showscale=i==0,  # Only show colorbar for first segment
                    ),
                    name=f"Segment {i+1}",
                    text=hover_text,
                    hoverinfo="text",
                )
            )

        # Update layout with modified mapbox configuration
        fig.update_layout(
            title=f"System Detail - {system_id}",
            mapbox=dict(
                **self._get_base_map_layout(base_map_style),
                center=dict(
                    lat=(lat_min + lat_max) / 2,
                    lon=(lon_min + lon_max) / 2
                ),
                zoom=zoom,
            ),
            showlegend=True,
            margin=dict(l=0, r=0, t=30, b=0),
            uirevision=True,
        )

        return fig

    def _setup_callbacks(self):
        """Set up Dash callbacks."""

        @self.app.callback(
            [
                Output('current-system-id', 'data'),
                Output('elevation-profile', 'figure'),
                Output('system-detail', 'figure'),
                Output('overview-map', 'figure')
            ],
            [
                Input('overview-map', 'clickData'),
                Input('overview-map-layer', 'value'),
                Input('detail-map-layer', 'value')
            ]
        )
        def update_all(click_data, overview_style, detail_style):
            # Get system ID from click or use current
            if click_data is not None:
                system_id = click_data['points'][0]['customdata'][0]
                print(f"Updating to system: {system_id}")  # Debug print
                self.current_system = system_id
            else:
                system_id = self.current_system

            # Update cache if needed
            if system_id not in self.data_cache:
                self._update_cache()

            # Create all plots with explicit updates
            elevation_profile = self._create_elevation_profile(system_id)
            system_detail = self._create_system_detail(system_id, detail_style)
            system_detail['layout']['mapbox']['uirevision'] = None
            overview_map = self._create_overview_map(overview_style)

            # Return all updates
            return (
                system_id,
                elevation_profile,
                system_detail,
                overview_map
            )

    def run_server(self, debug: bool = True, port: int = 8040):
        """Run the Dash server."""
        self.app.run_server(debug=debug, port=port)


def create_dash_app(
    data_dir: str | Path = "data/segments",
    raw_data: bool = False,
    debug: bool = True,
    port: int = 8040,
) -> LeveeDashboard:
    """Create and run the Dash app.

    Args:
        data_dir: Directory containing levee data
        raw_data: Whether to use raw data
        debug: Whether to run in debug mode
        port: Port to run the server on

    Returns:
        The LeveeDashboard instance
    """
    dashboard = LeveeDashboard(data_dir=data_dir, raw_data=raw_data)
    dashboard.run_server(debug=debug, port=port)
    return dashboard
