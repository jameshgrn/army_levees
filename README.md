# Army Levees Analysis

A Python package for analyzing elevation differences between the National Levee Database (NLD) and USGS 3DEP data.

## Overview

This package provides an interactive dashboard and utilities for analyzing USACE levee systems by:
1. Visualizing elevation profiles from NLD and 3DEP data
2. Comparing and analyzing elevation differences
3. Exploring geographic patterns and trends

## Installation

```bash
# Clone the repository
git clone <repo>
cd army_levees

# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies using Poetry
poetry install

# Activate the Poetry shell
poetry shell
```

## Usage

### Interactive Dashboard
```bash
# Run the dashboard with default settings
poetry run python -m army_levees.core.visualize

# Specify custom data directory and port
poetry run python -m army_levees.core.visualize --data-dir custom/data/path --port 8040
```

The dashboard provides:
- Interactive map of all levee systems
- Detailed elevation profiles
- Real-time elevation difference analysis
- Satellite/street map overlays

### Individual Profile Analysis
```python
from army_levees.core.visualize.individual import plot_elevation_profile
from army_levees.core.visualize.utils import load_system_data

# Load and visualize a specific system
data = load_system_data("5205000591")
if data is not None:
    plot_elevation_profile(data)
```

### Multi-Profile Analysis
```bash
# Plot all profile types
python -m army_levees.core.visualize.multi_profile_plot --type all

# Plot specific profile types
python -m army_levees.core.visualize.multi_profile_plot --type degradation
python -m army_levees.core.visualize.multi_profile_plot --type stable

# Specify custom directories
python -m army_levees.core.visualize.multi_profile_plot --type all --data_dir custom/data --output_dir custom/output
```

## CLI Arguments

### Dashboard (visualize module)
- `--data-dir`: Directory containing levee segment data (default: data/segments)
- `--port`: Port to run the dashboard on (default: 8050)
- `--debug`: Run in debug mode
- `--raw`: Use raw data instead of processed data

### Multi-Profile Plot
- `--type`: Profile type to plot (all, degradation, stable)
- `--data_dir`: Input data directory
- `--output_dir`: Output directory for plots

## Project Structure

```
army_levees/
├── army_levees/          # Main package
│   └── core/            # Core functionality
│       └── visualize/   # Visualization modules
│           ├── __init__.py
│           ├── dash_app.py    # Interactive dashboard
│           ├── individual.py  # Individual system plots
│           ├── multi_profile_plot.py  # Multi-system analysis
│           └── utils.py       # Shared utilities
├── data/
│   └── segments/        # Processed segment files
├── plots/              # Generated plots
└── pyproject.toml      # Poetry configuration
```

## Data Format

Each parquet file in the segments directory contains:
- `system_id`: USACE system ID
- `elevation`: NLD elevation (meters)
- `dep_elevation`: 3DEP elevation (meters)
- `difference`: NLD - 3DEP (meters)
- `distance_along_track`: Distance along levee (meters)
- `geometry`: Point geometry (EPSG:4326)
