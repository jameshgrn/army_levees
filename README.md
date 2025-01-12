# Army Levees Analysis

A Python package for analyzing elevation differences between the National Levee Database (NLD) and USGS 3DEP data.

## Overview

This package provides tools for analyzing USACE levee systems by:
1. Visualizing elevation profiles from NLD and 3DEP data
2. Comparing and analyzing elevation differences
3. Exploring geographic patterns and trends
4. Classifying levees based on elevation changes

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
poetry run python -m army_levees.core.visualize dashboard

# Specify custom data directory and port
poetry run python -m army_levees.core.visualize dashboard --data-dir custom/data/path --port 8040
```

The dashboard provides:
- Interactive map of all levee systems
- Detailed elevation profiles with filled difference areas
- Real-time elevation difference analysis
- Satellite/street map overlays
- Color-coded elevation differences

### Individual Profile Analysis
```bash
# Plot a single system
poetry run python -m army_levees.core.visualize plot 5205000591

# Show plot instead of saving
poetry run python -m army_levees.core.visualize plot 5205000591 --show

# Print diagnostic information
poetry run python -m army_levees.core.visualize diagnose 5205000591
```

### Multi-Profile Analysis
```bash
# Plot all profiles
poetry run python -m army_levees.core.visualize multi

# Plot specific profile types
poetry run python -m army_levees.core.visualize multi --type significant
poetry run python -m army_levees.core.visualize multi --type non_significant

# Show plots instead of saving
poetry run python -m army_levees.core.visualize multi --type significant --show

# Use raw data and specify directories
poetry run python -m army_levees.core.visualize multi --raw-data \
    --data-dir custom/data \
    --output-dir custom/output \
    --summary-dir custom/summary
```

## CLI Arguments

### Dashboard Command
- `--data-dir`: Directory containing levee segment data (default: data/segments)
- `--port`: Port to run the dashboard on (default: 8050)
- `--debug`: Run in debug mode
- `--raw`: Use raw data instead of processed data

### Plot Command
- `system_id`: System ID to plot
- `--data-dir`: Directory containing levee segment data
- `--save-dir`: Directory to save plots (default: plots)
- `--raw`: Use raw data instead of processed data
- `--show`: Show plot instead of saving

### Diagnose Command
- `system_id`: System ID to diagnose
- `--data-dir`: Directory containing levee segment data
- `--raw`: Use raw data instead of processed data

### Multi-Profile Command
- `--type`: Profile type to plot (all, significant, non_significant)
- `--data-dir`: Directory containing filtered segments
- `--raw-data`: Use raw data from data/processed
- `--output-dir`: Directory to save plots (default: plots/profiles)
- `--summary-dir`: Directory containing classification CSV files
- `--show`: Show plots instead of saving them

## Project Structure

```
army_levees/
├── army_levees/          # Main package
│   └── core/            # Core functionality
│       └── visualize/   # Visualization modules
│           ├── __init__.py
│           ├── __main__.py     # CLI entry point
│           ├── dash_app.py     # Interactive dashboard
│           ├── multi_profile_plot.py  # Multi-system analysis
│           └── utils.py        # Core utilities
├── data/
│   ├── processed/       # Raw data
│   └── segments/        # Filtered data
└── plots/              # Generated plots
```

## Data Format

Each parquet file in the segments directory contains:
- `system_id`: USACE system ID
- `elevation`: NLD elevation (meters)
- `dep_elevation`: 3DEP elevation (meters)
- `difference`: NLD - 3DEP (meters)
- `distance_along_track`: Distance along levee (meters)
- `geometry`: Point geometry (EPSG:4326)

## Classification Criteria

Levees are classified based on their mean elevation differences:
- **Significant**: Mean change > 0.1m or < -0.1m
- **Non-significant**: Mean change between -0.1m and 0.1m
