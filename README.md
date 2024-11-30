# Army Levees Analysis

A Python package for analyzing elevation differences between the National Levee Database (NLD) and USGS 3DEP data.

## Overview

This package helps collect and analyze elevation data for USACE levee systems by:
1. Getting profile data from the NLD API
2. Getting matching elevations from USGS 3DEP
3. Comparing and visualizing the differences
4. Analyzing geomorphic changes and patterns

## System Architecture

```mermaid
graph TD
    subgraph Data_Collection
        A[USACE System IDs] -->|get_usace_system_ids| B[Random Sample Selection]
        B -->|get_random_levees_async| C[Async Processing]
        C -->|get_nld_profile_async| D[NLD Profile Data]
        C -->|get_3dep_elevations_async| E[3DEP Elevation Data]
    end

    subgraph Data_Processing
        D --> F[Create GeoDataFrame]
        E --> F
        F --> G[Valid Segments]
        G -->|calculate| H[Elevation Differences]
    end

    subgraph Filtering_Process
        F -->|Input Data| V0[Initial Data]

        subgraph Validation_Checks
            V1[NLD elevation > 0.01m] --> VM[Validation Mask]
            V2[3DEP elevation > 1.0m] --> VM
            V3[No missing data] --> VM
            V4[No extreme jumps] --> VM
            V5[No unit mismatches] --> VM
        end

        subgraph Segment_Processing
            V0 --> S1[Apply Validation Mask]
            VM -->|filter| S1
            S1 --> S2[Sort by Distance]
            S2 --> S3[Calculate Differences]
            S3 -->|Offset Check| S4[Check Mean Diff]
            S4 -->|Std Dev Check| S5[Check Consistency]
            S5 --> S6[Final Valid Segments]
        end

        subgraph Floodwall_Check
            V0 -->|check segments| FW[Has Floodwalls?]
            FW -->|yes| Skip[Skip System]
            FW -->|no| S1
        end

        S6 --> G
    end

    subgraph Analysis_Visualization
        H --> V[Visualization Module]
        V --> P1[Individual Plots]
        V --> P2[Summary Statistics]
        V --> P3[Interactive Maps]
        V --> P4[Geomorphic Analysis]
        
        subgraph Change_Analysis
            P4 --> CA1[Degradation Analysis]
            P4 --> CA2[Aggradation Patterns]
            P4 --> CA3[Change Consistency]
            P4 --> CA4[Geographic Patterns]
        end
    end

    subgraph Data_Storage
        F -->|save_parquet| P[(Processed Data)]
        P2 -->|save_summary| S[(Summary Stats)]
    end

    linkStyle default stroke-width:2px

    style Data_Collection fill:#f4f4f4,stroke:#333,stroke-width:2px
    style Data_Processing fill:#e8f4ea,stroke:#333,stroke-width:2px
    style Filtering_Process fill:#f4e8ea,stroke:#333,stroke-width:2px
    style Analysis_Visualization fill:#e8eaf4,stroke:#333,stroke-width:2px
    style Data_Storage fill:#f4f4e8,stroke:#333,stroke-width:2px

    style Validation_Checks fill:#fff,stroke:#333,stroke-width:2px
    style Segment_Processing fill:#fff,stroke:#333,stroke-width:2px
    style Floodwall_Check fill:#fff,stroke:#333,stroke-width:2px
    style Change_Analysis fill:#fff,stroke:#333,stroke-width:2px
```

## Quick Start

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

## Key Features

1. **Data Collection**
   - Fetches levee profiles from NLD API
   - Samples matching elevations from USGS 3DEP
   - Filters out floodwalls automatically
   - Handles coordinate system transformations
   - Supports async/await for efficient data collection

2. **Analysis**
   - Calculates elevation differences
   - Identifies problematic sections
   - Analyzes geomorphic changes
   - Detects degradation/aggradation patterns
   - Geographic pattern analysis
   - Change consistency metrics

3. **Visualization**
   - Individual levee plots showing:
     * Elevation profiles (NLD vs 3DEP)
     * Difference distributions
     * Coverage statistics
   - Summary plots showing:
     * Distribution of changes by category
     * Cumulative change distributions
     * Change consistency analysis
     * Geographic patterns
   - Interactive maps with:
     * System locations
     * Change magnitudes
     * Elevation profiles
     * Detailed statistics

## Usage Examples

### 1. Get Random Levee Samples
```python
from army_levees.core.sample_levees import get_random_levees

# Get 10 new random samples (skips already processed systems)
results = get_random_levees(n_samples=10, max_concurrent=4)
```

### 2. Plot Individual Levee System
```python
from army_levees.core.visualize.individual import plot_elevation_profile
from army_levees.core.visualize.utils import load_levee_data

# Load and visualize a specific system
data = load_levee_data("5205000591")
if data is not None:
    plot_elevation_profile(data)
```

### 3. Analyze Existing Dataset
```python
from army_levees.core.visualize.summary import plot_summary

# Generate summary plots and statistics
plot_summary(save_dir="plots")
```

### 4. Create Interactive Map
```python
from army_levees.core.visualize.interactive import create_summary_map

# Generate interactive map of all systems
create_summary_map(save_path="plots/levee_summary_map.html")
```

### 5. Run Diagnostics
```python
from army_levees.core.visualize.individual import diagnose_elevation_differences

# Print diagnostic information for a system
diagnose_elevation_differences("5205000591")
```

## CLI

The package provides command-line tools for sampling and analyzing levee systems:

### Sample Levees
```bash
# Get 10 new random levee samples
poetry run python -m army_levees.core.sample_levees -n 10

# Include already processed systems in sampling
poetry run python -m army_levees.core.sample_levees -n 10 --include_existing

# Control concurrent connections
poetry run python -m army_levees.core.sample_levees -n 10 --max_concurrent 8
```

### Visualize Levees
```bash
# Plot a specific levee system
poetry run python -m army_levees.core.visualize_levee --plot 5205000591

# Plot a random levee system
poetry run python -m army_levees.core.visualize_levee --plot --random

# Create summary plots for all processed levees
poetry run python -m army_levees.core.visualize_levee --summary

# Create interactive map of all levees
poetry run python -m army_levees.core.visualize_levee --map

# Run diagnostics on a specific system
poetry run python -m army_levees.core.visualize_levee --diagnose 5205000591

# Specify custom save directory
poetry run python -m army_levees.core.visualize_levee --plot --random --save_dir custom_plots
```

### Multi Profile Plot
```bash
# Plot all profile types
python -m army_levees.core.visualize.multi_profile_plot --type all

# Plot only degradation profiles
python -m army_levees.core.visualize.multi_profile_plot --type degradation

# Plot only stable profiles
python -m army_levees.core.visualize.multi_profile_plot --type stable

# Specify custom directories
python -m army_levees.core.visualize.multi_profile_plot --type all --data_dir custom/data/path --output_dir custom/output/path
```

### CLI Arguments

**sample_levees.py**:
- `-n, --n_samples`: Number of systems to sample (default: 10)
- `--include_existing`: Include already processed systems in sampling
- `--max_concurrent`: Maximum number of concurrent connections (default: 4)

**visualize_levee.py**:
- `system_id`: USACE system ID to plot (required with --plot or --diagnose)
- `-r, --random`: Use a random levee system
- `-p, --plot`: Create plots for the system
- `-s, --summary`: Create summary plots for all processed levees
- `-d, --diagnose`: Run diagnostics on the system
- `-m, --map`: Create interactive summary map
- `--save_dir`: Directory to save outputs (default: plots)

## Project Structure

```
army_levees/
├── army_levees/          # Main package
│   └── core/            # Core functionality
│       ├── nld_api.py   # NLD API interface
│       ├── sample_levees.py  # Sampling functions
│       └── visualize/   # Visualization modules
│           ├── __init__.py
│           ├── individual.py  # Individual system plots
│           ├── interactive.py # Interactive maps
│           ├── summary.py     # Summary statistics
│           └── utils.py       # Shared utilities
├── data/
│   └── processed/       # Processed parquet files
├── docs/               # Documentation
├── plots/             # Generated plots
├── tests/             # Test suite
├── pyproject.toml     # Poetry configuration
└── README.md
```

## Data Format

Each parquet file contains:
- `system_id`: USACE system ID
- `elevation`: NLD elevation (meters)
- `dep_elevation`: 3DEP elevation (meters)
- `difference`: NLD - 3DEP (meters)
- `distance_along_track`: Distance along levee (meters)
- `geometry`: Point geometry (EPSG:4326)
- `elevation_diff`: Change between consecutive points
- `elevation_diff_pct`: Percent change between points

## Contributing

To add more samples to the dataset:
1. Install the package as above
2. Run `poetry run python -m army_levees.core.sample_levees -n 10`
3. New samples will be added to `data/processed/`

The script will:
- Skip systems that are already processed
- Show dataset statistics
- Generate visualizations
- Save a summary CSV

## Dependencies

Key packages (see pyproject.toml for full list):
- geopandas (^0.14.1)
- shapely (^2.0.2)
- pandas (^2.1.3)
- numpy (^1.26.2)
- matplotlib (^3.8.2)
- seaborn (^0.13.0)
- py3dep (^0.16.2)
- pyarrow (^15.0.0)
- requests (^2.31.0)
- duckdb (^0.9.2)
- cartopy (^0.22.0)
- scikit-learn (^1.3.2)
- earthengine-api (^0.1.390)
- geemap (^0.31.0)
- statsmodels (^0.14.1)
- utm (^0.7.0)

Additional dependencies for specific functionality:
- ccrs (^1.0.0) - Cartographic projections
- matplotlib-scalebar (^0.8.1) - Map scale bars
- topojson/geojson (^1.7/^3.1.0) - GeoJSON processing
- geopy (^2.4.1) - Geocoding utilities
- osmnx (^1.9.1) - OpenStreetMap data
- segment-geospatial (^0.10.2) - Geospatial segmentation
- pycrs (^1.0.2) - CRS transformations
- eemont (^0.3.6) - Earth Engine utilities

## Development

For development:
```bash
# Install dev dependencies
poetry install --with dev

# Run tests
poetry run pytest

# Format code
poetry run black .
```
