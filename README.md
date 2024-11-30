# Army Levees Analysis

A Python package for analyzing elevation differences between the National Levee Database (NLD) and USGS 3DEP data.

## Overview

This package helps collect and analyze elevation data for USACE levee systems by:
1. Getting profile data from the NLD API
2. Getting matching elevations from USGS 3DEP
3. Comparing and visualizing the differences

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
            V1[NLD elevation > 1.0m] --> VM[Validation Mask]
            V2[3DEP elevation > 1.0m] --> VM
            V3[No missing NLD data] --> VM
            V4[No missing 3DEP data] --> VM
        end

        subgraph Segment_Processing
            V0 --> S1[Apply Validation Mask]
            VM -->|filter| S1
            S1 --> S2[Sort by Distance]
            S2 --> S3[Calculate Distance Gaps]
            S3 -->|Gap > 100m| S4[Split into Sections]
            S4 --> S5[Remove Outliers]
            S5 -->|z-score > 3| S6[Final Valid Segments]
        end

        subgraph Floodwall_Check
            V0 -->|check segments| FW[Has Floodwalls?]
            FW -->|yes| Skip[Skip System]
            FW -->|no| S1
        end

        S6 --> G
    end

    subgraph Data_Storage
        F -->|save_parquet| P[(Processed Data)]
    end

    linkStyle default stroke-width:2px

    style Data_Collection fill:#f4f4f4,stroke:#333,stroke-width:2px
    style Data_Processing fill:#e8f4ea,stroke:#333,stroke-width:2px
    style Filtering_Process fill:#f4e8ea,stroke:#333,stroke-width:2px
    style Data_Storage fill:#f4f4e8,stroke:#333,stroke-width:2px

    style Validation_Checks fill:#fff,stroke:#333,stroke-width:2px
    style Segment_Processing fill:#fff,stroke:#333,stroke-width:2px
    style Floodwall_Check fill:#fff,stroke:#333,stroke-width:2px
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

2. **Analysis**
   - Calculates elevation differences
   - Removes outliers using z-score filtering
   - Generates comprehensive statistics
   - Identifies problematic sections

3. **Visualization**
   - Individual levee plots showing:
     * Elevation profiles (NLD vs 3DEP)
     * Difference distributions
     * Coverage statistics
   - Summary plots showing:
     * Distribution of levee lengths
     * Mean elevation differences vs length
     * CDF of differences
     * Boxplots of positive/negative differences
     * Distribution of mean differences
     * Valid data coverage

## Usage Examples

### 1. Get Random Levee Samples
```python
from army_levees import get_random_levees

# Get 10 new random samples (skips already processed systems)
results = get_random_levees(n_samples=10)
```

### 2. Plot Individual Levee System
```python
from army_levees import plot_levee_system

# Create visualization for a specific system
plot_levee_system("5205000591", save_dir="plots")
```

### 3. Analyze Existing Dataset
```python
from army_levees.core.visualize_levee import plot_summary

# Generate summary plots and statistics
plot_summary(save_dir="plots")
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
│       └── visualize_levee.py  # Visualization
├── data/
│   └── processed/       # Processed parquet files
│       ├── levee_*.parquet  # Individual system data
│       └── dataset_summary.csv
├── docs/               # Documentation
├── plots/             # Generated plots
├── tests/             # Test suite
├── pyproject.toml     # Poetry configuration
└── README.md
```

## Data Format

Each parquet file contains:
- `system_id`: USACE system ID
- `elevation`: NLD elevation (meters, converted from feet)
- `dep_elevation`: 3DEP elevation (meters)
- `difference`: NLD - 3DEP (meters)
- `distance_along_track`: Distance along levee (meters)
- `geometry`: Point geometry (EPSG:4326)

## Contributing

To add more samples to the dataset:
1. Install the package as above
2. Run `poetry run python scripts/sample_levees.py -n 10`
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
