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
    %% Data Sources
    A[NLD API] -->|Raw Profile Data| B[sample_levees.py]
    D[3DEP API] -->|Elevation Data| B
    
    %% Core Processing
    subgraph Core Processing [army_levees/core]
        B -->|Process Samples| E[sampling.py]
        E -->|Get Elevations| F[elevation.py]
        F --> G[plot_modular.py]
        H[nld_api.py] -->|Fetch Data| B
    end
    
    %% Data Storage
    G -->|Save| I[(data/processed/*.parquet)]
    
    %% Data Structure
    subgraph Parquet Schema
        I --> |Contains| J[system_id]
        I --> |Contains| K[elevation]
        I --> |Contains| L[dep_elevation]
        I --> |Contains| M[difference]
        I --> |Contains| N[distance_along_track]
        I --> |Contains| O[geometry]
    end
    
    %% Analysis & Visualization
    I --> P[analyze_levees.py]
    P --> Q[Visualization Outputs]
    Q --> R[plots/*.pdf]
    
    %% Entry Points
    S[get_random_samples.py] -->|Trigger| B
    
    %% Coordinate Systems
    subgraph CRS Handling
        T[EPSG:3857] -->|transform| U[EPSG:4326]
        U -->|used by| V[3DEP Sampling]
        U -->|stored in| I
    end
    
    %% Style
    classDef api fill:#f9f,stroke:#333,stroke-width:2px
    classDef data fill:#bbf,stroke:#333,stroke-width:2px
    classDef process fill:#bfb,stroke:#333,stroke-width:2px
    classDef crs fill:#ffd,stroke:#333,stroke-width:2px
    classDef output fill:#fbb,stroke:#333,stroke-width:2px
    
    class A,D api
    class I data
    class B,E,F,G process
    class T,U,V crs
    class R output
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

## Understanding the Visualizations

### Individual Levee Plots
- **Top Plot**: Shows elevation profiles
  * Blue line: NLD elevation
  * Red line: 3DEP elevation
  * Gaps indicate filtered sections (e.g., zero elevations)
  * Stats show coverage and mean differences

- **Bottom Plot**: Shows difference distribution
  * Histogram of NLD - 3DEP differences
  * Red line at zero for reference
  * Helps identify systematic biases

### Summary Plots
1. **Distribution of Levee Lengths**
   - Histogram showing system size distribution
   - Helps identify typical levee lengths

2. **Mean Elevation Difference vs Length**
   - Scatter plot of differences vs system length
   - Shows if longer systems have different characteristics

3. **CDF of Differences**
   - Shows cumulative distribution of differences
   - Helps identify overall bias and spread

4. **Boxplot of Differences**
   - Separates positive and negative differences
   - Shows median, quartiles, and outliers

5. **Distribution of Mean Differences**
   - Histogram with KDE showing difference distribution
   - Red line at zero for reference

6. **Valid Data Coverage**
   - Shows percentage of valid comparison points
   - Higher is better (typically >95%)

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

## Common Issues & Solutions

1. **Zero Elevations**
   - Problem: Some sections show zero elevation in either dataset
   - Solution: Automatically filtered out in analysis

2. **Missing Data**
   - Problem: Gaps in elevation data
   - Solution: Sections split at gaps >100m

3. **Outliers**
   - Problem: Extreme elevation differences
   - Solution: Z-score filtering (threshold = 3)

4. **Floodwalls**
   - Problem: Different characteristics than levees
   - Solution: Automatically detected and excluded

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