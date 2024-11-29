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
    A[National Levee Database API] -->|Async HTTP Requests| B[sample_levees.py]
    D[USGS 3DEP Elevation API] -->|Elevation Queries| B

    %% Core Processing Details
    subgraph "sample_levees.py Processing"
        B -->|1. Get System IDs| E[get_usace_system_ids]
        B -->|2. Get Profile| F[get_nld_profile_async]
        B -->|3. Get Elevations| G[get_3dep_elevations_async]
        B -->|4. Process Data| H[process_system_async]

        E -->|Returns List| F
        F -->|Returns Coords| G
        G -->|Returns Elevs| H

        H -->|Creates| H1[GeoDataFrame]
        H1 -->|Contains| H2[system_id<br/>elevation<br/>dep_elevation<br/>difference<br/>geometry]
    end

    %% Data Storage
    H -->|Save| I[(Parquet Files<br/>data/processed/*.parquet)]

    %% Visualization Pipeline
    subgraph "visualize_levee.py Processing"
        I -->|Load| J[plot_levee_system]
        I -->|Load Multiple| S[plot_summary]

        J -->|Creates| K[Individual Plots]
        K -->|Top Plot| K1[Elevation Profile<br/>NLD vs 3DEP]
        K -->|Bottom Plot| K2[Difference<br/>Distribution]

        S -->|Creates| L[Summary Plots]
        L -->|Plot 1| L1[Levee Lengths<br/>Distribution]
        L -->|Plot 2| L2[Mean Difference<br/>vs Length]
        L -->|Plot 3| L3[Difference CDF]
        L -->|Plot 4| L4[Positive/Negative<br/>Differences]
        L -->|Plot 5| L5[Mean Difference<br/>Distribution]
        L -->|Plot 6| L6[Valid Data<br/>Coverage]
    end

    %% Data Flow Details
    subgraph "Data Processing Steps"
        direction LR
        Z1[1. Fetch Raw Data] --> Z2[2. Convert Units<br/>feet → meters]
        Z2 --> Z3[3. Filter Invalid<br/>Points]
        Z3 --> Z4[4. Calculate<br/>Differences]
        Z4 --> Z5[5. Save to<br/>Parquet]
    end

    %% Error Handling
    subgraph "Error Handling"
        Y1[Retry Logic] --> Y2[HTTP Timeouts]
        Y2 --> Y3[Invalid Data<br/>Filtering]
        Y3 --> Y4[Missing Value<br/>Handling]
    end

    %% Style Definitions
    classDef api fill:#f9f,stroke:#333,stroke-width:2px,rx:5px
    classDef process fill:#bfb,stroke:#333,stroke-width:2px,rx:5px
    classDef storage fill:#bbf,stroke:#333,stroke-width:2px,rx:5px
    classDef viz fill:#fbb,stroke:#333,stroke-width:2px,rx:5px
    classDef error fill:#ffb,stroke:#333,stroke-width:2px,rx:5px
    classDef step fill:#ddd,stroke:#333,stroke-width:1px,rx:5px

    %% Apply Styles
    class A,D api
    class B,E,F,G,H process
    class I storage
    class J,K,L,K1,K2,L1,L2,L3,L4,L5,L6 viz
    class Y1,Y2,Y3,Y4 error
    class Z1,Z2,Z3,Z4,Z5 step
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
