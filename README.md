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
    %% Main Functions
    A[get_random_levees] --> B[analyze_levee_system]
    B --> C[get_nld_profile]
    B --> D[get_3dep_elevations]
    B --> E[save parquet]
    
    %% NLD Data Flow
    C --> F[NLD API]
    F --> G[Extract Points]
    G --> H[Convert Units]
    H --> I[Create GeoDataFrame]
    
    %% 3DEP Data Flow
    D --> J[py3dep API]
    J --> K[Match Points]
    K --> L[Add to GeoDataFrame]
    
    %% Data Storage
    E --> M[(data/processed/)]
    M --> N[levee_SYSTEMID.parquet]
    M --> O[dataset_summary.csv]
    
    %% Visualization
    P[plot_levee_system] --> Q[Load parquet]
    Q --> R[Plot Profiles]
    Q --> S[Plot Differences]
    
    %% Data Structure
    N --> |Contains|T[system_id]
    N --> |Contains|U[nld_elevation]
    N --> |Contains|V[dep_elevation]
    N --> |Contains|W[difference]
    N --> |Contains|X[distance_along_track]
    N --> |Contains|Y[geometry]
    
    %% Coordinate Systems
    Z1[EPSG:3857] --> |Convert|Z2[EPSG:4326]
    
    %% Unit Conversions
    U1[NLD feet] --> |x 0.3048|U2[meters]
    
    %% Style
    classDef api fill:#f9f,stroke:#333,stroke-width:2px
    classDef data fill:#bbf,stroke:#333,stroke-width:2px
    classDef process fill:#bfb,stroke:#333,stroke-width:2px
    
    class F,J api
    class M,N,O data
    class G,H,I,K,L,R,S process
```

## Installation

```bash
# Clone the repository
git clone <repo>
cd army_levees

# Install dependencies with Poetry
poetry install
```

## Usage

The package provides a simple interface to collect and analyze levee data:

```python
from army_levees import get_random_levees, plot_levee_system

# Get new samples (skips already processed systems by default)
results = get_random_levees(n_samples=10)

# Plot a specific system
plot_levee_system("5205000591", save_dir="plots")
```

For a complete example, run:
```bash
poetry run python examples/demo.py
```

## Data Structure

```
army_levees/
├── data/
│   └── processed/          # Processed parquet files
│       ├── levee_*.parquet # Individual system data
│       └── dataset_summary.csv
└── plots/                  # Generated plots
```

Each parquet file contains:
- `system_id`: USACE system ID
- `nld_elevation`: Elevation from NLD (meters)
- `dep_elevation`: Elevation from 3DEP (meters)
- `difference`: NLD - 3DEP (meters)
- `distance_along_track`: Distance along levee (meters)
- `geometry`: Point geometry (EPSG:4326)

## Documentation

See the `docs/` directory for:
- Comparative Analysis Report
- ERDC Updates
- Presentations
- Technical Guides

## Contributing

To add more samples to the dataset:
1. Install the package as above
2. Run `poetry run python examples/demo.py`
3. New samples will be added to `data/processed/`

The script will:
- Skip systems that are already processed
- Show dataset statistics
- Generate visualizations
- Save a summary CSV
