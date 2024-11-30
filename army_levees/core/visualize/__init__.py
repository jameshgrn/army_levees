"""Visualization functions for army_levees."""

from .individual import diagnose_elevation_differences, plot_elevation_profile
from .interactive import create_summary_map, create_system_map
from .summary import (analyze_geographic_patterns, analyze_large_differences,
                      collect_system_statistics,
                      investigate_problematic_systems)

__all__ = [
    "collect_system_statistics",
    "analyze_large_differences",
    "investigate_problematic_systems",
    "analyze_geographic_patterns",
    "plot_elevation_profile",
    "diagnose_elevation_differences",
    "create_system_map",
    "create_summary_map",
]
