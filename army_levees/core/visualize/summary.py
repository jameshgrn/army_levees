"""Summary visualization functions for levee data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_elevation_differences(segments: list, save_dir: Path) -> None:
    """Plot histogram of elevation differences between NLD and 3DEP."""
    diffs = []
    ratios = []  # Track ratios to identify potential unit conversion issues
    for segment in segments:
        segment_diffs = segment.elevation - segment.dep_elevation
        diffs.extend(segment_diffs)

        # Calculate ratios where both elevations are non-zero
        mask = (segment.elevation != 0) & (segment.dep_elevation != 0)
        if mask.any():
            segment_ratios = segment.elevation[mask] / segment.dep_elevation[mask]
            ratios.extend(segment_ratios)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot difference histogram
    ax1.hist(diffs, bins=50)
    ax1.set_xlabel("Elevation Difference (m)")
    ax1.set_ylabel("Count")
    ax1.set_title("Distribution of NLD vs 3DEP Elevation Differences")
    ax1.grid(True, alpha=0.3)

    # Plot ratio histogram
    ax2.hist(ratios, bins=50)
    ax2.set_xlabel("NLD/3DEP Ratio")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of NLD/3DEP Ratios")
    ax2.grid(True, alpha=0.3)

    # Add vertical line at ratio=3.28084 (feet to meters conversion)
    ax2.axvline(
        x=3.28084, color="r", linestyle="--", label="Feet to Meters Ratio (3.28084)"
    )
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_dir / "elevation_differences.png")
    plt.close()


def plot_segment_lengths(segments: list, save_dir: Path) -> None:
    """Plot histogram of segment lengths."""
    lengths = [segment.distance_along_track.max() for segment in segments]

    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50)
    plt.xlabel("Segment Length (m)")
    plt.ylabel("Count")
    plt.title("Distribution of Segment Lengths")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / "segment_lengths.png")
    plt.close()


def plot_point_counts(segments: list, save_dir: Path) -> None:
    """Plot histogram of points per segment."""
    counts = [len(segment) for segment in segments]

    plt.figure(figsize=(10, 6))
    plt.hist(counts, bins=50)
    plt.xlabel("Points per Segment")
    plt.ylabel("Count")
    plt.title("Distribution of Points per Segment")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / "point_counts.png")
    plt.close()
