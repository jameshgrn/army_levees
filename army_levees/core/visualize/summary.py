"""Functions for creating summary visualizations across all levee systems."""

import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configure matplotlib
plt.style.use("default")
plt.rcParams["figure.figsize"] = [10, 6]
plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.size"] = 12

logger = logging.getLogger(__name__)


def plot_levee_profile(data: gpd.GeoDataFrame) -> None:
    """Plot levee profile with both NLD and 3DEP elevations.

    Args:
        data: GeoDataFrame with levee data
    """
    # Calculate elevation changes
    data["elevation_diff"] = data["elevation"].diff()
    spike_mask = abs(data["elevation_diff"]) > 100

    # Plot normal points
    plt.plot(
        data["distance_along_track"], data["elevation"], "b-", label="NLD", alpha=0.7
    )
    plt.plot(
        data["distance_along_track"],
        data["dep_elevation"],
        "r-",
        label="3DEP",
        alpha=0.7,
    )

    # Highlight spikes in red
    if spike_mask.any():
        plt.scatter(
            data[spike_mask]["distance_along_track"],
            data[spike_mask]["elevation"],
            color="red",
            s=50,
            alpha=0.5,
            label="Elevation Spikes",
        )

    # Add mean difference to title
    mean_diff = (data["elevation"] - data["dep_elevation"]).mean()
    plt.title(f"Mean Elevation Difference: {mean_diff:.1f}m")

    plt.xlabel("Distance Along Track (m)")
    plt.ylabel("Elevation (m)")
    plt.legend()
    plt.grid(True)


def collect_system_statistics() -> pd.DataFrame:
    """Collect statistics for all levee systems.

    Returns:
        DataFrame with system-level statistics
    """
    stats_list = []
    invalid_systems = []

    # Get list of all processed files
    processed_dir = Path("data/processed")
    levee_files = list(processed_dir.glob("levee_*.parquet"))

    for file_path in levee_files:
        try:
            # Extract system ID from filename
            system_id = file_path.stem.split("_")[1]

            # Load data
            data = gpd.read_parquet(file_path)

            # Skip if no valid points after filtering zeros
            valid_data = data[data["elevation"] > 0].copy()
            if len(valid_data) == 0:
                invalid_systems.append(system_id)
                continue

            # Calculate statistics - Note: mean_diff is now 3DEP - NLD
            stats = {
                "system_id": system_id,
                "n_points": len(valid_data),
                "total_length": valid_data["distance_along_track"].max(),
                "min_elevation": valid_data["elevation"].min(),
                "max_elevation": valid_data["elevation"].max(),
                "mean_elevation": valid_data["elevation"].mean(),
                "mean_diff": (
                    valid_data["dep_elevation"] - valid_data["elevation"]
                ).mean(),  # Changed order
                "rmse": np.sqrt(
                    (
                        (valid_data["dep_elevation"] - valid_data["elevation"]) ** 2
                    ).mean()
                ),  # Changed order
            }

            stats_list.append(stats)

        except Exception as e:
            logger.error(f"Error processing system {system_id}: {str(e)}")
            continue

    if invalid_systems:
        logger.info(f"\nSkipped {len(invalid_systems)} systems with no valid points:")
        logger.info(f"Example systems: {', '.join(invalid_systems[:5])}...")

    if not stats_list:
        logger.error("No valid systems found")
        return pd.DataFrame()

    return pd.DataFrame(stats_list)


def analyze_large_differences(threshold: float = 5.0) -> pd.DataFrame:
    """Analyze systems with large mean elevation differences.

    Args:
        threshold: Absolute difference threshold in meters

    Returns:
        DataFrame with statistics for systems exceeding threshold
    """
    df = collect_system_statistics()

    if df.empty:
        logger.error("No valid systems to analyze")
        return df

    # Filter systems with large differences
    large_diff = df[df["mean_diff"].abs() >= threshold].copy()

    if large_diff.empty:
        logger.info(f"No systems found with mean differences >= {threshold}m")
        return large_diff

    # Sort by absolute mean difference
    large_diff["abs_mean_diff"] = large_diff["mean_diff"].abs()
    large_diff = large_diff.sort_values("abs_mean_diff", ascending=False)

    # Add percentage difference (handle zero mean elevations)
    large_diff["pct_diff"] = 0.0  # default value
    mask = (
        large_diff["mean_elevation"] > 1.0
    )  # Only calculate percentage for meaningful elevations
    large_diff.loc[mask, "pct_diff"] = (
        large_diff.loc[mask, "mean_diff"] / large_diff.loc[mask, "mean_elevation"]
    ) * 100

    # Select and reorder columns
    cols = [
        "system_id",
        "mean_diff",
        "abs_mean_diff",
        "pct_diff",
        "mean_elevation",
        "min_elevation",
        "max_elevation",
        "rmse",
        "total_length",
        "n_points",
    ]

    result = large_diff[cols].copy()

    # Log summary
    logger.info(f"\nSystems with mean differences >= {threshold}m:")
    logger.info(f"Total systems: {len(result)}")
    logger.info(f"Mean absolute difference: {result['abs_mean_diff'].mean():.2f}m")
    logger.info(f"Max absolute difference: {result['abs_mean_diff'].max():.2f}m")
    logger.info(
        f"Mean percent difference (for elevations > 1m): {result.loc[mask, 'pct_diff'].mean():.1f}%"
    )
    logger.info("\nTop 10 largest differences:")

    for _, row in result.head(10).iterrows():
        if row["mean_elevation"] <= 1.0:
            pct_str = "N/A%"
        else:
            pct_str = f"{row['pct_diff']:.1f}%"

        logger.info(
            f"System {row['system_id']}: "
            f"diff={row['mean_diff']:.1f}m ({pct_str}), "
            f"NLD={row['mean_elevation']:.1f}m, "
            f"min={row['min_elevation']:.1f}m, "
            f"max={row['max_elevation']:.1f}m, "
            f"length={row['total_length']:.0f}m"
        )

    return result


def investigate_problematic_systems(n_systems: int = 5) -> None:
    """Investigate systems with the largest elevation differences.

    Args:
        n_systems: Number of systems to investigate
    """
    from .individual import (diagnose_elevation_differences,
                             plot_elevation_profile)

    # Get systems with large differences
    df = analyze_large_differences(threshold=5.0)

    # Take top N systems
    systems = df.head(n_systems)

    logger.info(f"\nInvestigating top {n_systems} systems with largest differences:")

    # Create plots directory
    plots_dir = Path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    for _, row in systems.iterrows():
        system_id = row["system_id"]
        logger.info(f"\nSystem {system_id}:")

        # Run diagnostics
        diagnose_elevation_differences(system_id)

        # Create detailed plot
        plot_elevation_profile(system_id, save_dir=plots_dir)

        logger.info("-" * 80)


def analyze_geographic_patterns(threshold: float = 800.0) -> None:
    """Analyze geographic patterns of systems with large elevation differences.

    Args:
        threshold: Absolute difference threshold in meters
    """
    # Get systems with large differences
    logger.info("Collecting system statistics...")
    df = collect_system_statistics()

    if df.empty:
        logger.error("No valid systems to analyze")
        return

    logger.info(f"Found {len(df)} total systems")

    # Filter systems with large differences
    large_diff = df[df["mean_diff"].abs() >= threshold].copy()

    if large_diff.empty:
        logger.info(f"No systems found with mean differences >= {threshold}m")
        return

    logger.info(f"Found {len(large_diff)} systems with differences >= {threshold}m")

    # Load first point of each system to get location
    for idx, row in large_diff.iterrows():
        try:
            system_id = row["system_id"]
            logger.info(f"Processing system {system_id}...")
            data = gpd.read_parquet(f"data/processed/levee_{system_id}.parquet")
            # Get first point's coordinates
            point = data.iloc[0].geometry
            large_diff.loc[idx, "longitude"] = point.x
            large_diff.loc[idx, "latitude"] = point.y
        except Exception as e:
            logger.error(f"Error processing system {system_id}: {str(e)}")
            continue

    # Group by first two digits of system ID
    large_diff["region"] = large_diff["system_id"].str[:2]
    region_stats = (
        large_diff.groupby("region")
        .agg(
            {
                "system_id": "count",
                "mean_diff": ["mean", "std"],
                "latitude": ["mean", "std"],
                "longitude": ["mean", "std"],
            }
        )
        .round(2)
    )

    logger.info("\nGeographic patterns in problematic systems:")
    logger.info(f"\nRegion statistics (for differences >= {threshold}m):")
    for region, stats in region_stats.iterrows():
        logger.info(f"\nRegion {region}:")
        logger.info(f"Number of systems: {stats[('system_id', 'count')]}")
        logger.info(
            f"Mean difference: {stats[('mean_diff', 'mean')]:.1f}m ± {stats[('mean_diff', 'std')]:.1f}m"
        )
        logger.info(
            f"Mean location: ({stats[('latitude', 'mean')]:.2f}°N, {stats[('longitude', 'mean')]:.2f}°W)"
        )

    return large_diff


def format_consistency_panel(ax):
    """Format the change consistency panel."""
    ax.set_title(
        "Change Pattern Consistency by Category\n"
        "(% points agreeing with mean change direction)"
    )
    ax.set_xlabel("Change Category")
    ax.set_ylabel("Change Consistency (%)")

    # Add reference lines
    ax.axhline(y=50, color="r", linestyle="--", alpha=0.5)
    ax.axhline(y=20, color="gray", linestyle=":", alpha=0.3)
    ax.axhline(y=80, color="gray", linestyle=":", alpha=0.3)

    # Move explanatory text to bottom of plot
    ax.text(
        0.02,
        0.02,  # Changed from 0.98 to 0.02 for bottom placement
        "100% = all points changing in same direction\n"
        "80% = strong directional consistency\n"
        "50% = random/mixed changes\n"
        "20% = strong opposing changes\n"
        "0% = all points changing in opposite direction",
        transform=ax.transAxes,
        verticalalignment="bottom",  # Changed from 'top' to 'bottom'
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )


def plot_summary(save_dir: Path = Path("plots")) -> None:
    """Create a 6-panel summary plot of levee statistics focused on geomorphic change.

    Args:
        save_dir: Directory to save the plot
    """
    # Get system statistics
    df = collect_system_statistics()

    # Calculate additional statistics
    df["degradation"] = df["mean_diff"].apply(
        lambda x: "Degradation" if x < -0.5 else "Aggradation" if x > 0.5 else "Stable"
    )
    df["relative_change"] = (
        df["mean_diff"] / df["mean_elevation"]
    ) * 100  # percent change

    # Calculate change consistency (what percentage of points agree with mean change direction)
    def calculate_change_consistency(row):
        """Calculate what percentage of points agree with the overall change direction.

        Returns:
            float: Percentage of points (0-100) that agree with mean change direction
        """
        try:
            data = gpd.read_parquet(f"data/processed/levee_{row['system_id']}.parquet")
            # Calculate point-wise changes
            point_changes = data["dep_elevation"] - data["elevation"]
            mean_change = point_changes.mean()

            # If mean change is negligible, calculate variance instead
            if abs(mean_change) < 0.1:  # Increased threshold for stability
                # For stable sections, look at variance
                variance = point_changes.var()
                if variance < 0.01:  # Very stable
                    return 100.0
                else:  # Mixed changes but small magnitude
                    return 75.0

            # Calculate agreement with mean change direction
            total_points = len(point_changes)
            if total_points == 0:
                return 50.0

            # Count points in same direction (including exact matches)
            same_direction = (
                (point_changes * mean_change > 0)
                | (abs(point_changes) < 0.01)  # Count very small changes as agreeing
            ).sum()

            # Calculate percentage
            consistency = (same_direction / total_points) * 100

            return consistency

        except Exception as e:
            logger.warning(
                f"Error calculating consistency for system {row['system_id']}: {e}"
            )
            return 50.0  # Return neutral value on error

    df["change_consistency"] = df.apply(calculate_change_consistency, axis=1)
    print(df["change_consistency"].describe())

    # Remove extreme outliers (beyond 99th percentile)
    for col in ["mean_diff", "relative_change", "rmse", "change_consistency"]:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[f"{col}_clipped"] = df[col].clip(lower, upper)

    # Create figure with 6 subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"Levee System Geomorphic Change (n={len(df)})\n3DEP (T₂) vs NLD (T₁) Elevation Comparison",
        fontsize=14,
    )

    # Panel 1: Elevation difference histogram with geomorphic categories
    sns.histplot(
        data=df,
        x="mean_diff_clipped",
        hue="degradation",
        multiple="stack",
        bins=50,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title(
        "Distribution of Mean Elevation Changes\nby Geomorphic Change Category"
    )
    axes[0, 0].set_xlabel(
        "3DEP - NLD Elevation (m)\n(negative = degradation, positive = aggradation)"
    )
    axes[0, 0].set_ylabel("Count")

    # Panel 2: Cumulative distribution of changes
    sorted_changes = np.sort(df["mean_diff_clipped"])
    cumulative = np.arange(1, len(sorted_changes) + 1) / len(sorted_changes)
    axes[0, 1].plot(sorted_changes, cumulative, "b-")
    axes[0, 1].set_title("Cumulative Distribution of Elevation Changes")
    axes[0, 1].set_xlabel("Elevation Change (m)")
    axes[0, 1].set_ylabel("Cumulative Proportion")
    axes[0, 1].grid(True)
    axes[0, 1].axvline(x=0, color="r", linestyle="--", alpha=0.5)

    # Add summary statistics to CDF plot
    stats_text = (
        f"Summary Statistics:\n"
        f"μ = {df['mean_diff'].mean():.2f}m ± {df['mean_diff'].std():.2f}m\n"
        f"median = {df['mean_diff'].median():.2f}m\n"
        f"1st-99th: [{df['mean_diff'].quantile(0.01):.1f}, {df['mean_diff'].quantile(0.99):.1f}]m\n"
        f"\nChange Categories:\n"
        f"Degradation: {(df['degradation'] == 'Degradation').mean()*100:.0f}%\n"
        f"Stable: {(df['degradation'] == 'Stable').mean()*100:.0f}%\n"
        f"Aggradation: {(df['degradation'] == 'Aggradation').mean()*100:.0f}%"
    )
    axes[0, 1].text(
        0.05,
        0.95,
        stats_text,
        transform=axes[0, 1].transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    # Panel 3: Change vs Elevation with density
    sns.kdeplot(
        data=df,
        x="mean_elevation",
        y="mean_diff_clipped",
        cmap="viridis",
        fill=True,
        ax=axes[0, 2],
    )
    axes[0, 2].set_title("Elevation Change vs Mean Elevation\n(density contours)")
    axes[0, 2].set_xlabel("Mean Elevation (m)")
    axes[0, 2].set_ylabel("Elevation Change (m)")
    axes[0, 2].axhline(y=0, color="r", linestyle="--", alpha=0.5)

    # Panel 4: Change magnitude distribution
    sns.boxplot(data=df, y="mean_diff_clipped", x="degradation", ax=axes[1, 0])
    axes[1, 0].set_title("Change Magnitude by Category")
    axes[1, 0].set_xlabel("Change Category")
    axes[1, 0].set_ylabel("Elevation Change (m)")

    # Panel 5: Relative change distribution
    sns.violinplot(data=df, y="relative_change_clipped", x="degradation", ax=axes[1, 1])
    axes[1, 1].set_title("Relative Change Distribution by Category")
    axes[1, 1].set_xlabel("Change Category")
    axes[1, 1].set_ylabel("Relative Change (%)")

    # Panel 6: Change consistency by category
    sns.boxplot(data=df, y="change_consistency_clipped", x="degradation", ax=axes[1, 2])
    format_consistency_panel(axes[1, 2])

    # Adjust layout and save
    plt.tight_layout()
    save_path = save_dir / "levee_summary_statistics.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

    logger.info(f"Saved summary statistics plot to {save_path}")

    # Log additional statistics
    logger.info("\nDetailed Change Statistics:")
    logger.info(f"Total systems analyzed: {len(df)}")
    logger.info(
        f"Mean absolute change: {df['mean_diff'].abs().mean():.2f}m ± {df['mean_diff'].abs().std():.2f}m"
    )
    logger.info(f"Median absolute change: {df['mean_diff'].abs().median():.2f}m")
    logger.info(
        f"Systems showing significant degradation (>0.5m): {(df['mean_diff'] < -0.5).sum()} ({(df['mean_diff'] < -0.5).mean()*100:.1f}%)"
    )
    logger.info(
        f"Systems showing significant aggradation (>0.5m): {(df['mean_diff'] > 0.5).sum()} ({(df['mean_diff'] > 0.5).mean()*100:.1f}%)"
    )
    logger.info(f"Extreme changes (>2m): {(abs(df['mean_diff']) > 2).sum()} systems")

    # Log change category statistics
    logger.info("\nChange Category Statistics:")
    logger.info(
        f"Degradation (< -0.5m): {(df['degradation'] == 'Degradation').sum()} systems "
        f"({(df['degradation'] == 'Degradation').mean()*100:.1f}%)"
    )
    logger.info(
        f"Stable (-0.5m to 0.5m): {(df['degradation'] == 'Stable').sum()} systems "
        f"({(df['degradation'] == 'Stable').mean()*100:.1f}%)"
    )
    logger.info(
        f"Aggradation (> 0.5m): {(df['degradation'] == 'Aggradation').sum()} systems "
        f"({(df['degradation'] == 'Aggradation').mean()*100:.1f}%)"
    )
