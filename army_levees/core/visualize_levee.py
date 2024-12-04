"""Command-line interface for visualizing levee elevation data."""

import argparse
import random
from pathlib import Path
from tqdm import tqdm

from .visualize.individual import (diagnose_elevation_differences,
                                   plot_elevation_profile)
from .visualize.interactive import create_summary_map
from .visualize.utils import get_processed_systems


def main():
    parser = argparse.ArgumentParser(
        description="Create elevation comparison plots for levee systems"
    )

    # System ID selection group
    id_group = parser.add_mutually_exclusive_group()
    id_group.add_argument("system_id", nargs="?", help="USACE system ID to plot")
    id_group.add_argument(
        "-r", "--random", action="store_true", help="Use a random levee system"
    )
    id_group.add_argument(
        "-a", "--all", action="store_true", help="Process all available systems"
    )

    # Action group
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "-p", "--plot", action="store_true", help="Create plots for the system(s)"
    )
    action_group.add_argument(
        "-s",
        "--summary",
        action="store_true",
        help="Create summary plots for all processed levees",
    )
    action_group.add_argument(
        "-d", "--diagnose", action="store_true", help="Run diagnostics on the system(s)"
    )
    action_group.add_argument(
        "-m", "--map", action="store_true", help="Create interactive summary map"
    )

    # Data source selection
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/segments",
        help="Directory containing levee data (default: data/segments)"
    )
    parser.add_argument(
        "--raw-data",
        action="store_true",
        help="Use raw data from data/processed instead of filtered segments"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="plots",
        help="Directory to save plots (default: plots)"
    )

    args = parser.parse_args()

    # If using raw data, switch to processed directory
    data_dir = "data/processed" if args.raw_data else args.data_dir

    # Get system ID(s)
    if args.all:
        system_ids = get_processed_systems(data_dir=data_dir)
        if not system_ids:
            print("No levee files found.")
            exit(1)
        print(f"Processing {len(system_ids)} systems...")
    elif args.random:
        # Get list of processed systems
        available_systems = get_processed_systems(data_dir=data_dir)
        if not available_systems:
            print("No levee files found.")
            exit(1)

        # Pick a random system that exists
        while True:
            system_id = random.choice(list(available_systems))
            file_path = Path(data_dir) / f"levee_{system_id}.parquet"
            if file_path.exists():
                system_ids = [system_id]
                print(f"Randomly selected system ID: {system_id}")
                break
    else:
        if args.system_id is None:
            parser.error("Either system_id, -r/--random, or -a/--all is required")
            
        # Check if file exists
        file_path = Path(data_dir) / f"levee_{args.system_id}.parquet"
        if not file_path.exists():
            print(f"Error: No data found for system {args.system_id}")
            print(f"File not found: {file_path}")
            exit(1)
            
        system_ids = [args.system_id]

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Execute requested action
    if args.diagnose:
        for system_id in system_ids:
            file_path = Path(data_dir) / f"levee_{system_id}.parquet"
            if not file_path.exists():
                print(f"Warning: Could not find profile data for system {system_id}")
                continue
            print(f"\nDiagnosing system {system_id}:")
            diagnose_elevation_differences(system_id, data_dir=data_dir)
    elif args.summary:
        # Create summary plots
        from .visualize.summary import plot_summary
        plot_summary(save_dir=save_dir, data_dir=data_dir)
    elif args.map:
        # Create interactive map
        map_path = str(save_dir / "levee_summary_map.html")
        summary_map = create_summary_map(save_path=map_path, data_dir=data_dir)
        if summary_map:
            print(f"Saved summary map to {map_path}")
    else:  # plot
        for system_id in tqdm(system_ids, desc="Creating plots"):
            file_path = Path(data_dir) / f"levee_{system_id}.parquet"
            if not file_path.exists():
                print(f"Warning: Could not find profile data for system {system_id}")
                continue
            plot_elevation_profile(system_id, save_dir=save_dir, data_dir=data_dir)

if __name__ == "__main__":
    main()
