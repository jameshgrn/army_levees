"""Command-line interface for visualizing levee elevation data."""

import argparse
import random
from pathlib import Path

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

    # Action group
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "-p", "--plot", action="store_true", help="Create plots for the system"
    )
    action_group.add_argument(
        "-s",
        "--summary",
        action="store_true",
        help="Create summary plots for all processed levees",
    )
    action_group.add_argument(
        "-d", "--diagnose", action="store_true", help="Run diagnostics on the system"
    )
    action_group.add_argument(
        "-m", "--map", action="store_true", help="Create interactive summary map"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="plots",
        help="Directory to save plots (default: plots)",
    )

    args = parser.parse_args()

    # Get system ID if needed
    if not (args.summary or args.map):
        if args.random:
            # Get list of processed levee files
            processed_systems = get_processed_systems()
            if not processed_systems:
                print("No processed levee files found.")
                exit(1)

            # Pick a random system
            system_id = random.choice(processed_systems)
            print(f"Randomly selected system ID: {system_id}")
        else:
            if args.system_id is None:
                parser.error("Either system_id or -r/--random is required")
            system_id = args.system_id

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Execute requested action
    if args.diagnose:
        diagnose_elevation_differences(system_id)
    elif args.summary:
        # Create summary plots
        from .visualize.summary import plot_summary

        plot_summary(save_dir=save_dir)
    elif args.map:
        # Create interactive map
        map_path = str(save_dir / "levee_summary_map.html")
        summary_map = create_summary_map(save_path=map_path)
        if summary_map:
            print(f"Saved summary map to {map_path}")
    else:  # plot
        plot_elevation_profile(system_id, save_dir=save_dir)


if __name__ == "__main__":
    main()
