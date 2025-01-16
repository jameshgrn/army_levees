"""Main entry point for visualization module."""

import argparse
import logging
import sys
from pathlib import Path

from .utils import (
    plot_elevation_profile,
    diagnose_elevation_differences,
    get_processed_systems,
    calculate_system_difference
)
from .dash_app import create_dash_app
from .multi_profile_plot import main as plot_multi_profiles

logger = logging.getLogger(__name__)

def main():
    """Main entry point for visualization."""
    # Store original argv
    original_argv = sys.argv[:]

    parser = argparse.ArgumentParser(
        description="Visualize levee elevation data"
    )

    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Dashboard command
    dash_parser = subparsers.add_parser('dashboard', help='Run interactive dashboard')
    dash_parser.add_argument(
        "--data-dir",
        type=str,
        default="data/segments",
        help="Directory containing levee segment data",
    )
    dash_parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port to run the dashboard on",
    )
    dash_parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode",
    )
    dash_parser.add_argument(
        "--raw",
        action="store_true",
        help="Use raw data instead of processed data",
    )

    # Plot command
    plot_parser = subparsers.add_parser('plot', help='Plot a single system')
    plot_parser.add_argument(
        'system_id',
        type=str,
        help='System ID to plot',
    )
    plot_parser.add_argument(
        "--data-dir",
        type=str,
        default="data/segments",
        help="Directory containing levee segment data",
    )
    plot_parser.add_argument(
        "--save-dir",
        type=str,
        default="plots",
        help="Directory to save plots",
    )
    plot_parser.add_argument(
        "--raw",
        action="store_true",
        help="Use raw data instead of processed data",
    )
    plot_parser.add_argument(
        "--show",
        action="store_true",
        help="Show plot instead of saving",
    )

    # Diagnose command
    diag_parser = subparsers.add_parser('diagnose', help='Print diagnostic information')
    diag_parser.add_argument(
        'system_id',
        type=str,
        help='System ID to diagnose',
    )
    diag_parser.add_argument(
        "--data-dir",
        type=str,
        default="data/segments",
        help="Directory containing levee segment data",
    )
    diag_parser.add_argument(
        "--raw",
        action="store_true",
        help="Use raw data instead of processed data",
    )

    # Multi-profile command
    multi_parser = subparsers.add_parser('multi', help='Plot multiple profiles')
    multi_parser.add_argument(
        "--type",
        type=str,
        choices=['all', 'significant', 'non_significant'],
        default='all',
        help="Type of profiles to plot",
    )
    multi_parser.add_argument(
        "--data-dir",
        type=str,
        default="data/segments",
        help="Directory containing filtered segments",
    )
    multi_parser.add_argument(
        "--raw-data",
        action="store_true",
        help="Use raw data from data/processed instead of filtered segments",
    )
    multi_parser.add_argument(
        "--output-dir",
        type=str,
        default="plots/profiles",
        help="Directory to save plots",
    )
    multi_parser.add_argument(
        "--summary-dir",
        type=str,
        default="data/system_id_summary",
        help="Directory containing classification CSV files",
    )
    multi_parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots instead of saving them",
    )

    # Add new stats command
    stats_parser = subparsers.add_parser('stats', help='Calculate elevation difference statistics')
    stats_parser.add_argument(
        'system_id',
        type=str,
        help='System ID to analyze'
    )
    stats_parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing processed data files"
    )

    args = parser.parse_args()

    if args.command == 'dashboard':
        create_dash_app(
            data_dir=args.data_dir,
            raw_data=args.raw,
            debug=args.debug,
            port=args.port,
        )

    elif args.command == 'plot':
        plot_elevation_profile(
            args.system_id,
            save_dir=args.save_dir,
            data_dir=args.data_dir,
            raw_data=args.raw,
            show=args.show,
        )

    elif args.command == 'diagnose':
        diagnose_elevation_differences(
            args.system_id,
            data_dir=args.data_dir,
            raw_data=args.raw,
        )

    elif args.command == 'multi':
        # Reconstruct argv for multi_profile_plot
        multi_argv = [sys.argv[0]]  # Keep the script name
        if args.type != 'all':
            multi_argv.extend(['--type', args.type])
        if args.data_dir != 'data/segments':
            multi_argv.extend(['--data-dir', args.data_dir])
        if args.raw_data:
            multi_argv.append('--raw-data')
        if args.output_dir != 'plots/profiles':
            multi_argv.extend(['--output-dir', args.output_dir])
        if args.summary_dir != 'data/system_id_summary':
            multi_argv.extend(['--summary-dir', args.summary_dir])
        if args.show:
            multi_argv.append('--show')

        # Replace sys.argv and run multi_profile_plot
        sys.argv = multi_argv
        plot_multi_profiles()

        # Restore original argv
        sys.argv = original_argv

    elif args.command == 'stats':
        calculate_system_difference(
            args.system_id,
            data_dir=args.data_dir
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
