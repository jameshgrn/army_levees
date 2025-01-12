"""Main entry point for visualization module."""

import argparse
from pathlib import Path

from .dash_app import create_dash_app

def main():
    """Run the visualization dashboard."""
    parser = argparse.ArgumentParser(description="Interactive levee visualization dashboard")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/segments",
        help="Directory containing levee segment data (default: data/segments)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port to run the dashboard on (default: 8050)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Use raw data instead of processed data",
    )

    args = parser.parse_args()

    # Create and run dashboard
    create_dash_app(
        data_dir=args.data_dir,
        raw_data=args.raw,
        debug=args.debug,
        port=args.port,
    )

if __name__ == "__main__":
    main()
