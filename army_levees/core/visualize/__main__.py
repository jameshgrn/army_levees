"""Main entry point for visualization module.

Usage:
    poetry run python -m army_levees.core.visualize [--systems N] [--threshold T] [--output-dir DIR]
"""

import argparse
import logging
from pathlib import Path

from .interactive import create_interactive_dashboard
from .utils import get_processed_systems

logger = logging.getLogger(__name__)


def main():
    """Run main visualization analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze and visualize levee elevation data"
    )
    parser.add_argument(
        "--systems",
        type=int,
        default=5,
        help="Number of problematic systems to investigate",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Elevation difference threshold (meters)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Output directory for plots",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/segments"),
        help="Directory containing levee data (default: data/segments)",
    )
    parser.add_argument(
        "--raw-data",
        action="store_true",
        help="Use raw data from data/processed instead of filtered data",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the interactive dashboard in browser",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Use raw or filtered data
    data_dir = Path("data/processed") if args.raw_data else args.data_dir

    # Get total number of systems
    systems = get_processed_systems()
    logger.info(f"Found {len(systems)} processed levee systems")

    # Create interactive dashboard
    logger.info("\nCreating interactive dashboard...")
    dashboard = create_interactive_dashboard(
        save_path=args.output_dir / "levee_dashboard.html",
        data_dir=data_dir,
        raw_data=args.raw_data,
        show=args.show,
    )
    if dashboard:
        logger.info(
            f"Saved interactive dashboard to {args.output_dir}/levee_dashboard.html"
        )
        if args.show:
            logger.info("Opening dashboard in browser...")

    logger.info("\nVisualization complete! Check the plots directory for results.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()
