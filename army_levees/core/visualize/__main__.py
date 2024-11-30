"""Main entry point for visualization module.

Usage:
    poetry run python -m army_levees.core.visualize [--systems N] [--threshold T] [--output-dir DIR]
"""

import argparse
import logging
from pathlib import Path

from .interactive import create_summary_map
from .summary import analyze_large_differences, investigate_problematic_systems
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

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Get total number of systems
    systems = get_processed_systems()
    logger.info(f"Found {len(systems)} processed levee systems")

    # Run analysis pipeline
    logger.info("\n1. Analyzing elevation differences...")
    large_diffs = analyze_large_differences(threshold=args.threshold)

    logger.info("\n2. Investigating problematic systems...")
    investigate_problematic_systems(n_systems=args.systems)

    logger.info("\n3. Creating interactive summary map...")
    summary_map = create_summary_map(
        save_path=args.output_dir / "levee_summary_map.html"
    )
    if summary_map:
        logger.info(f"Saved summary map to {args.output_dir}/levee_summary_map.html")

    logger.info("\nVisualization complete! Check the plots directory for results.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()
