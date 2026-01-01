#!/usr/bin/env python3
"""
Build archetype cache from trained surrogates.

This script converts the 40 trained GP surrogates into a pre-computed
cache that enables instant portfolio analysis.

Usage:
    python scripts/build_archetype_cache.py

    # Or with custom paths
    python scripts/build_archetype_cache.py \
        --surrogates ./surrogates_production \
        --output ./archetype_cache

Result:
    Creates archetype_cache/ directory with:
    - index.json - List of cached archetypes
    - {archetype_id}.json - Pre-computed results for each archetype

Performance:
    - Build time: ~30 seconds (one-time)
    - Portfolio lookup: ~30 seconds for 37,489 buildings
    - vs Full E+: 3.3 days for same portfolio
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.archetype_cache import (
    ArchetypeCacheBuilder,
    ArchetypeSimulationCache,
    get_portfolio_results_fast,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Build archetype cache from trained surrogates"
    )
    parser.add_argument(
        "--surrogates",
        type=Path,
        default=Path("./surrogates_production"),
        help="Directory with trained GP surrogates",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./archetype_cache"),
        help="Output directory for cache",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a test query after building",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("RAIDEN - Build Archetype Cache")
    print("=" * 60)
    print(f"Surrogates: {args.surrogates}")
    print(f"Output: {args.output}")
    print()

    # Check surrogates exist
    if not args.surrogates.exists():
        logger.error(f"Surrogates directory not found: {args.surrogates}")
        logger.info("Run: python scripts/train_all_surrogates.py --samples 150")
        return 1

    surrogate_index = args.surrogates / "index.json"
    if not surrogate_index.exists():
        logger.error(f"Surrogate index not found: {surrogate_index}")
        return 1

    # Build cache
    start = time.time()
    logger.info("Building archetype cache from surrogates...")

    builder = ArchetypeCacheBuilder(args.output)
    builder.build_from_surrogates(args.surrogates)

    build_time = time.time() - start
    print(f"\nCache built in {build_time:.1f}s")

    # Load and validate
    cache = ArchetypeSimulationCache.load(args.output)
    print(f"Loaded {len(cache.cache)} archetypes")

    # Test query
    if args.test:
        print("\nTest Query:")
        print("-" * 40)

        # Try a typical miljonprogrammet building
        try:
            result = cache.get_building_results(
                archetype_id="mfh_1961_1975",
                atemp_m2=2340,
                address="Aktergatan 11, Stockholm",
            )

            print(f"Address: {result.address}")
            print(f"Archetype: {result.archetype_id}")
            print(f"Atemp: {result.atemp_m2:,.0f} m²")
            print(f"Baseline: {result.baseline_kwh_m2:.1f} ± {result.baseline_uncertainty:.1f} kWh/m²")
            print("\nECM Packages:")
            for pkg_id, pkg in result.packages.items():
                if pkg.get("savings_percent", 0) > 0:
                    print(f"  {pkg_id}: {pkg['kwh_m2']:.1f} kWh/m² ({pkg['savings_percent']:.1f}% savings)")

        except Exception as e:
            logger.warning(f"Test query failed: {e}")

    # Show usage
    print("\n" + "=" * 60)
    print("USAGE")
    print("=" * 60)
    print("""
from src.simulation import ArchetypeSimulationCache, get_portfolio_results_fast

# Load cache
cache = ArchetypeSimulationCache.load("./archetype_cache")

# Single building
result = cache.get_building_results(
    archetype_id="mfh_1961_1975",
    atemp_m2=2340,
    address="Aktergatan 11",
)

# Entire portfolio (37,489 buildings in ~30 seconds)
buildings = [
    {"archetype_id": "mfh_1961_1975", "atemp_m2": 2340, "address": "Aktergatan 11"},
    {"archetype_id": "mfh_pre_1930", "atemp_m2": 1200, "address": "Bellmansgatan 16"},
    ...
]
results = get_portfolio_results_fast(buildings, cache)
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
