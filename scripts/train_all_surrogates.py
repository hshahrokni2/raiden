#!/usr/bin/env python3
"""
Train GP surrogates for all Swedish archetypes.

This script trains Gaussian Process surrogate models for all 40 archetypes,
enabling fast ECM predictions without running EnergyPlus for each building.

Usage:
    python scripts/train_all_surrogates.py --samples 150 --workers 8

Requirements:
    - EnergyPlus installed and configured
    - Weather files in WEATHER_FILE_DIR
    - ~6000 E+ simulations (40 archetypes × 150 samples)

Output:
    - surrogates/{archetype_id}_gp.pkl for each archetype
    - surrogates/index.json with metadata
"""

import argparse
import json
import logging
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baseline import get_all_archetypes, get_archetype
from src.calibration import SurrogateConfig, SurrogateTrainer, SurrogatePredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Result of training a single archetype surrogate."""
    archetype_id: str
    success: bool
    train_r2: float = 0.0
    test_r2: float = 0.0
    n_samples: int = 0
    training_time_sec: float = 0.0
    error: Optional[str] = None


# Default parameter bounds for Swedish buildings
DEFAULT_PARAM_BOUNDS = {
    "infiltration_ach": (0.02, 0.30),
    "wall_u_value": (0.10, 1.50),
    "roof_u_value": (0.10, 0.80),
    "window_u_value": (0.80, 3.00),
    "heat_recovery_eff": (0.00, 0.90),
    "heating_setpoint": (18.0, 23.0),
}


def get_archetype_specific_bounds(archetype_id: str) -> Dict[str, Tuple[float, float]]:
    """Get parameter bounds adjusted for specific archetype."""
    bounds = DEFAULT_PARAM_BOUNDS.copy()

    archetype = get_archetype(archetype_id)
    if not archetype:
        return bounds

    # Handle both SwedishArchetype (envelope) and DetailedArchetype (wall_constructions)
    base_u = 0.5
    roof_u = 0.3
    win_u = 2.0
    has_heat_recovery = False
    heat_recovery_eff = 0.0

    if hasattr(archetype, 'envelope') and archetype.envelope:
        # Old SwedishArchetype structure
        base_u = archetype.envelope.wall_u_value
        roof_u = archetype.envelope.roof_u_value or 0.3
        win_u = archetype.envelope.window_u_value or 2.0
        if hasattr(archetype, 'ventilation') and archetype.ventilation:
            has_heat_recovery = archetype.ventilation.has_heat_recovery
            heat_recovery_eff = archetype.ventilation.heat_recovery_efficiency or 0.0
    elif hasattr(archetype, 'wall_constructions') and archetype.wall_constructions:
        # DetailedArchetype structure
        base_u = archetype.wall_constructions[0].u_value
        if archetype.roof_construction:
            roof_u = archetype.roof_construction.u_value
        if archetype.window_construction:
            # WindowConstruction uses u_value_installed
            win_u = getattr(archetype.window_construction, 'u_value_installed',
                           getattr(archetype.window_construction, 'u_value_glass', 2.0))
        # Check for FTX
        if hasattr(archetype, 'heat_recovery_efficiency') and archetype.heat_recovery_efficiency:
            has_heat_recovery = archetype.heat_recovery_efficiency > 0
            heat_recovery_eff = archetype.heat_recovery_efficiency

    # Adjust wall U-value bounds (ensure lower < upper)
    wall_lower = max(0.10, base_u * 0.5)
    wall_upper = min(1.50, base_u * 2.0)
    if wall_lower >= wall_upper:
        wall_lower, wall_upper = 0.10, 1.50  # Reset to defaults
    bounds["wall_u_value"] = (wall_lower, wall_upper)

    # Roof (ensure lower < upper)
    roof_lower = max(0.10, roof_u * 0.5)
    roof_upper = min(0.80, roof_u * 2.0)
    if roof_lower >= roof_upper:
        roof_lower, roof_upper = 0.10, 0.80
    bounds["roof_u_value"] = (roof_lower, roof_upper)

    # Window (ensure lower < upper)
    win_lower = max(0.80, win_u * 0.7)
    win_upper = min(3.00, win_u * 1.5)
    if win_lower >= win_upper:
        win_lower, win_upper = 0.80, 3.00
    bounds["window_u_value"] = (win_lower, win_upper)

    # Adjust heat recovery for archetypes with FTX
    if has_heat_recovery:
        bounds["heat_recovery_eff"] = (
            max(0.50, heat_recovery_eff - 0.20),
            min(0.95, heat_recovery_eff + 0.10),
        )
    else:
        # No FTX - lower range
        bounds["heat_recovery_eff"] = (0.0, 0.50)

    return bounds


def train_single_archetype(
    archetype_id: str,
    n_samples: int,
    output_dir: Path,
    use_simulation: bool = True,
    base_idf_path: Optional[Path] = None,
    weather_path: Optional[Path] = None,
    n_workers: int = 4,
) -> TrainingResult:
    """
    Train surrogate for a single archetype.

    Args:
        archetype_id: Archetype identifier
        n_samples: Number of LHS samples
        output_dir: Directory for output files
        use_simulation: If True, run actual E+ simulations
        base_idf_path: Path to base IDF template (required for simulation)
        weather_path: Path to weather file (required for simulation)
        n_workers: Number of parallel simulation workers

    Returns:
        TrainingResult with metrics
    """
    start_time = time.time()

    try:
        logger.info(f"Training surrogate for {archetype_id} with {n_samples} samples")

        # Get archetype-specific bounds
        bounds = get_archetype_specific_bounds(archetype_id)

        # Configure trainer
        config = SurrogateConfig(
            n_samples=n_samples,
            param_bounds=bounds,
        )

        trainer = SurrogateTrainer(config=config)

        if use_simulation:
            # Train with actual E+ simulations
            if not base_idf_path or not weather_path:
                raise ValueError(
                    "Production training requires --base-idf and --weather arguments"
                )

            sim_output_dir = output_dir / f"{archetype_id}_sims"
            trained = trainer.train_with_simulation(
                archetype_id=archetype_id,
                base_idf_path=base_idf_path,
                weather_path=weather_path,
                output_dir=sim_output_dir,
                n_workers=n_workers,
            )
        else:
            # Mock training for testing (uses archetype-based model)
            trained = trainer.train_mock(archetype_id=archetype_id)

        # Save surrogate
        output_path = output_dir / f"{archetype_id}_gp.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(trained.model, f)

        training_time = time.time() - start_time

        logger.info(
            f"Trained {archetype_id}: train_r2={trained.train_r2:.3f}, "
            f"test_r2={trained.test_r2:.3f}, time={training_time:.1f}s"
        )

        return TrainingResult(
            archetype_id=archetype_id,
            success=True,
            train_r2=trained.train_r2,
            test_r2=trained.test_r2,
            n_samples=n_samples,
            training_time_sec=training_time,
        )

    except Exception as e:
        logger.error(f"Failed to train {archetype_id}: {e}")
        return TrainingResult(
            archetype_id=archetype_id,
            success=False,
            error=str(e),
            training_time_sec=time.time() - start_time,
        )


def train_all_surrogates(
    n_samples: int = 150,
    workers: int = 4,
    output_dir: Optional[Path] = None,
    archetypes: Optional[List[str]] = None,
    use_simulation: bool = True,
    base_idf_path: Optional[Path] = None,
    weather_path: Optional[Path] = None,
    sim_workers: int = 4,
) -> Dict[str, TrainingResult]:
    """
    Train surrogates for all archetypes in parallel.

    Args:
        n_samples: Samples per archetype
        workers: Number of parallel archetype workers
        output_dir: Output directory for surrogates
        archetypes: Specific archetypes to train (None = all)
        use_simulation: If True, run actual E+ simulations
        base_idf_path: Path to base IDF template (for production training)
        weather_path: Path to weather file (for production training)
        sim_workers: Number of parallel E+ simulation workers per archetype

    Returns:
        Dict of archetype_id → TrainingResult
    """
    output_dir = output_dir or Path("./surrogates")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get archetypes to train
    if archetypes:
        archetype_ids = archetypes
    else:
        all_archetypes = get_all_archetypes()
        archetype_ids = list(all_archetypes.keys())

    logger.info(f"Training {len(archetype_ids)} archetypes with {n_samples} samples each")
    logger.info(f"Total simulations: {len(archetype_ids) * n_samples}")
    logger.info(f"Workers: {workers}")

    if use_simulation:
        logger.info(f"Mode: PRODUCTION (E+ simulations)")
        logger.info(f"Base IDF: {base_idf_path}")
        logger.info(f"Weather: {weather_path}")
        logger.info(f"Sim workers per archetype: {sim_workers}")
    else:
        logger.info(f"Mode: MOCK (synthetic data)")

    results: Dict[str, TrainingResult] = {}

    # For production training, we run archetypes sequentially to avoid
    # overwhelming the system with too many E+ processes
    if use_simulation:
        # Sequential archetype training, parallel simulations within each
        for arch_id in archetype_ids:
            try:
                result = train_single_archetype(
                    arch_id,
                    n_samples,
                    output_dir,
                    use_simulation=True,
                    base_idf_path=base_idf_path,
                    weather_path=weather_path,
                    n_workers=sim_workers,
                )
                results[arch_id] = result
            except Exception as e:
                logger.error(f"Exception training {arch_id}: {e}")
                results[arch_id] = TrainingResult(
                    archetype_id=arch_id,
                    success=False,
                    error=str(e),
                )
    else:
        # Mock training can run archetypes in parallel
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    train_single_archetype,
                    arch_id,
                    n_samples,
                    output_dir,
                    use_simulation,
                ): arch_id
                for arch_id in archetype_ids
            }

            for future in as_completed(futures):
                arch_id = futures[future]
                try:
                    result = future.result()
                    results[arch_id] = result
                except Exception as e:
                    logger.error(f"Exception training {arch_id}: {e}")
                    results[arch_id] = TrainingResult(
                        archetype_id=arch_id,
                        success=False,
                        error=str(e),
                    )

    # Save index
    index = {}
    for arch_id, result in results.items():
        if result.success:
            index[arch_id] = {
                "filename": f"{arch_id}_gp.pkl",
                "train_r2": result.train_r2,
                "test_r2": result.test_r2,
                "n_samples": result.n_samples,
                "trained_date": datetime.now().isoformat()[:10],
                "param_bounds": get_archetype_specific_bounds(arch_id),
            }

    index_path = output_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    logger.info(f"Index saved to {index_path}")

    # Summary
    successful = sum(1 for r in results.values() if r.success)
    failed = len(results) - successful
    avg_train_r2 = np.mean([r.train_r2 for r in results.values() if r.success])
    avg_test_r2 = np.mean([r.test_r2 for r in results.values() if r.success])

    logger.info("=" * 60)
    logger.info(f"TRAINING COMPLETE")
    logger.info(f"  Successful: {successful}/{len(results)}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Avg train R²: {avg_train_r2:.3f}")
    logger.info(f"  Avg test R²: {avg_test_r2:.3f}")
    logger.info("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train GP surrogates for all Swedish archetypes"
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=150,
        help="Number of LHS samples per archetype (default: 150)",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("./surrogates"),
        help="Output directory for surrogates",
    )
    parser.add_argument(
        "--archetypes", "-a",
        nargs="+",
        help="Specific archetypes to train (default: all)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock training (no E+ simulations) for testing",
    )
    parser.add_argument(
        "--base-idf",
        type=Path,
        default=Path("examples/sjostaden_2/energyplus/sjostaden_7zone.idf"),
        help="Base IDF template for production training",
    )
    parser.add_argument(
        "--weather",
        type=Path,
        default=Path("tests/fixtures/stockholm.epw"),
        help="Weather file for production training",
    )
    parser.add_argument(
        "--sim-workers",
        type=int,
        default=4,
        help="Number of parallel E+ workers per archetype (default: 4)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available archetypes and exit",
    )

    args = parser.parse_args()

    if args.list:
        archetypes = get_all_archetypes()
        print(f"\nAvailable archetypes ({len(archetypes)}):\n")
        for arch_id in sorted(archetypes.keys()):
            print(f"  {arch_id}")
        return

    # Validate paths for production training
    if not args.mock:
        if not args.base_idf.exists():
            print(f"Error: Base IDF not found: {args.base_idf}")
            print("Use --mock for synthetic training, or provide --base-idf")
            return
        if not args.weather.exists():
            print(f"Error: Weather file not found: {args.weather}")
            print("Use --mock for synthetic training, or provide --weather")
            return
        print(f"\n🚀 PRODUCTION TRAINING (E+ simulations)")
        print(f"   Base IDF: {args.base_idf}")
        print(f"   Weather: {args.weather}")
        print(f"   Samples per archetype: {args.samples}")
        print(f"   E+ workers: {args.sim_workers}")
        print(f"   Total simulations: ~{args.samples * 40} (40 archetypes)\n")
    else:
        print(f"\n📋 MOCK TRAINING (synthetic data)")
        print(f"   Samples per archetype: {args.samples}\n")

    results = train_all_surrogates(
        n_samples=args.samples,
        workers=args.workers,
        output_dir=args.output,
        archetypes=args.archetypes,
        use_simulation=not args.mock,
        base_idf_path=args.base_idf if not args.mock else None,
        weather_path=args.weather if not args.mock else None,
        sim_workers=args.sim_workers,
    )

    # Print failures
    failures = [r for r in results.values() if not r.success]
    if failures:
        print("\nFailed archetypes:")
        for r in failures:
            print(f"  {r.archetype_id}: {r.error}")


if __name__ == "__main__":
    main()
