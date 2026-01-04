#!/usr/bin/env python3
"""
Pre-train surrogate library for all 40 Swedish archetypes.

This script generates training data by running EnergyPlus simulations
with Latin Hypercube sampling, then trains Gaussian Process surrogates.

Usage:
    # Train all archetypes (200 sims each = 8,000 total)
    python scripts/train_surrogate_library.py --all --samples 200

    # Train specific archetype
    python scripts/train_surrogate_library.py --archetype mfh_1961_1975 --samples 200

    # Resume interrupted training
    python scripts/train_surrogate_library.py --all --samples 200 --resume

    # Quick test with fewer samples
    python scripts/train_surrogate_library.py --archetype mfh_1996_2010 --samples 50

Estimated time:
    - 200 samples × 40 archetypes = 8,000 E+ simulations
    - ~30 seconds per simulation = ~67 hours sequential
    - With 8 parallel workers = ~8 hours
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import qmc

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for surrogate training."""
    archetype_id: str
    n_samples: int = 200
    n_workers: int = 4
    output_dir: Path = Path("./surrogates")
    weather_file: Optional[Path] = None
    param_bounds: Optional[Dict[str, Tuple[float, float]]] = None

    def __post_init__(self):
        if self.param_bounds is None:
            self.param_bounds = {
                'infiltration_ach': (0.02, 0.30),
                'wall_u_value': (0.10, 1.50),
                'roof_u_value': (0.10, 0.80),
                'window_u_value': (0.80, 3.00),
                'heat_recovery_eff': (0.00, 0.90),
                'heating_setpoint': (18.0, 23.0),
            }


@dataclass
class TrainingResult:
    """Result of surrogate training."""
    archetype_id: str
    train_r2: float
    test_r2: float
    is_overfit: bool
    n_samples: int
    training_time_seconds: float
    surrogate_path: Path
    param_bounds: Dict[str, Tuple[float, float]]


def generate_lhs_samples(
    n_samples: int,
    param_bounds: Dict[str, Tuple[float, float]],
) -> np.ndarray:
    """Generate Latin Hypercube samples for parameter space."""
    param_names = list(param_bounds.keys())
    n_params = len(param_names)

    # Latin Hypercube Sampling
    sampler = qmc.LatinHypercube(d=n_params)
    samples_unit = sampler.random(n=n_samples)

    # Scale to bounds
    lower = np.array([param_bounds[p][0] for p in param_names])
    upper = np.array([param_bounds[p][1] for p in param_names])

    return qmc.scale(samples_unit, lower, upper), param_names


def run_single_simulation(
    archetype_id: str,
    params: Dict[str, float],
    weather_path: Path,
    output_dir: Path,
    sample_idx: int,
) -> Optional[float]:
    """
    Run a single EnergyPlus simulation with given parameters.

    Returns heating_kwh_m2 or None if simulation fails.
    """
    try:
        from src.baseline import get_archetype
        from src.baseline.idf_generator import IDFGenerator
        from src.simulation.runner import EnergyPlusRunner

        # Get archetype
        archetype = get_archetype(archetype_id)
        if not archetype:
            logger.error(f"Archetype not found: {archetype_id}")
            return None

        # Generate IDF with params
        generator = IDFGenerator()

        # Apply params to archetype
        modified_archetype = archetype.copy()

        # Modify envelope U-values
        if modified_archetype.envelope:
            if 'wall_u_value' in params:
                modified_archetype.envelope.wall_u_value = params['wall_u_value']
            if 'roof_u_value' in params:
                modified_archetype.envelope.roof_u_value = params['roof_u_value']
            if 'window_u_value' in params:
                modified_archetype.envelope.window_u_value = params['window_u_value']
            if 'infiltration_ach' in params:
                modified_archetype.envelope.infiltration_ach = params['infiltration_ach']

        # Modify ventilation
        if modified_archetype.ventilation and 'heat_recovery_eff' in params:
            modified_archetype.ventilation.heat_recovery_efficiency = params['heat_recovery_eff']

        # Generate IDF
        sim_dir = output_dir / f"sim_{sample_idx:04d}"
        sim_dir.mkdir(parents=True, exist_ok=True)

        idf_path = sim_dir / "model.idf"
        generator.generate(
            archetype=modified_archetype,
            atemp_m2=1000,  # Standard area for normalization
            floors=4,
            output_path=idf_path,
        )

        # Run simulation
        runner = EnergyPlusRunner()
        result = runner.run(
            idf_path=idf_path,
            weather_path=weather_path,
            output_dir=sim_dir,
        )

        if result and result.heating_kwh:
            return result.heating_kwh / 1000  # Return kWh/m² (assuming 1000 m²)

        return None

    except Exception as e:
        logger.error(f"Simulation {sample_idx} failed: {e}")
        return None


def train_surrogate_for_archetype(
    config: TrainingConfig,
) -> Optional[TrainingResult]:
    """
    Train a Gaussian Process surrogate for an archetype.

    This runs LHS sampling, E+ simulations, and GP training.
    """
    start_time = time.time()
    logger.info(f"Training surrogate for {config.archetype_id} with {config.n_samples} samples")

    # Ensure output directory exists
    config.output_dir.mkdir(parents=True, exist_ok=True)
    sim_dir = config.output_dir / f"{config.archetype_id}_sims"
    sim_dir.mkdir(parents=True, exist_ok=True)

    # Generate LHS samples
    samples, param_names = generate_lhs_samples(config.n_samples, config.param_bounds)
    logger.info(f"Generated {len(samples)} LHS samples for {len(param_names)} parameters")

    # Check for cached results
    cache_path = sim_dir / "simulation_results.json"
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        logger.info(f"Found {len(cached)} cached simulation results")
    else:
        cached = {}

    # Run simulations
    results = []
    valid_samples = []

    # Use weather file
    if config.weather_file is None:
        # Default to Stockholm
        weather_path = Path("data/weather/SWE_ST_Stockholm_Arlanda_020440_TMYx.2004-2018.epw")
        if not weather_path.exists():
            logger.error(f"Weather file not found: {weather_path}")
            return None
    else:
        weather_path = config.weather_file

    # Run simulations in parallel
    with ProcessPoolExecutor(max_workers=config.n_workers) as executor:
        futures = {}

        for i, sample in enumerate(samples):
            # Check cache
            cache_key = f"{i:04d}"
            if cache_key in cached:
                results.append(cached[cache_key])
                valid_samples.append(sample)
                continue

            # Create params dict
            params = {name: sample[j] for j, name in enumerate(param_names)}

            # Submit simulation
            future = executor.submit(
                run_single_simulation,
                config.archetype_id,
                params,
                weather_path,
                sim_dir,
                i,
            )
            futures[future] = (i, sample)

        # Collect results
        for future in as_completed(futures):
            i, sample = futures[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                    valid_samples.append(sample)
                    cached[f"{i:04d}"] = result

                    # Save progress
                    with open(cache_path, 'w') as f:
                        json.dump(cached, f)

                    if len(results) % 10 == 0:
                        logger.info(f"Completed {len(results)}/{config.n_samples} simulations")
            except Exception as e:
                logger.error(f"Simulation {i} failed: {e}")

    logger.info(f"Completed {len(results)} valid simulations")

    if len(results) < 30:
        logger.error(f"Insufficient valid simulations: {len(results)} < 30")
        return None

    # Train GP surrogate
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
    from sklearn.model_selection import train_test_split

    X = np.array(valid_samples)
    y = np.array(results)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define kernel (Matern 5/2 for physical systems)
    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) *
        Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5) +
        WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
    )

    # Train GP
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=30,
        normalize_y=True,
        random_state=42,
    )

    gp.fit(X_train, y_train)

    # Evaluate
    train_r2 = gp.score(X_train, y_train)
    test_r2 = gp.score(X_test, y_test)
    is_overfit = (train_r2 - test_r2) > 0.10

    if is_overfit:
        logger.warning(
            f"⚠️ Surrogate overfitting detected: "
            f"train R²={train_r2:.3f} vs test R²={test_r2:.3f}"
        )
    else:
        logger.info(f"Surrogate trained: train R²={train_r2:.3f}, test R²={test_r2:.3f}")

    # Save surrogate
    surrogate_path = config.output_dir / f"{config.archetype_id}_gp.pkl"
    with open(surrogate_path, 'wb') as f:
        pickle.dump({
            'model': gp,
            'param_names': param_names,
            'param_bounds': config.param_bounds,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'n_samples': len(results),
            'archetype_id': config.archetype_id,
        }, f)

    training_time = time.time() - start_time
    logger.info(f"Training complete in {training_time/60:.1f} minutes")

    return TrainingResult(
        archetype_id=config.archetype_id,
        train_r2=train_r2,
        test_r2=test_r2,
        is_overfit=is_overfit,
        n_samples=len(results),
        training_time_seconds=training_time,
        surrogate_path=surrogate_path,
        param_bounds=config.param_bounds,
    )


def train_synthetic(
    config: TrainingConfig,
) -> Optional[TrainingResult]:
    """
    Train a surrogate using synthetic data (no E+ simulations).

    Uses archetype physics model to generate training data.
    This is faster but less accurate than E+ simulations.
    """
    start_time = time.time()
    logger.info(f"Training SYNTHETIC surrogate for {config.archetype_id}")

    from src.baseline import get_archetype

    archetype = get_archetype(config.archetype_id)
    if not archetype:
        logger.error(f"Archetype not found: {config.archetype_id}")
        return None

    # Generate LHS samples
    samples, param_names = generate_lhs_samples(config.n_samples, config.param_bounds)

    # Generate synthetic outputs using simplified physics
    results = []
    for sample in samples:
        params = {name: sample[j] for j, name in enumerate(param_names)}

        # Simplified energy model:
        # E = U_wall * A_wall * HDD + U_window * A_window * HDD
        #   - recovery_eff * ventilation_losses

        # Base energy from archetype
        base_energy = getattr(archetype, 'typical_heating_kwh_m2', 80.0) or 80.0

        # Get default U-values from archetype (DetailedArchetype structure)
        default_wall_u = 0.5
        default_roof_u = 0.3
        default_window_u = 2.0
        default_infil = 0.06

        if hasattr(archetype, 'wall_constructions') and archetype.wall_constructions:
            default_wall_u = archetype.wall_constructions[0].u_value
        if hasattr(archetype, 'roof_construction') and archetype.roof_construction:
            default_roof_u = archetype.roof_construction.u_value
        if hasattr(archetype, 'window_construction') and archetype.window_construction:
            default_window_u = getattr(archetype.window_construction, 'u_value_installed', 2.0) or 2.0
        if hasattr(archetype, 'infiltration_ach'):
            default_infil = archetype.infiltration_ach or 0.06

        # Adjust for U-values (relative to archetype defaults)
        u_wall_ratio = params.get('wall_u_value', default_wall_u) / default_wall_u
        u_roof_ratio = params.get('roof_u_value', default_roof_u) / default_roof_u
        u_window_ratio = params.get('window_u_value', default_window_u) / default_window_u
        infil_ratio = params.get('infiltration_ach', default_infil) / default_infil

        # Weight envelope components
        envelope_factor = 0.4 * u_wall_ratio + 0.2 * u_roof_ratio + 0.2 * u_window_ratio + 0.2 * infil_ratio

        # Adjust for heat recovery
        base_hr = getattr(archetype, 'heat_recovery_efficiency', None) or 0.0
        if base_hr > 0:
            new_hr = params.get('heat_recovery_eff', base_hr)
            # Heat recovery reduces ventilation losses (typically 20-40% of total)
            vent_factor = 1.0 - 0.3 * (new_hr - base_hr)
        else:
            vent_factor = 1.0

        # Calculate energy
        energy = base_energy * envelope_factor * vent_factor

        # Add some noise
        noise = np.random.normal(0, 2)
        energy = max(10, energy + noise)

        results.append(energy)

    # Train GP
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
    from sklearn.model_selection import train_test_split

    X = np.array(samples)
    y = np.array(results)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    kernel = (
        ConstantKernel(1.0) *
        Matern(length_scale=1.0, nu=2.5) +
        WhiteKernel(noise_level=1.0)
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=20,
        normalize_y=True,
        random_state=42,
    )

    gp.fit(X_train, y_train)

    train_r2 = gp.score(X_train, y_train)
    test_r2 = gp.score(X_test, y_test)
    is_overfit = (train_r2 - test_r2) > 0.10

    # Save
    config.output_dir.mkdir(parents=True, exist_ok=True)
    surrogate_path = config.output_dir / f"{config.archetype_id}_gp_synthetic.pkl"
    with open(surrogate_path, 'wb') as f:
        pickle.dump({
            'model': gp,
            'param_names': param_names,
            'param_bounds': config.param_bounds,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'n_samples': len(results),
            'archetype_id': config.archetype_id,
            'synthetic': True,
        }, f)

    training_time = time.time() - start_time
    logger.info(
        f"Synthetic training complete: "
        f"train R²={train_r2:.3f}, test R²={test_r2:.3f}, "
        f"time={training_time:.1f}s"
    )

    return TrainingResult(
        archetype_id=config.archetype_id,
        train_r2=train_r2,
        test_r2=test_r2,
        is_overfit=is_overfit,
        n_samples=len(results),
        training_time_seconds=training_time,
        surrogate_path=surrogate_path,
        param_bounds=config.param_bounds,
    )


def update_library_index(
    results: List[TrainingResult],
    output_dir: Path,
):
    """Update the surrogate library index file."""
    index_path = output_dir / "index.json"

    # Load existing index
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
    else:
        index = {}

    # Add new results
    for result in results:
        index[result.archetype_id] = {
            'filename': result.surrogate_path.name,
            'train_r2': result.train_r2,
            'test_r2': result.test_r2,
            'is_overfit': result.is_overfit,
            'n_samples': result.n_samples,
            'training_time_seconds': result.training_time_seconds,
            'trained_date': datetime.now().isoformat()[:10],
            'param_bounds': {k: list(v) for k, v in result.param_bounds.items()},
        }

    # Save index
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)

    logger.info(f"Updated index with {len(results)} surrogates")


def main():
    parser = argparse.ArgumentParser(
        description='Pre-train surrogate library for Swedish archetypes'
    )
    parser.add_argument(
        '--archetype', '-a',
        help='Specific archetype ID to train (e.g., mfh_1961_1975)'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Train all 40 archetypes'
    )
    parser.add_argument(
        '--samples', '-n', type=int, default=200,
        help='Number of LHS samples per archetype (default: 200)'
    )
    parser.add_argument(
        '--workers', '-w', type=int, default=4,
        help='Number of parallel E+ workers (default: 4)'
    )
    parser.add_argument(
        '--output', '-o', default='./surrogates',
        help='Output directory for surrogates (default: ./surrogates)'
    )
    parser.add_argument(
        '--weather', help='Path to weather file (default: Stockholm)'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume from cached simulations'
    )
    parser.add_argument(
        '--synthetic', action='store_true',
        help='Use synthetic data (faster, less accurate)'
    )
    parser.add_argument(
        '--list', action='store_true',
        help='List available archetypes'
    )

    args = parser.parse_args()

    if args.list:
        from src.baseline import get_all_archetypes
        print("\nAvailable archetypes:")
        archetypes = get_all_archetypes()
        # Handle both dict and list return types
        if isinstance(archetypes, dict):
            for arch_id, arch in archetypes.items():
                name = getattr(arch, 'name_en', arch_id)
                print(f"  {arch_id}: {name}")
        else:
            for arch in archetypes:
                print(f"  {arch.id}: {arch.name_en}")
        return 0

    if not args.all and not args.archetype:
        parser.print_help()
        return 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get archetypes to train
    if args.all:
        from src.baseline import get_all_archetypes
        all_archs = get_all_archetypes()
        # Handle both dict and list return types
        if isinstance(all_archs, dict):
            archetypes = list(all_archs.keys())
        else:
            archetypes = [a.id for a in all_archs]
    else:
        archetypes = [args.archetype]

    logger.info(f"Training surrogates for {len(archetypes)} archetypes")
    logger.info(f"Samples per archetype: {args.samples}")
    logger.info(f"Parallel workers: {args.workers}")
    logger.info(f"Output directory: {output_dir}")

    # Train each archetype
    results = []
    for i, archetype_id in enumerate(archetypes):
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i+1}/{len(archetypes)}] Training {archetype_id}")
        logger.info('='*60)

        config = TrainingConfig(
            archetype_id=archetype_id,
            n_samples=args.samples,
            n_workers=args.workers,
            output_dir=output_dir,
            weather_file=Path(args.weather) if args.weather else None,
        )

        try:
            if args.synthetic:
                result = train_synthetic(config)
            else:
                result = train_surrogate_for_archetype(config)

            if result:
                results.append(result)

        except Exception as e:
            logger.error(f"Failed to train {archetype_id}: {e}")
            import traceback
            traceback.print_exc()

    # Update library index
    if results:
        update_library_index(results, output_dir)

    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total archetypes: {len(archetypes)}")
    print(f"Successfully trained: {len(results)}")
    print(f"Failed: {len(archetypes) - len(results)}")

    if results:
        avg_train_r2 = np.mean([r.train_r2 for r in results])
        avg_test_r2 = np.mean([r.test_r2 for r in results])
        total_time = sum(r.training_time_seconds for r in results)

        print(f"\nAverage train R²: {avg_train_r2:.3f}")
        print(f"Average test R²: {avg_test_r2:.3f}")
        print(f"Total training time: {total_time/3600:.1f} hours")

        # List any overfit models
        overfit = [r for r in results if r.is_overfit]
        if overfit:
            print(f"\n⚠️ Overfit models ({len(overfit)}):")
            for r in overfit:
                print(f"  {r.archetype_id}: train={r.train_r2:.3f}, test={r.test_r2:.3f}")

    print(f"\nSurrogates saved to: {output_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
