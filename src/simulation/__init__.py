"""
Simulation Module - Run EnergyPlus simulations.

Features:
- Parallel execution of ECM scenarios
- Results parsing and aggregation
- Error handling for E+ crashes
- Archetype-based caching for fast portfolio analysis
"""

from .runner import SimulationRunner, SimulationResult, run_simulation
from .results import ResultsParser, AnnualResults, parse_results
from .archetype_cache import (
    ArchetypeSimulationCache,
    ArchetypeCacheBuilder,
    InterpolatedResult,
    build_cache_from_surrogates,
    get_portfolio_results_fast,
)
from .distributed_worker import (
    EPlusWorker,
    DistributedCoordinator,
    SimulationTask,
)

__all__ = [
    # Core simulation
    'SimulationRunner',
    'SimulationResult',
    'run_simulation',
    'ResultsParser',
    'AnnualResults',
    'parse_results',
    # Archetype cache (fast portfolio)
    'ArchetypeSimulationCache',
    'ArchetypeCacheBuilder',
    'InterpolatedResult',
    'build_cache_from_surrogates',
    'get_portfolio_results_fast',
    # Distributed simulation
    'EPlusWorker',
    'DistributedCoordinator',
    'SimulationTask',
]
