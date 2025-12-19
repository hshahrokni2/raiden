"""
Simulation Module - Run EnergyPlus simulations.

Features:
- Parallel execution of ECM scenarios
- Results parsing and aggregation
- Error handling for E+ crashes
"""

from .runner import SimulationRunner, SimulationResult, run_simulation
from .results import ResultsParser, AnnualResults, parse_results

__all__ = [
    'SimulationRunner',
    'SimulationResult',
    'run_simulation',
    'ResultsParser',
    'AnnualResults',
    'parse_results',
]
