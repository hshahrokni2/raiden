"""
Simulation Module - Run EnergyPlus simulations.

Features:
- Parallel execution of ECM scenarios
- Results parsing and aggregation
- Error handling for E+ crashes
"""

from .runner import SimulationRunner, SimulationResult
from .results import ResultsParser, AnnualResults

__all__ = ['SimulationRunner', 'SimulationResult', 'ResultsParser', 'AnnualResults']
