"""
Baseline Module - Auto-generate calibrated building energy models.

Takes building data from public sources and creates:
- Archetype-based initial model
- Auto-calibrated to energy declaration
- Ready for ECM analysis

Key principle: Maximum accuracy from minimum owner input.
"""

from .archetypes import SwedishArchetype, ArchetypeMatcher
from .generator import BaselineGenerator
from .calibrator import BaselineCalibrator

__all__ = ['SwedishArchetype', 'ArchetypeMatcher', 'BaselineGenerator', 'BaselineCalibrator']
