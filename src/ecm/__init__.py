"""
ECM Module - Energy Conservation Measures with Swedish context.

Key features:
- Constraint-aware: No facade insulation on brick, etc.
- Swedish costs: Investment costs in SEK
- Swedish energy prices: District heating, electricity
- Sensible combinations: Prunes invalid/dominated options
"""

from .catalog import ECMCatalog, ECM, ECMCategory
from .constraints import ConstraintEngine, BuildingContext
from .combinations import CombinationGenerator
from .idf_modifier import IDFModifier

__all__ = [
    'ECMCatalog', 'ECM', 'ECMCategory',
    'ConstraintEngine', 'BuildingContext',
    'CombinationGenerator',
    'IDFModifier'
]
