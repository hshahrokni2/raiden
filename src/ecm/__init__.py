"""
ECM Module - Energy Conservation Measures with Swedish context.

Key features:
- Constraint-aware: No facade insulation on brick, etc.
- Swedish costs: Investment costs in SEK
- Swedish energy prices: District heating, electricity
- Sensible combinations: Prunes invalid/dominated options

Usage:
    from src.ecm import get_all_ecms, get_ecm, ECMCatalog

    # Quick access
    all_ecms = get_all_ecms()  # List of all 22+ ECMs
    ecm = get_ecm('wall_external_insulation')

    # Or use the catalog class
    catalog = ECMCatalog()
    envelope_ecms = catalog.by_category(ECMCategory.ENVELOPE)
"""

from .catalog import (
    ECMCatalog,
    ECM,
    ECMCategory,
    ECMParameter,
    ECMConstraint,
    SWEDISH_ECM_CATALOG,
    ECM_CATALOG,  # Alias
    get_all_ecms,
    get_ecm,
    get_ecms_by_category,
    list_ecm_ids,
)
from .constraints import ConstraintEngine, BuildingContext
from .combinations import CombinationGenerator
from .idf_modifier import IDFModifier
from .dependencies import (
    ECMDependencyMatrix,
    ECMRelation,
    RelationType,
    get_dependency_matrix,
    validate_package,
    get_package_synergy,
    suggest_additions,
    adjust_package_savings,
)

__all__ = [
    # ECM definitions
    'ECMCatalog',
    'ECM',
    'ECMCategory',
    'ECMParameter',
    'ECMConstraint',
    # Catalogs
    'SWEDISH_ECM_CATALOG',
    'ECM_CATALOG',
    # Convenience functions
    'get_all_ecms',
    'get_ecm',
    'get_ecms_by_category',
    'list_ecm_ids',
    # Constraint engine
    'ConstraintEngine',
    'BuildingContext',
    # Combinations
    'CombinationGenerator',
    # IDF modifier
    'IDFModifier',
    # Dependencies
    'ECMDependencyMatrix',
    'ECMRelation',
    'RelationType',
    'get_dependency_matrix',
    'validate_package',
    'get_package_synergy',
    'suggest_additions',
    'adjust_package_savings',
]
