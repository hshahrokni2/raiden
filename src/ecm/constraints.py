"""
Constraint Engine - Determine valid ECMs for a building.

Evaluates ECM constraints against building context to determine
which ECMs are applicable and which are excluded.

This is the "smart" part - no facade insulation on brick, etc.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

from .catalog import ECM, ECMConstraint, ECMCatalog


@dataclass
class BuildingContext:
    """
    Building context for constraint evaluation.

    Populated from:
    - Energy declaration
    - Mapillary analysis (facade material)
    - Archetype matching
    - Geometry calculations
    """
    # Required fields (no defaults)
    construction_year: int
    building_type: str  # 'multi_family', 'single_family', etc.
    facade_material: str  # 'brick', 'concrete', 'render', 'wood', 'glass'
    heating_system: str  # 'district', 'electric', 'heat_pump_ground', etc.
    ventilation_type: str  # 'natural', 'f', 'ftx'

    # Optional fields with defaults
    heritage_listed: bool = False
    current_window_u: float = 2.0  # W/m²K
    current_infiltration_ach: float = 0.10
    current_heat_recovery: float = 0.0  # 0 for no HR
    has_hydronic_distribution: bool = True

    # Geometry (from calculations)
    floor_area_m2: float = 0.0
    wall_area_m2: float = 0.0
    window_area_m2: float = 0.0
    roof_area_m2: float = 0.0
    available_pv_area_m2: float = 0.0
    roof_type: str = "flat"  # 'flat', 'pitched', 'pitched_south'
    shading_factor: float = 0.0  # 0 = no shading, 1 = fully shaded

    # Current performance
    current_lighting_w_m2: float = 8.0

    # Financial (optional - for later filtering)
    max_investment_sek: Optional[float] = None
    min_payback_years: Optional[float] = None


@dataclass
class ConstraintResult:
    """Result of constraint evaluation."""
    ecm_id: str
    is_valid: bool
    failed_constraints: List[Tuple[str, str]]  # (field, reason)


class ConstraintEngine:
    """
    Evaluate ECM constraints against building context.

    Usage:
        engine = ConstraintEngine()
        context = BuildingContext(
            facade_material='brick',
            construction_year=1968,
            ...
        )
        valid_ecms = engine.get_valid_ecms(context)
    """

    def __init__(self, catalog: ECMCatalog = None):
        self.catalog = catalog or ECMCatalog()

    def evaluate_constraint(
        self,
        constraint: ECMConstraint,
        context: BuildingContext
    ) -> Tuple[bool, Optional[str]]:
        """
        Evaluate a single constraint.

        Returns:
            (passed, failure_reason)
        """
        # Get value from context
        if not hasattr(context, constraint.field):
            return True, None  # Unknown field, skip constraint

        value = getattr(context, constraint.field)

        # Evaluate based on operator
        passed = False
        if constraint.operator == 'eq':
            passed = value == constraint.value
        elif constraint.operator == 'ne':
            passed = value != constraint.value
        elif constraint.operator == 'in':
            passed = value in constraint.value
        elif constraint.operator == 'not_in':
            passed = value not in constraint.value
        elif constraint.operator == 'gt':
            passed = value > constraint.value
        elif constraint.operator == 'lt':
            passed = value < constraint.value
        elif constraint.operator == 'gte':
            passed = value >= constraint.value
        elif constraint.operator == 'lte':
            passed = value <= constraint.value
        else:
            # Unknown operator, skip
            passed = True

        return passed, None if passed else constraint.reason

    def evaluate_ecm(self, ecm: ECM, context: BuildingContext) -> ConstraintResult:
        """
        Evaluate all constraints for an ECM.

        Returns:
            ConstraintResult indicating if ECM is valid
        """
        failed = []

        for constraint in ecm.constraints:
            passed, reason = self.evaluate_constraint(constraint, context)
            if not passed:
                failed.append((constraint.field, reason))

        return ConstraintResult(
            ecm_id=ecm.id,
            is_valid=len(failed) == 0,
            failed_constraints=failed
        )

    def get_valid_ecms(self, context: BuildingContext) -> List[ECM]:
        """
        Get all valid ECMs for a building context.

        Returns:
            List of ECMs that pass all constraints
        """
        valid = []
        for ecm in self.catalog.all():
            result = self.evaluate_ecm(ecm, context)
            if result.is_valid:
                valid.append(ecm)
        return valid

    def get_excluded_ecms(
        self,
        context: BuildingContext
    ) -> List[Tuple[ECM, List[Tuple[str, str]]]]:
        """
        Get excluded ECMs with reasons.

        Returns:
            List of (ECM, [(field, reason), ...]) for excluded ECMs
        """
        excluded = []
        for ecm in self.catalog.all():
            result = self.evaluate_ecm(ecm, context)
            if not result.is_valid:
                excluded.append((ecm, result.failed_constraints))
        return excluded

    def explain_constraints(self, context: BuildingContext) -> str:
        """
        Generate human-readable explanation of valid/excluded ECMs.

        Returns:
            Formatted string explaining ECM applicability
        """
        lines = []
        lines.append(f"ECM Applicability for {context.building_type} ({context.construction_year})")
        lines.append(f"Facade: {context.facade_material}, Heating: {context.heating_system}")
        lines.append("")

        lines.append("VALID ECMs:")
        for ecm in self.get_valid_ecms(context):
            lines.append(f"  ✓ {ecm.name}")

        lines.append("")
        lines.append("EXCLUDED ECMs:")
        for ecm, reasons in self.get_excluded_ecms(context):
            lines.append(f"  ✗ {ecm.name}")
            for field, reason in reasons:
                lines.append(f"      - {reason}")

        return "\n".join(lines)
