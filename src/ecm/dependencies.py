"""
ECM Dependency and Conflict Matrix.

Defines relationships between ECMs:
- Conflicts: ECMs that cannot be installed together
- Dependencies: ECMs that require other ECMs first
- Synergies: ECMs that work better together (combined savings > sum of parts)
- Anti-synergies: ECMs with diminishing returns when combined

This enables intelligent package generation and constraint-aware optimization.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RelationType(Enum):
    """Type of relationship between ECMs."""
    CONFLICT = "conflict"           # Cannot install both
    REQUIRES = "requires"           # Must install dependency first
    SYNERGY = "synergy"             # Combined savings > sum
    ANTI_SYNERGY = "anti_synergy"   # Diminishing returns when combined
    SUPERSEDES = "supersedes"       # One makes other unnecessary


@dataclass
class ECMRelation:
    """A relationship between two ECMs."""
    ecm_a: str
    ecm_b: str
    relation_type: RelationType
    factor: float = 1.0  # Synergy/anti-synergy multiplier (1.2 = 20% bonus)
    reason: str = ""     # Human-readable explanation


# =============================================================================
# CONFLICT DEFINITIONS
# =============================================================================

CONFLICTS: List[ECMRelation] = [
    # Wall insulation conflicts
    ECMRelation(
        "wall_external_insulation", "wall_internal_insulation",
        RelationType.CONFLICT,
        reason="Choose one insulation approach; both would be wasteful"
    ),
    ECMRelation(
        "wall_external_insulation", "facade_renovation",
        RelationType.CONFLICT,
        reason="Facade renovation includes external insulation"
    ),

    # Ventilation system conflicts
    ECMRelation(
        "ftx_installation", "ftx_upgrade",
        RelationType.CONFLICT,
        reason="New FTX makes upgrade unnecessary"
    ),
    ECMRelation(
        "ftx_installation", "apartment_ventilation_units",
        RelationType.CONFLICT,
        reason="Central vs decentralized ventilation are mutually exclusive"
    ),

    # Heat pump conflicts
    ECMRelation(
        "exhaust_air_heat_pump", "ground_source_heat_pump",
        RelationType.CONFLICT,
        reason="Two primary heating systems not economical"
    ),
    ECMRelation(
        "exhaust_air_heat_pump", "air_source_heat_pump",
        RelationType.CONFLICT,
        reason="Two primary heat pumps not needed"
    ),
    ECMRelation(
        "ground_source_heat_pump", "air_source_heat_pump",
        RelationType.CONFLICT,
        reason="Two primary heat pumps not needed"
    ),

    # Solar conflicts
    ECMRelation(
        "solar_pv", "solar_thermal",
        RelationType.CONFLICT,
        reason="Roof area usually insufficient for both; choose one"
    ),

    # Lighting conflicts
    ECMRelation(
        "led_lighting", "led_common_areas",
        RelationType.CONFLICT,
        reason="led_lighting is comprehensive, includes common areas"
    ),
]


# =============================================================================
# DEPENDENCY DEFINITIONS
# =============================================================================

DEPENDENCIES: List[ECMRelation] = [
    # Battery storage requires solar
    ECMRelation(
        "battery_storage", "solar_pv",
        RelationType.REQUIRES,
        reason="Battery storage only makes sense with solar PV"
    ),

    # NOTE: DCV does NOT strictly require FTX - works with F-system (exhaust only) too
    # Removed overly strict dependency. DCV can be installed in buildings with:
    # - FTX (balanced ventilation)
    # - F-system (exhaust only)
    # - Even natural ventilation with exhaust fans

    # Predictive control requires BMS
    ECMRelation(
        "predictive_control", "building_automation_system",
        RelationType.REQUIRES,
        reason="Predictive algorithms need BMS infrastructure"
    ),

    # Fault detection requires monitoring
    ECMRelation(
        "fault_detection", "energy_monitoring",
        RelationType.REQUIRES,
        reason="Fault detection needs energy data from monitoring"
    ),
]


# =============================================================================
# SYNERGY DEFINITIONS (Combined > Sum)
# =============================================================================

SYNERGIES: List[ECMRelation] = [
    # Envelope + HVAC synergies
    ECMRelation(
        "air_sealing", "ftx_installation",
        RelationType.SYNERGY,
        factor=1.15,  # 15% bonus
        reason="Tight envelope allows FTX to control all ventilation"
    ),
    ECMRelation(
        "wall_external_insulation", "ftx_installation",
        RelationType.SYNERGY,
        factor=1.10,
        reason="Better envelope allows lower ventilation rates"
    ),
    ECMRelation(
        "window_replacement", "air_sealing",
        RelationType.SYNERGY,
        factor=1.12,
        reason="New windows dramatically improve air tightness"
    ),

    # Heat pump synergies
    ECMRelation(
        "ground_source_heat_pump", "floor_heating_conversion",
        RelationType.SYNERGY,
        factor=1.20,
        reason="Low temp distribution improves GSHP COP significantly"
    ),
    ECMRelation(
        "exhaust_air_heat_pump", "ftx_installation",
        RelationType.SYNERGY,
        factor=1.18,
        reason="FTX pre-conditions exhaust air for FVP"
    ),

    # Solar + storage synergy
    ECMRelation(
        "solar_pv", "battery_storage",
        RelationType.SYNERGY,
        factor=1.25,
        reason="Battery enables self-consumption above 70%"
    ),

    # Controls synergies
    ECMRelation(
        "smart_thermostats", "individual_metering",
        RelationType.SYNERGY,
        factor=1.15,
        reason="Feedback increases occupant engagement"
    ),
    ECMRelation(
        "building_automation_system", "energy_monitoring",
        RelationType.SYNERGY,
        factor=1.20,
        reason="BAS enables automated optimization from monitoring data"
    ),
]


# =============================================================================
# ANTI-SYNERGY DEFINITIONS (Diminishing Returns)
# =============================================================================

ANTI_SYNERGIES: List[ECMRelation] = [
    # =========================================================================
    # ENVELOPE COMBINATIONS (diminishing returns on heat loss reduction)
    # =========================================================================
    ECMRelation(
        "wall_external_insulation", "roof_insulation",
        RelationType.ANTI_SYNERGY,
        factor=0.85,  # Only get 85% of sum of individual savings
        reason="After walls, less remaining heat loss through roof"
    ),
    ECMRelation(
        "window_replacement", "wall_external_insulation",
        RelationType.ANTI_SYNERGY,
        factor=0.88,
        reason="Thermal bridges at window frames already addressed"
    ),
    ECMRelation(
        "window_replacement", "roof_insulation",
        RelationType.ANTI_SYNERGY,
        factor=0.87,
        reason="After windows, less relative gain from roof insulation"
    ),
    ECMRelation(
        "wall_external_insulation", "basement_insulation",
        RelationType.ANTI_SYNERGY,
        factor=0.88,
        reason="After walls, basement losses are smaller fraction"
    ),
    ECMRelation(
        "wall_external_insulation", "air_sealing",
        RelationType.ANTI_SYNERGY,
        factor=0.90,
        reason="External insulation often improves airtightness"
    ),
    ECMRelation(
        "window_replacement", "air_sealing",
        RelationType.ANTI_SYNERGY,
        factor=0.85,
        reason="New windows greatly reduce infiltration at frames"
    ),

    # =========================================================================
    # VENTILATION COMBINATIONS (FTX + DCV compete for same savings)
    # =========================================================================
    ECMRelation(
        "ftx_installation", "demand_controlled_ventilation",
        RelationType.ANTI_SYNERGY,
        factor=0.70,  # CRITICAL: Only get 70% of sum!
        reason="FTX already recovers heat; DCV adds less when combined"
    ),
    ECMRelation(
        "ftx_upgrade", "demand_controlled_ventilation",
        RelationType.ANTI_SYNERGY,
        factor=0.75,
        reason="Better FTX already recovers more; DCV gains reduced"
    ),
    ECMRelation(
        "demand_controlled_ventilation", "occupancy_sensors",
        RelationType.ANTI_SYNERGY,
        factor=0.75,
        reason="DCV already responds to occupancy via CO2"
    ),
    ECMRelation(
        "ftx_installation", "exhaust_air_heat_pump",
        RelationType.ANTI_SYNERGY,
        factor=0.60,  # Strong anti-synergy: FVP needs exhaust heat
        reason="FTX recovers heat that FVP needs; reduced FVP benefit"
    ),

    # =========================================================================
    # HEAT PUMP + OTHER HEATING MEASURES
    # =========================================================================
    ECMRelation(
        "ground_source_heat_pump", "air_sealing",
        RelationType.ANTI_SYNERGY,
        factor=0.82,
        reason="GSHP COP benefits less from reduced load"
    ),
    ECMRelation(
        "heat_pump_integration", "solar_thermal",
        RelationType.ANTI_SYNERGY,
        factor=0.78,
        reason="Both target DHW; less combined benefit"
    ),
    ECMRelation(
        "heat_pump_water_heater", "solar_thermal",
        RelationType.ANTI_SYNERGY,
        factor=0.72,
        reason="Both target same DHW load"
    ),

    # =========================================================================
    # CONTROL/OPERATIONAL COMBINATIONS
    # =========================================================================
    ECMRelation(
        "smart_thermostats", "heating_curve_adjustment",
        RelationType.ANTI_SYNERGY,
        factor=0.80,
        reason="Both optimize same heating controls"
    ),
    ECMRelation(
        "radiator_balancing", "smart_thermostats",
        RelationType.ANTI_SYNERGY,
        factor=0.82,
        reason="Smart thermostats can compensate for imbalance"
    ),
    ECMRelation(
        "heating_curve_adjustment", "radiator_balancing",
        RelationType.ANTI_SYNERGY,
        factor=0.85,
        reason="Both improve distribution efficiency"
    ),
    ECMRelation(
        "effektvakt_optimization", "heating_curve_adjustment",
        RelationType.ANTI_SYNERGY,
        factor=0.88,
        reason="Both optimize heating system operation"
    ),
    ECMRelation(
        "bms_optimization", "heating_curve_adjustment",
        RelationType.ANTI_SYNERGY,
        factor=0.82,
        reason="BMS already optimizes heating curves"
    ),

    # =========================================================================
    # LIGHTING COMBINATIONS
    # =========================================================================
    ECMRelation(
        "led_lighting", "daylight_sensors",
        RelationType.ANTI_SYNERGY,
        factor=0.80,
        reason="LED already efficient; sensors add less marginal savings"
    ),
    ECMRelation(
        "led_lighting", "occupancy_sensors",
        RelationType.ANTI_SYNERGY,
        factor=0.78,
        reason="LED already efficient; sensors add less marginal savings"
    ),
]


# =============================================================================
# SUPERSEDES DEFINITIONS (One makes other unnecessary)
# =============================================================================

SUPERSEDES: List[ECMRelation] = [
    ECMRelation(
        "ftx_installation", "exhaust_air_heat_pump",
        RelationType.SUPERSEDES,
        reason="New FTX with high HR recovers heat without pump"
    ),
    ECMRelation(
        "ground_source_heat_pump", "heat_pump_integration",
        RelationType.SUPERSEDES,
        reason="GSHP is complete solution, not generic integration"
    ),
    ECMRelation(
        "building_automation_system", "smart_thermostats",
        RelationType.SUPERSEDES,
        reason="Full BAS includes thermostat control"
    ),
    ECMRelation(
        "facade_renovation", "thermal_bridge_remediation",
        RelationType.SUPERSEDES,
        reason="Full facade renovation addresses thermal bridges"
    ),
]


# =============================================================================
# DEPENDENCY MATRIX CLASS
# =============================================================================

class ECMDependencyMatrix:
    """
    Manages ECM relationships and validates combinations.

    Usage:
        matrix = ECMDependencyMatrix()

        # Check if combination is valid
        is_valid, issues = matrix.validate_combination(['solar_pv', 'battery_storage'])

        # Get synergy factor for a package
        factor = matrix.calculate_synergy_factor(['air_sealing', 'ftx_installation'])

        # Get all valid ECMs given current selection
        valid_ecms = matrix.get_valid_additions(['wall_external_insulation'])
    """

    def __init__(
        self,
        conflicts: List[ECMRelation] = None,
        dependencies: List[ECMRelation] = None,
        synergies: List[ECMRelation] = None,
        anti_synergies: List[ECMRelation] = None,
        supersedes: List[ECMRelation] = None,
    ):
        self.conflicts = conflicts or CONFLICTS
        self.dependencies = dependencies or DEPENDENCIES
        self.synergies = synergies or SYNERGIES
        self.anti_synergies = anti_synergies or ANTI_SYNERGIES
        self.supersedes = supersedes or SUPERSEDES

        # Build lookup indices
        self._conflict_pairs: Set[Tuple[str, str]] = set()
        self._requires: Dict[str, Set[str]] = {}  # ecm -> required ECMs
        self._synergy_factors: Dict[Tuple[str, str], float] = {}
        self._anti_synergy_factors: Dict[Tuple[str, str], float] = {}
        self._superseded_by: Dict[str, Set[str]] = {}  # ecm -> ECMs that supersede it

        self._build_indices()

    def _build_indices(self):
        """Build lookup structures for fast queries."""
        # Conflicts (bidirectional)
        for rel in self.conflicts:
            self._conflict_pairs.add((rel.ecm_a, rel.ecm_b))
            self._conflict_pairs.add((rel.ecm_b, rel.ecm_a))

        # Dependencies (directional: ecm_a requires ecm_b)
        for rel in self.dependencies:
            if rel.ecm_a not in self._requires:
                self._requires[rel.ecm_a] = set()
            self._requires[rel.ecm_a].add(rel.ecm_b)

        # Synergies (bidirectional lookup)
        for rel in self.synergies:
            self._synergy_factors[(rel.ecm_a, rel.ecm_b)] = rel.factor
            self._synergy_factors[(rel.ecm_b, rel.ecm_a)] = rel.factor

        # Anti-synergies (bidirectional)
        for rel in self.anti_synergies:
            self._anti_synergy_factors[(rel.ecm_a, rel.ecm_b)] = rel.factor
            self._anti_synergy_factors[(rel.ecm_b, rel.ecm_a)] = rel.factor

        # Supersedes (directional: ecm_a supersedes ecm_b)
        for rel in self.supersedes:
            if rel.ecm_b not in self._superseded_by:
                self._superseded_by[rel.ecm_b] = set()
            self._superseded_by[rel.ecm_b].add(rel.ecm_a)

    def validate_combination(
        self,
        ecm_ids: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Validate a combination of ECMs.

        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        ecm_set = set(ecm_ids)

        # Check conflicts
        for i, ecm_a in enumerate(ecm_ids):
            for ecm_b in ecm_ids[i+1:]:
                if (ecm_a, ecm_b) in self._conflict_pairs:
                    issues.append(f"Conflict: {ecm_a} and {ecm_b} cannot be combined")

        # Check missing dependencies
        for ecm in ecm_ids:
            required = self._requires.get(ecm, set())
            missing = required - ecm_set
            if missing:
                issues.append(
                    f"Dependency: {ecm} requires {', '.join(missing)} to be included"
                )

        # Check superseded measures (warning, not error)
        for ecm in ecm_ids:
            superseding = self._superseded_by.get(ecm, set()) & ecm_set
            if superseding:
                issues.append(
                    f"Redundant: {ecm} is superseded by {', '.join(superseding)}"
                )

        # Conflicts and missing deps are errors; superseded is warning
        has_errors = any(
            issue.startswith("Conflict:") or issue.startswith("Dependency:")
            for issue in issues
        )

        return not has_errors, issues

    def calculate_synergy_factor(self, ecm_ids: List[str]) -> float:
        """
        Calculate overall synergy/anti-synergy factor for a combination.

        Factor > 1.0: Package performs better than sum of parts
        Factor < 1.0: Diminishing returns
        Factor = 1.0: No interaction
        """
        if len(ecm_ids) < 2:
            return 1.0

        total_factor = 1.0

        # Check all pairs
        for i, ecm_a in enumerate(ecm_ids):
            for ecm_b in ecm_ids[i+1:]:
                # Check synergy
                syn_factor = self._synergy_factors.get((ecm_a, ecm_b))
                if syn_factor:
                    # Synergies add multiplicatively
                    total_factor *= syn_factor

                # Check anti-synergy
                anti_factor = self._anti_synergy_factors.get((ecm_a, ecm_b))
                if anti_factor:
                    # Anti-synergies reduce multiplicatively
                    total_factor *= anti_factor

        return total_factor

    def get_valid_additions(
        self,
        current_ecms: List[str],
        all_ecms: List[str]
    ) -> List[str]:
        """
        Get ECMs that can be added to current selection without conflicts.

        Args:
            current_ecms: Currently selected ECMs
            all_ecms: All available ECM IDs

        Returns:
            List of ECM IDs that can be added
        """
        current_set = set(current_ecms)
        valid = []

        for ecm in all_ecms:
            if ecm in current_set:
                continue

            # Check if adding this would create conflicts
            has_conflict = any(
                (ecm, curr) in self._conflict_pairs
                for curr in current_ecms
            )

            if not has_conflict:
                valid.append(ecm)

        return valid

    def get_required_ecms(self, ecm_id: str) -> Set[str]:
        """Get ECMs that must be included if this ECM is selected."""
        return self._requires.get(ecm_id, set()).copy()

    def get_conflicting_ecms(self, ecm_id: str) -> Set[str]:
        """Get ECMs that cannot be combined with this ECM."""
        return {
            ecm_b for (ecm_a, ecm_b) in self._conflict_pairs
            if ecm_a == ecm_id
        }

    def get_synergistic_ecms(self, ecm_id: str) -> List[Tuple[str, float]]:
        """Get ECMs that have synergy with this one."""
        synergies = []
        for (ecm_a, ecm_b), factor in self._synergy_factors.items():
            if ecm_a == ecm_id:
                synergies.append((ecm_b, factor))
        return synergies

    def suggest_complementary_ecms(
        self,
        ecm_id: str,
        available_ecms: List[str],
        max_suggestions: int = 5
    ) -> List[Tuple[str, float, str]]:
        """
        Suggest ECMs that complement the given ECM.

        Returns:
            List of (ecm_id, synergy_factor, reason)
        """
        suggestions = []

        # Find synergistic ECMs
        for rel in self.synergies:
            partner = None
            if rel.ecm_a == ecm_id and rel.ecm_b in available_ecms:
                partner = rel.ecm_b
            elif rel.ecm_b == ecm_id and rel.ecm_a in available_ecms:
                partner = rel.ecm_a

            if partner:
                suggestions.append((partner, rel.factor, rel.reason))

        # Add required ECMs with high priority
        for req in self.get_required_ecms(ecm_id):
            if req in available_ecms:
                suggestions.append((req, 2.0, "Required dependency"))

        # Sort by synergy factor descending
        suggestions.sort(key=lambda x: x[1], reverse=True)

        return suggestions[:max_suggestions]

    def to_dict(self) -> Dict:
        """Export relationships as dictionary for serialization."""
        return {
            "conflicts": [
                {"ecm_a": r.ecm_a, "ecm_b": r.ecm_b, "reason": r.reason}
                for r in self.conflicts
            ],
            "dependencies": [
                {"ecm": r.ecm_a, "requires": r.ecm_b, "reason": r.reason}
                for r in self.dependencies
            ],
            "synergies": [
                {"ecm_a": r.ecm_a, "ecm_b": r.ecm_b, "factor": r.factor, "reason": r.reason}
                for r in self.synergies
            ],
            "anti_synergies": [
                {"ecm_a": r.ecm_a, "ecm_b": r.ecm_b, "factor": r.factor, "reason": r.reason}
                for r in self.anti_synergies
            ],
            "supersedes": [
                {"superseding": r.ecm_a, "superseded": r.ecm_b, "reason": r.reason}
                for r in self.supersedes
            ],
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global instance
_default_matrix: Optional[ECMDependencyMatrix] = None


def get_dependency_matrix() -> ECMDependencyMatrix:
    """Get the default dependency matrix instance."""
    global _default_matrix
    if _default_matrix is None:
        _default_matrix = ECMDependencyMatrix()
    return _default_matrix


def validate_package(ecm_ids: List[str]) -> Tuple[bool, List[str]]:
    """Validate an ECM package using the default matrix."""
    return get_dependency_matrix().validate_combination(ecm_ids)


def get_package_synergy(ecm_ids: List[str]) -> float:
    """Get synergy factor for a package using the default matrix."""
    return get_dependency_matrix().calculate_synergy_factor(ecm_ids)


def suggest_additions(
    current_ecms: List[str],
    available_ecms: List[str]
) -> List[str]:
    """Get valid ECM additions using the default matrix."""
    return get_dependency_matrix().get_valid_additions(current_ecms, available_ecms)


def adjust_package_savings(
    baseline_kwh_m2: float,
    ecm_results: Dict[str, float],  # {ecm_id: simulated_kwh_m2}
) -> Dict:
    """
    Adjust package results for ECM interactions.

    When combining multiple ECMs, their savings don't simply add up.
    This function applies synergy/anti-synergy factors to get realistic
    combined package savings.

    Args:
        baseline_kwh_m2: Baseline heating consumption (kWh/m²)
        ecm_results: Dictionary of ECM ID to simulated kWh/m²

    Returns:
        Dict with adjusted savings and interaction info:
        - 'package_kwh_m2': Adjusted total consumption
        - 'total_savings_adjusted': Adjusted total savings percent
        - 'synergy_factor': Overall synergy/anti-synergy factor
        - 'warnings': List of package warnings
    """
    matrix = get_dependency_matrix()

    # Validate the package
    ecm_ids = list(ecm_results.keys())
    is_valid, issues = matrix.validate_combination(ecm_ids)

    # Calculate synergy factor
    synergy_factor = matrix.calculate_synergy_factor(ecm_ids)

    # Calculate individual savings
    individual_savings = {}
    for ecm_id, result in ecm_results.items():
        saving = (baseline_kwh_m2 - result) / baseline_kwh_m2 * 100
        individual_savings[ecm_id] = saving

    # Calculate naive total (simple sum)
    naive_total_savings = sum(individual_savings.values())

    # Apply synergy factor
    # Note: Factor > 1 means synergy (more savings), < 1 means anti-synergy
    adjusted_total_savings = naive_total_savings * synergy_factor

    # Cap at 100% savings
    adjusted_total_savings = min(adjusted_total_savings, 100.0)

    # Calculate adjusted kWh/m²
    package_kwh_m2 = baseline_kwh_m2 * (1 - adjusted_total_savings / 100)

    return {
        'baseline_kwh_m2': baseline_kwh_m2,
        'individual_savings': individual_savings,
        'naive_total_savings': naive_total_savings,
        'synergy_factor': synergy_factor,
        'total_savings_adjusted': adjusted_total_savings,
        'package_kwh_m2': package_kwh_m2,
        'is_valid': is_valid,
        'warnings': issues,
        'n_ecms': len(ecm_ids),
    }
