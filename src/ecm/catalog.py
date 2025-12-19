"""
ECM Catalog - Swedish Energy Conservation Measures.

Defines all available ECMs with:
- Technical parameters
- Applicable building types
- Constraints (what makes this ECM invalid)
- Cost estimates (SEK)
- Expected savings ranges
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class ECMCategory(Enum):
    """ECM categories."""
    ENVELOPE = "envelope"
    HVAC = "hvac"
    RENEWABLE = "renewable"
    CONTROLS = "controls"
    LIGHTING = "lighting"
    OPERATIONAL = "operational"  # Zero-cost operational optimization


@dataclass
class ECMParameter:
    """A parameter that can be varied for an ECM."""
    name: str
    values: List[Any]  # Possible values
    unit: str
    description: str


@dataclass
class ECMConstraint:
    """A constraint that must be satisfied for ECM to be valid."""
    field: str  # Field in BuildingContext to check
    operator: str  # 'eq', 'ne', 'in', 'not_in', 'gt', 'lt', 'gte', 'lte'
    value: Any
    reason: str  # Human-readable explanation


@dataclass
class ECM:
    """Energy Conservation Measure definition."""
    id: str
    name: str
    name_sv: str  # Swedish name
    category: ECMCategory
    description: str

    # Cost estimation
    cost_per_unit: float  # SEK per unit (m², kW, etc.)
    cost_unit: str  # What the cost is per

    # Typical savings (for initial ranking)
    typical_savings_percent: float  # % reduction in relevant end use
    affected_end_use: str  # 'heating', 'cooling', 'electricity', 'all'

    # Implementation notes
    disruption_level: str  # 'low', 'medium', 'high'

    # Parameters that can be varied (defaults after required fields)
    parameters: List[ECMParameter] = field(default_factory=list)

    # Constraints - if ANY fail, ECM is not applicable
    constraints: List[ECMConstraint] = field(default_factory=list)

    # Optional fields with defaults
    fixed_cost: float = 0  # Fixed cost component
    typical_lifetime_years: int = 25


# =============================================================================
# SWEDISH ECM CATALOG
# =============================================================================

SWEDISH_ECM_CATALOG: Dict[str, ECM] = {

    # =========================================================================
    # ENVELOPE ECMs
    # =========================================================================

    "wall_external_insulation": ECM(
        id="wall_external_insulation",
        name="External Wall Insulation",
        name_sv="Tilläggsisolering utsida",
        category=ECMCategory.ENVELOPE,
        description="Add insulation to exterior of walls. Not suitable for brick facades.",
        parameters=[
            ECMParameter("thickness_mm", [50, 100, 150, 200], "mm", "Insulation thickness"),
            ECMParameter("material", ["mineral_wool", "eps", "pir"], "", "Insulation type"),
        ],
        constraints=[
            ECMConstraint("facade_material", "not_in", ["brick"],
                         "Cannot add external insulation to brick facade - destroys appearance"),
            ECMConstraint("heritage_listed", "eq", False,
                         "Heritage buildings cannot have exterior changes"),
        ],
        cost_per_unit=1500,  # SEK/m² wall area
        cost_unit="m² wall",
        typical_savings_percent=20,
        affected_end_use="heating",
        disruption_level="high",
        typical_lifetime_years=40,
    ),

    "wall_internal_insulation": ECM(
        id="wall_internal_insulation",
        name="Internal Wall Insulation",
        name_sv="Tilläggsisolering insida",
        category=ECMCategory.ENVELOPE,
        description="Add insulation to interior of walls. Use for brick buildings. Creates thermal bridges.",
        parameters=[
            ECMParameter("thickness_mm", [30, 50, 80], "mm", "Insulation thickness"),
        ],
        constraints=[
            # No hard constraints - but note in description about thermal bridges
        ],
        cost_per_unit=800,  # SEK/m² wall area
        cost_unit="m² wall",
        typical_savings_percent=10,  # Less effective due to thermal bridges
        affected_end_use="heating",
        disruption_level="high",  # Requires relocating residents
        typical_lifetime_years=40,
    ),

    "roof_insulation": ECM(
        id="roof_insulation",
        name="Roof Insulation Upgrade",
        name_sv="Tilläggsisolering tak",
        category=ECMCategory.ENVELOPE,
        description="Add insulation to roof/attic. Easy access on flat roofs.",
        parameters=[
            ECMParameter("thickness_mm", [100, 150, 200, 300], "mm", "Added insulation"),
        ],
        constraints=[
            # Generally always possible
        ],
        cost_per_unit=400,  # SEK/m² roof area
        cost_unit="m² roof",
        typical_savings_percent=10,
        affected_end_use="heating",
        disruption_level="low",
        typical_lifetime_years=40,
    ),

    "window_replacement": ECM(
        id="window_replacement",
        name="Window Replacement",
        name_sv="Fönsterbyte",
        category=ECMCategory.ENVELOPE,
        description="Replace windows with modern triple glazing.",
        parameters=[
            ECMParameter("u_value", [1.0, 0.9, 0.8], "W/m²K", "Window U-value"),
            ECMParameter("shgc", [0.4, 0.5, 0.6], "", "Solar heat gain coefficient"),
        ],
        constraints=[
            ECMConstraint("heritage_listed", "eq", False,
                         "Heritage buildings must keep original windows"),
            ECMConstraint("current_window_u", "gt", 1.2,
                         "Windows already efficient, limited benefit"),
        ],
        cost_per_unit=6000,  # SEK/m² window area
        cost_unit="m² window",
        typical_savings_percent=15,
        affected_end_use="heating",
        disruption_level="medium",
        typical_lifetime_years=30,
    ),

    "air_sealing": ECM(
        id="air_sealing",
        name="Air Sealing",
        name_sv="Tätning",
        category=ECMCategory.ENVELOPE,
        description="Seal air leakage paths. Very cost-effective.",
        parameters=[
            ECMParameter("reduction_factor", [0.5, 0.7], "", "ACH reduction factor"),
        ],
        constraints=[
            ECMConstraint("current_infiltration_ach", "gt", 0.05,
                         "Building already very airtight"),
        ],
        cost_per_unit=50,  # SEK/m² floor area
        cost_unit="m² floor",
        typical_savings_percent=8,
        affected_end_use="heating",
        disruption_level="low",
        typical_lifetime_years=20,
    ),

    # =========================================================================
    # HVAC ECMs
    # =========================================================================

    "ftx_upgrade": ECM(
        id="ftx_upgrade",
        name="FTX Heat Recovery Upgrade",
        name_sv="FTX-uppgradering",
        category=ECMCategory.HVAC,
        description="Upgrade heat recovery unit to higher efficiency.",
        parameters=[
            ECMParameter("effectiveness", [0.80, 0.85, 0.90], "", "Heat recovery effectiveness"),
        ],
        constraints=[
            ECMConstraint("ventilation_type", "in", ["ftx", "f"],
                         "Requires existing mechanical ventilation"),
            ECMConstraint("current_heat_recovery", "lt", 0.80,
                         "Already high efficiency, limited benefit"),
        ],
        cost_per_unit=200,  # SEK/m² floor area
        cost_unit="m² floor",
        typical_savings_percent=20,
        affected_end_use="heating",
        disruption_level="medium",
        typical_lifetime_years=20,
    ),

    "ftx_installation": ECM(
        id="ftx_installation",
        name="FTX Installation (from F or natural)",
        name_sv="FTX-installation",
        category=ECMCategory.HVAC,
        description="Install balanced ventilation with heat recovery. Major improvement for F or natural ventilation.",
        parameters=[
            ECMParameter("effectiveness", [0.75, 0.80, 0.85], "", "Heat recovery effectiveness"),
        ],
        constraints=[
            ECMConstraint("ventilation_type", "in", ["f", "natural"],
                         "Building must have F or natural ventilation"),
        ],
        cost_per_unit=400,  # SEK/m² floor area (new installation)
        cost_unit="m² floor",
        typical_savings_percent=35,
        affected_end_use="heating",
        disruption_level="high",
        typical_lifetime_years=25,
    ),

    "demand_controlled_ventilation": ECM(
        id="demand_controlled_ventilation",
        name="Demand Controlled Ventilation",
        name_sv="Behovsstyrd ventilation",
        category=ECMCategory.HVAC,
        description="Add CO2/humidity sensors to control ventilation rate.",
        parameters=[
            ECMParameter("co2_setpoint", [800, 1000], "ppm", "CO2 setpoint"),
        ],
        constraints=[
            ECMConstraint("ventilation_type", "in", ["ftx", "f"],
                         "Requires mechanical ventilation"),
        ],
        cost_per_unit=150,  # SEK/m² floor area
        cost_unit="m² floor",
        typical_savings_percent=15,
        affected_end_use="heating",
        disruption_level="low",
        typical_lifetime_years=15,
    ),

    "heat_pump_integration": ECM(
        id="heat_pump_integration",
        name="Heat Pump Integration",
        name_sv="Värmepumpsinstallation",
        category=ECMCategory.HVAC,
        description="Add heat pump to reduce purchased heating. Best for electric/oil heated buildings.",
        parameters=[
            ECMParameter("cop", [3.0, 3.5, 4.0], "", "Coefficient of Performance"),
            ECMParameter("coverage", [0.6, 0.8, 1.0], "", "Fraction of load covered"),
        ],
        constraints=[
            ECMConstraint("heating_system", "not_in", ["heat_pump_ground", "heat_pump_air"],
                         "Already has heat pump"),
            ECMConstraint("has_hydronic_distribution", "eq", True,
                         "Requires hydronic heating system for retrofit"),
            # Note: District heating areas may not benefit much (already low-carbon)
        ],
        cost_per_unit=3000,  # SEK/kW heating capacity
        cost_unit="kW",
        typical_savings_percent=60,  # Primary energy reduction
        affected_end_use="heating",
        disruption_level="high",
        typical_lifetime_years=20,
    ),

    # =========================================================================
    # RENEWABLE ECMs
    # =========================================================================

    "solar_pv": ECM(
        id="solar_pv",
        name="Solar PV Installation",
        name_sv="Solceller",
        category=ECMCategory.RENEWABLE,
        description="Install rooftop solar PV. Best on flat roofs facing south.",
        parameters=[
            ECMParameter("coverage_fraction", [0.5, 0.7, 0.9], "", "Roof coverage"),
            ECMParameter("panel_efficiency", [0.18, 0.20, 0.22], "", "Panel efficiency"),
        ],
        constraints=[
            ECMConstraint("roof_type", "in", ["flat", "pitched_south"],
                         "Requires suitable roof orientation"),
            ECMConstraint("available_pv_area_m2", "gt", 50,
                         "Insufficient roof area for viable system"),
            ECMConstraint("shading_factor", "lt", 0.3,
                         "Too much shading for viable PV"),
        ],
        cost_per_unit=12000,  # SEK/kWp installed
        cost_unit="kWp",
        typical_savings_percent=20,  # Of electricity use
        affected_end_use="electricity",
        disruption_level="low",
        typical_lifetime_years=25,
    ),

    # =========================================================================
    # CONTROLS ECMs
    # =========================================================================

    "smart_thermostats": ECM(
        id="smart_thermostats",
        name="Smart Thermostats",
        name_sv="Smarta termostater",
        category=ECMCategory.CONTROLS,
        description="Install smart thermostats with presence detection and optimization.",
        parameters=[
            ECMParameter("setback_c", [1, 2, 3], "°C", "Night/away setback"),
        ],
        constraints=[
            # Generally always applicable
        ],
        cost_per_unit=30,  # SEK/m² floor area
        cost_unit="m² floor",
        typical_savings_percent=5,
        affected_end_use="heating",
        disruption_level="low",
        typical_lifetime_years=10,
    ),

    # =========================================================================
    # LIGHTING ECMs
    # =========================================================================

    "led_lighting": ECM(
        id="led_lighting",
        name="LED Lighting Retrofit",
        name_sv="LED-belysning",
        category=ECMCategory.LIGHTING,
        description="Replace all lighting with LED. Quick payback.",
        parameters=[
            ECMParameter("power_density", [4, 6], "W/m²", "New lighting power density"),
        ],
        constraints=[
            ECMConstraint("current_lighting_w_m2", "gt", 6,
                         "Lighting already efficient"),
        ],
        cost_per_unit=100,  # SEK/m² floor area
        cost_unit="m² floor",
        typical_savings_percent=50,  # Of lighting energy
        affected_end_use="electricity",
        disruption_level="low",
        typical_lifetime_years=15,
    ),

    # =========================================================================
    # OPERATIONAL ECMs (Zero/Low Cost)
    # =========================================================================

    "duc_calibration": ECM(
        id="duc_calibration",
        name="DUC Calibration",
        name_sv="DUC-optimering",
        category=ECMCategory.OPERATIONAL,
        description="Optimize district heating control unit (DUC/UC) settings. "
                   "Adjusts heating curve, night setback, outdoor reset.",
        parameters=[
            ECMParameter("heating_curve_offset", [-2, -1, 0], "°C", "Curve offset"),
            ECMParameter("night_setback", [1, 2, 3], "°C", "Night setback"),
        ],
        constraints=[
            ECMConstraint("heating_system", "in", ["district", "central_boiler"],
                         "Requires central heating with DUC"),
        ],
        cost_per_unit=0,  # Zero material cost
        cost_unit="building",
        typical_savings_percent=5,
        affected_end_use="heating",
        disruption_level="none",
        typical_lifetime_years=5,
    ),

    "effektvakt_optimization": ECM(
        id="effektvakt_optimization",
        name="Effektvakt Optimization",
        name_sv="Effektvaktsoptimering",
        category=ECMCategory.OPERATIONAL,
        description="Optimize power guard (effektvakt) settings to reduce peak demand "
                   "and tariff charges. No energy savings but cost reduction.",
        parameters=[
            ECMParameter("peak_reduction", [10, 15, 20], "%", "Peak demand reduction"),
        ],
        constraints=[
            ECMConstraint("has_effektvakt", "eq", True,
                         "Requires existing effektvakt installation"),
        ],
        cost_per_unit=0,
        cost_unit="building",
        typical_savings_percent=0,  # Cost savings, not energy
        affected_end_use="heating",
        disruption_level="none",
        typical_lifetime_years=5,
    ),

    "heating_curve_adjustment": ECM(
        id="heating_curve_adjustment",
        name="Heating Curve Adjustment",
        name_sv="Värmekurvejustering",
        category=ECMCategory.OPERATIONAL,
        description="Optimize supply water temperature (framledningstemperatur) based on "
                   "actual building response. Often set too high by default.",
        parameters=[
            ECMParameter("curve_reduction", [2, 4, 6], "°C", "Curve reduction"),
        ],
        constraints=[
            ECMConstraint("heating_system", "in", ["district", "central_boiler", "heat_pump_ground"],
                         "Requires hydronic heating system"),
        ],
        cost_per_unit=0,
        cost_unit="building",
        typical_savings_percent=5,
        affected_end_use="heating",
        disruption_level="none",
        typical_lifetime_years=3,
    ),

    "ventilation_schedule_optimization": ECM(
        id="ventilation_schedule_optimization",
        name="Ventilation Schedule Optimization",
        name_sv="Ventilationsschemaoptimering",
        category=ECMCategory.OPERATIONAL,
        description="Adjust ventilation schedules to actual occupancy patterns. "
                   "Many buildings run full ventilation 24/7 unnecessarily.",
        parameters=[
            ECMParameter("night_reduction", [30, 50, 70], "%", "Night flow reduction"),
        ],
        constraints=[
            ECMConstraint("ventilation_type", "in", ["ftx", "f"],
                         "Requires mechanical ventilation with variable speed"),
        ],
        cost_per_unit=0,
        cost_unit="building",
        typical_savings_percent=5,
        affected_end_use="heating",
        disruption_level="none",
        typical_lifetime_years=3,
    ),

    "radiator_balancing": ECM(
        id="radiator_balancing",
        name="Radiator Balancing",
        name_sv="Injustering av radiatorer",
        category=ECMCategory.OPERATIONAL,
        description="Hydraulic balancing of radiator system. Ensures even heat "
                   "distribution, prevents overheating in some apartments.",
        parameters=[],
        constraints=[
            ECMConstraint("heating_distribution", "eq", "radiator",
                         "Requires radiator heating system"),
        ],
        cost_per_unit=200,  # Per radiator
        cost_unit="radiator",
        typical_savings_percent=5,
        affected_end_use="heating",
        disruption_level="low",
        typical_lifetime_years=10,
    ),

    "night_setback": ECM(
        id="night_setback",
        name="Night Setback",
        name_sv="Nattsänkning",
        category=ECMCategory.OPERATIONAL,
        description="Reduce heating setpoint 2-3°C during unoccupied hours (22:00-06:00). "
                   "Most buildings have this feature available but disabled.",
        parameters=[
            ECMParameter("setback_c", [1, 2, 3], "°C", "Setback amount"),
            ECMParameter("start_hour", [21, 22, 23], "h", "Start hour"),
            ECMParameter("end_hour", [5, 6, 7], "h", "End hour"),
        ],
        constraints=[],  # Works for all buildings with heating
        cost_per_unit=0,
        cost_unit="building",
        fixed_cost=1000,  # BMS programming
        typical_savings_percent=5,
        affected_end_use="heating",
        disruption_level="none",
        typical_lifetime_years=5,
    ),

    "summer_bypass": ECM(
        id="summer_bypass",
        name="Summer Bypass",
        name_sv="Sommaravstängning",
        category=ECMCategory.OPERATIONAL,
        description="Disable heating when outdoor temperature exceeds threshold (17°C). "
                   "Prevents unnecessary heating during warm periods.",
        parameters=[
            ECMParameter("threshold_c", [15, 17, 19], "°C", "Outdoor temp threshold"),
        ],
        constraints=[],
        cost_per_unit=0,
        cost_unit="building",
        fixed_cost=500,
        typical_savings_percent=3,
        affected_end_use="heating",
        disruption_level="none",
        typical_lifetime_years=5,
    ),

    "hot_water_temperature": ECM(
        id="hot_water_temperature",
        name="DHW Temperature Reduction",
        name_sv="Sänkt varmvattentemperatur",
        category=ECMCategory.OPERATIONAL,
        description="Reduce domestic hot water setpoint from 60°C to 55°C where safe. "
                   "Requires circulation to prevent Legionella.",
        parameters=[
            ECMParameter("target_temp_c", [55, 57], "°C", "Target temperature"),
        ],
        constraints=[
            ECMConstraint("has_dhw_circulation", "eq", True,
                         "Requires DHW circulation for Legionella safety"),
        ],
        cost_per_unit=0,
        cost_unit="building",
        fixed_cost=500,
        typical_savings_percent=3,
        affected_end_use="heating",
        disruption_level="none",
        typical_lifetime_years=5,
    ),

    "pump_optimization": ECM(
        id="pump_optimization",
        name="Pump Speed Optimization",
        name_sv="Pumpoptimering",
        category=ECMCategory.OPERATIONAL,
        description="Reduce circulation pump speeds. Many pumps run at full speed "
                   "unnecessarily. Variable speed saves 30-50% pump electricity.",
        parameters=[
            ECMParameter("speed_reduction", [20, 30, 40], "%", "Speed reduction"),
        ],
        constraints=[
            ECMConstraint("has_variable_speed_pumps", "eq", True,
                         "Requires variable speed pumps or VFD"),
        ],
        cost_per_unit=0,
        cost_unit="building",
        fixed_cost=2000,
        typical_savings_percent=2,
        affected_end_use="electricity",
        disruption_level="none",
        typical_lifetime_years=5,
    ),

    "bms_optimization": ECM(
        id="bms_optimization",
        name="BMS Tune-Up",
        name_sv="Styr- och reglertrimning",
        category=ECMCategory.OPERATIONAL,
        description="Building Management System tune-up. Review all setpoints, schedules, "
                   "alarms. Often finds 5-10% savings from drift and incorrect settings.",
        parameters=[],
        constraints=[
            ECMConstraint("has_bms", "eq", True,
                         "Requires Building Management System"),
        ],
        cost_per_unit=0,
        cost_unit="building",
        fixed_cost=5000,  # Consultant analysis
        typical_savings_percent=5,
        affected_end_use="all",
        disruption_level="none",
        typical_lifetime_years=3,
    ),
}


class ECMCatalog:
    """
    Access and filter ECM catalog.

    Usage:
        catalog = ECMCatalog()
        envelope_ecms = catalog.by_category(ECMCategory.ENVELOPE)
        all_ecms = catalog.all()
    """

    def __init__(self, ecms: Dict[str, ECM] = None):
        self.ecms = ecms or SWEDISH_ECM_CATALOG

    def all(self) -> List[ECM]:
        """Get all ECMs."""
        return list(self.ecms.values())

    def get(self, ecm_id: str) -> Optional[ECM]:
        """Get ECM by ID."""
        return self.ecms.get(ecm_id)

    def by_category(self, category: ECMCategory) -> List[ECM]:
        """Get ECMs in a category."""
        return [ecm for ecm in self.ecms.values() if ecm.category == category]

    def by_end_use(self, end_use: str) -> List[ECM]:
        """Get ECMs affecting an end use."""
        return [ecm for ecm in self.ecms.values() if ecm.affected_end_use == end_use]


# =============================================================================
# CONVENIENCE FUNCTIONS AND ALIASES
# =============================================================================

# Alias for backward compatibility
ECM_CATALOG = SWEDISH_ECM_CATALOG


def get_all_ecms() -> List[ECM]:
    """Get all ECMs as a list."""
    return list(SWEDISH_ECM_CATALOG.values())


def get_ecm(ecm_id: str) -> Optional[ECM]:
    """Get an ECM by ID."""
    return SWEDISH_ECM_CATALOG.get(ecm_id)


def get_ecms_by_category(category: ECMCategory) -> List[ECM]:
    """Get all ECMs in a category."""
    return [ecm for ecm in SWEDISH_ECM_CATALOG.values() if ecm.category == category]


def list_ecm_ids() -> List[str]:
    """List all ECM IDs."""
    return list(SWEDISH_ECM_CATALOG.keys())
