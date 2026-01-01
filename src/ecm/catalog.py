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

    # IDF implementation status
    # 'full' = modifies IDF and affects thermal simulation
    # 'partial' = some IDF changes but incomplete model
    # 'comment_only' = adds comment but no thermal effect (no-op)
    # 'cost_only' = affects cost calculation, not thermal simulation
    idf_implementation: str = "full"


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

    "ftx_overhaul": ECM(
        id="ftx_overhaul",
        name="FTX System Overhaul/Repair",
        name_sv="FTX-renovering",
        category=ECMCategory.HVAC,
        description="Complete overhaul of malfunctioning FTX: clean heat exchanger, replace filters, repair fans/dampers, recalibrate. For systems not performing to spec.",
        parameters=[
            ECMParameter("target_effectiveness", [0.70, 0.75, 0.80], "", "Restored heat recovery effectiveness"),
        ],
        constraints=[
            ECMConstraint("ventilation_type", "in", ["ftx"],
                         "Building must have existing FTX"),
        ],
        cost_per_unit=50,  # SEK/m² floor area (much cheaper than replacement)
        cost_unit="m² floor",
        typical_savings_percent=25,  # If system was broken, this can be significant
        affected_end_use="heating",
        disruption_level="low",
        typical_lifetime_years=10,  # Until next major service
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

    # =========================================================================
    # SWEDISH-SPECIFIC ECMs (High-impact measures common in Sweden)
    # =========================================================================

    "exhaust_air_heat_pump": ECM(
        id="exhaust_air_heat_pump",
        name="Exhaust Air Heat Pump (FVP)",
        name_sv="Frånluftsvärmepump (FVP)",
        category=ECMCategory.HVAC,
        description="""
        Extract heat from exhaust ventilation air for heating and DHW.
        Very common retrofit for F-ventilated Swedish multi-family buildings.
        Uses NIBE F470/F750 or similar. COP 3.0-3.5.
        Cannot combine with FTX (competing for same heat source).
        """,
        parameters=[
            ECMParameter("capacity_kw", [5, 8, 12, 16], "kW", "Heat pump capacity"),
            ECMParameter("cop", [3.0, 3.2, 3.5], "", "Coefficient of Performance"),
        ],
        constraints=[
            ECMConstraint("ventilation_type", "in", ["f", "exhaust"],
                         "Requires F-ventilation (exhaust only) system"),
            ECMConstraint("has_ftx", "eq", False,
                         "Cannot combine with FTX - both use exhaust air"),
        ],
        cost_per_unit=12000,  # SEK per kW capacity
        cost_unit="kW",
        fixed_cost=40000,  # Installation base cost
        typical_savings_percent=50,  # Heating + DHW savings
        affected_end_use="heating",
        disruption_level="medium",
        typical_lifetime_years=20,
    ),

    "ground_source_heat_pump": ECM(
        id="ground_source_heat_pump",
        name="Ground Source Heat Pump",
        name_sv="Bergvärmepump",
        category=ECMCategory.HVAC,
        description="""
        Install ground source heat pump with borehole(s).
        High efficiency (COP 4-5) but requires drilling.
        Best for buildings without district heating.
        Works well with low-temperature distribution (underfloor/radiator fans).
        """,
        parameters=[
            ECMParameter("capacity_kw", [10, 15, 20, 30, 50], "kW", "Heat pump capacity"),
            ECMParameter("cop", [4.0, 4.5, 5.0], "", "Seasonal COP"),
            ECMParameter("borehole_depth_m", [100, 150, 200], "m", "Borehole depth"),
        ],
        constraints=[
            ECMConstraint("heating_system", "not_in", ["heat_pump_ground"],
                         "Already has ground source heat pump"),
            ECMConstraint("heating_system", "not_in", ["district"],
                         "District heating is typically more cost-effective"),
        ],
        cost_per_unit=10000,  # SEK per kW
        cost_unit="kW",
        fixed_cost=80000,  # Borehole drilling (~150m @ 300 SEK/m + setup)
        typical_savings_percent=65,  # Primary energy savings
        affected_end_use="heating",
        disruption_level="high",
        typical_lifetime_years=25,
    ),

    "district_heating_optimization": ECM(
        id="district_heating_optimization",
        name="District Heating Optimization",
        name_sv="Fjärrvärmeoptimering",
        category=ECMCategory.OPERATIONAL,
        description="""
        Optimize district heating substation for better efficiency:
        - Lower return temperature (many utilities penalize high return)
        - Higher delta-T (more efficient heat transfer)
        - Two-stage DHW preheating (use return water)
        - Correct sizing of heat exchangers

        Target: Return temp < 35°C, Delta-T > 40°C
        Often zero/low cost with significant tariff benefits.
        """,
        parameters=[
            ECMParameter("target_return_temp_c", [30, 35, 40], "°C", "Target return temperature"),
            ECMParameter("target_delta_t_c", [35, 40, 45], "°C", "Target delta-T"),
        ],
        constraints=[
            ECMConstraint("heating_system", "eq", "district",
                         "Requires district heating connection"),
        ],
        cost_per_unit=0,
        cost_unit="building",
        fixed_cost=15000,  # Substation adjustment + consultant
        typical_savings_percent=8,  # Cost savings from better tariff + less energy
        affected_end_use="heating",
        disruption_level="none",
        typical_lifetime_years=10,
    ),

    "solar_thermal": ECM(
        id="solar_thermal",
        name="Solar Thermal Collectors",
        name_sv="Solfångare",
        category=ECMCategory.RENEWABLE,
        description="""
        Install solar thermal collectors for DHW preheating.
        Well-suited for Swedish multi-family buildings with large DHW demand.
        Covers 30-50% of DHW energy in Stockholm climate.
        Best combined with accumulator tank.
        """,
        parameters=[
            ECMParameter("area_m2", [20, 40, 60, 100], "m²", "Collector area"),
            ECMParameter("collector_type", ["flat_plate", "vacuum_tube"], "", "Collector technology"),
        ],
        constraints=[
            ECMConstraint("has_roof_access", "eq", True,
                         "Requires suitable roof area for collectors"),
        ],
        cost_per_unit=8000,  # SEK per m² collector
        cost_unit="m²",
        fixed_cost=30000,  # Tank, piping, controls
        typical_savings_percent=35,  # DHW energy savings
        affected_end_use="dhw",
        disruption_level="low",
        typical_lifetime_years=25,
    ),

    "low_flow_fixtures": ECM(
        id="low_flow_fixtures",
        name="Low-Flow Water Fixtures",
        name_sv="Snålspolande armaturer",
        category=ECMCategory.OPERATIONAL,
        description="""
        Install low-flow showerheads and faucet aerators.
        Very low cost, easy to implement, immediate savings.
        Reduces both water and energy for DHW.
        """,
        parameters=[
            ECMParameter("showerhead_flow_lpm", [6, 8, 10], "L/min", "Showerhead flow rate"),
            ECMParameter("faucet_flow_lpm", [4, 6], "L/min", "Faucet flow rate"),
        ],
        constraints=[
            # No hard constraints - applicable everywhere
        ],
        cost_per_unit=500,  # SEK per apartment
        cost_unit="apartment",
        typical_savings_percent=25,  # DHW energy savings
        affected_end_use="dhw",
        disruption_level="none",
        typical_lifetime_years=10,
    ),

    # =========================================================================
    # ADDITIONAL ENVELOPE ECMs
    # =========================================================================

    "basement_insulation": ECM(
        id="basement_insulation",
        name="Basement Insulation",
        name_sv="Källarisolering",
        category=ECMCategory.ENVELOPE,
        description="Insulate basement ceiling or walls to reduce ground losses.",
        parameters=[
            ECMParameter("thickness_mm", [50, 100, 150], "mm", "Insulation thickness"),
            ECMParameter("location", ["ceiling", "walls"], "", "Where to insulate"),
        ],
        constraints=[
            ECMConstraint("has_basement", "eq", True, "Requires basement"),
        ],
        cost_per_unit=300,
        cost_unit="m²",
        typical_savings_percent=8,
        affected_end_use="heating",
        disruption_level="medium",
        typical_lifetime_years=40,
    ),

    "thermal_bridge_remediation": ECM(
        id="thermal_bridge_remediation",
        name="Thermal Bridge Remediation",
        name_sv="Köldbryggsåtgärd",
        category=ECMCategory.ENVELOPE,
        description="Address thermal bridges at balconies, floor/wall connections.",
        parameters=[
            ECMParameter("bridge_type", ["balcony", "foundation", "window_reveal"], "", "Type of bridge"),
        ],
        constraints=[
            ECMConstraint("has_balconies", "eq", True, "Most effective for buildings with balconies"),
        ],
        cost_per_unit=800,
        cost_unit="linear_m",
        fixed_cost=10000,
        typical_savings_percent=10,
        affected_end_use="heating",
        disruption_level="medium",
        typical_lifetime_years=30,
    ),

    "facade_renovation": ECM(
        id="facade_renovation",
        name="Complete Facade Renovation",
        name_sv="Total fasadrenovering",
        category=ECMCategory.ENVELOPE,
        description="Complete facade renovation package: insulation + windows + air sealing.",
        parameters=[
            ECMParameter("insulation_mm", [100, 150, 200], "mm", "Wall insulation"),
            ECMParameter("window_u", [0.8, 1.0, 1.2], "W/m²K", "Window U-value"),
        ],
        constraints=[
            ECMConstraint("facade_material", "not_in", ["brick"],
                         "Cannot modify brick facades"),
            ECMConstraint("heritage_listed", "eq", False, "No heritage buildings"),
        ],
        cost_per_unit=2500,
        cost_unit="m² facade",
        fixed_cost=150000,
        typical_savings_percent=40,
        affected_end_use="heating",
        disruption_level="high",
        typical_lifetime_years=40,
    ),

    "entrance_door_replacement": ECM(
        id="entrance_door_replacement",
        name="Entrance Door Replacement",
        name_sv="Dörrbytte",
        category=ECMCategory.ENVELOPE,
        description="Replace entrance doors with insulated, airtight doors.",
        parameters=[
            ECMParameter("door_u", [0.8, 1.0, 1.2], "W/m²K", "Door U-value"),
        ],
        constraints=[],
        cost_per_unit=25000,
        cost_unit="door",
        typical_savings_percent=3,
        affected_end_use="heating",
        disruption_level="low",
        typical_lifetime_years=30,
    ),

    # =========================================================================
    # ADDITIONAL HVAC ECMs
    # =========================================================================

    "air_source_heat_pump": ECM(
        id="air_source_heat_pump",
        name="Air Source Heat Pump",
        name_sv="Luftvärmepump",
        category=ECMCategory.HVAC,
        description="Install air-to-water heat pump for heating supplement.",
        parameters=[
            ECMParameter("capacity_kw", [8, 12, 16, 20], "kW", "Heating capacity"),
            ECMParameter("cop", [3.0, 3.5, 4.0], "", "Coefficient of Performance"),
        ],
        constraints=[
            ECMConstraint("heating_system", "not_in", ["district"],
                         "Not economical with district heating"),
        ],
        cost_per_unit=8000,
        cost_unit="kW",
        fixed_cost=40000,
        typical_savings_percent=45,
        affected_end_use="heating",
        disruption_level="medium",
        typical_lifetime_years=15,
    ),

    "radiator_fans": ECM(
        id="radiator_fans",
        name="Radiator Fans",
        name_sv="Radiatorfläktar",
        category=ECMCategory.HVAC,
        description="Add fans behind radiators to improve heat transfer, enable lower supply temps.",
        parameters=[
            ECMParameter("fan_type", ["passive", "active_thermostat"], "", "Fan control type"),
        ],
        constraints=[
            ECMConstraint("heating_distribution", "eq", "radiators", "Requires radiator system"),
        ],
        cost_per_unit=2000,
        cost_unit="radiator",
        typical_savings_percent=8,
        affected_end_use="heating",
        disruption_level="low",
        typical_lifetime_years=10,
    ),

    "heat_recovery_dhw": ECM(
        id="heat_recovery_dhw",
        name="DHW Drain Heat Recovery",
        name_sv="Spillvattenvärmeåtervinning",
        category=ECMCategory.HVAC,
        description="Recover heat from shower/bath drain water to preheat cold water.",
        parameters=[
            ECMParameter("effectiveness", [0.4, 0.5, 0.6], "", "Heat recovery effectiveness"),
        ],
        constraints=[],
        cost_per_unit=15000,
        cost_unit="apartment",
        typical_savings_percent=8,
        affected_end_use="dhw",
        disruption_level="medium",
        typical_lifetime_years=20,
    ),

    "vrf_system": ECM(
        id="vrf_system",
        name="VRF HVAC System",
        name_sv="VRF-system",
        category=ECMCategory.HVAC,
        description="Variable Refrigerant Flow system for heating/cooling.",
        parameters=[
            ECMParameter("capacity_kw", [20, 40, 60, 80], "kW", "System capacity"),
        ],
        constraints=[
            ECMConstraint("building_use", "in", ["commercial", "office"],
                         "Primarily for commercial buildings"),
        ],
        cost_per_unit=4000,
        cost_unit="kW",
        fixed_cost=100000,
        typical_savings_percent=30,
        affected_end_use="hvac",
        disruption_level="high",
        typical_lifetime_years=20,
    ),

    # =========================================================================
    # ADDITIONAL CONTROLS ECMs
    # =========================================================================

    "occupancy_sensors": ECM(
        id="occupancy_sensors",
        name="Occupancy Sensors",
        name_sv="Närvarosensorer",
        category=ECMCategory.CONTROLS,
        description="Motion/occupancy sensors for lighting and HVAC control in common areas.",
        parameters=[
            ECMParameter("sensor_type", ["pir", "ultrasonic", "dual"], "", "Sensor technology"),
        ],
        constraints=[],
        cost_per_unit=800,
        cost_unit="sensor",
        typical_savings_percent=20,
        affected_end_use="electricity",
        disruption_level="low",
        typical_lifetime_years=10,
    ),

    "daylight_sensors": ECM(
        id="daylight_sensors",
        name="Daylight Harvesting",
        name_sv="Dagsljusstyrning",
        category=ECMCategory.CONTROLS,
        description="Photocell-based lighting dimming based on available daylight.",
        parameters=[
            ECMParameter("dimming_type", ["stepped", "continuous"], "", "Dimming method"),
        ],
        constraints=[],
        cost_per_unit=1200,
        cost_unit="zone",
        typical_savings_percent=30,
        affected_end_use="electricity",
        disruption_level="low",
        typical_lifetime_years=15,
    ),

    "predictive_control": ECM(
        id="predictive_control",
        name="Predictive HVAC Control",
        name_sv="Prediktiv styrning",
        category=ECMCategory.CONTROLS,
        description="Weather-predictive heating optimization using forecasts.",
        parameters=[
            ECMParameter("forecast_horizon_h", [12, 24, 48], "hours", "Forecast lookahead"),
        ],
        constraints=[
            ECMConstraint("has_bms", "eq", True, "Requires Building Management System"),
        ],
        cost_per_unit=50,
        cost_unit="m²",
        fixed_cost=30000,
        typical_savings_percent=8,
        affected_end_use="heating",
        disruption_level="low",
        typical_lifetime_years=10,
    ),

    "fault_detection": ECM(
        id="fault_detection",
        name="Fault Detection & Diagnostics",
        name_sv="Feldetektering (FDD)",
        category=ECMCategory.CONTROLS,
        description="Automated detection of HVAC faults and inefficiencies.",
        parameters=[
            ECMParameter("coverage", ["hvac_only", "full_building"], "", "FDD scope"),
        ],
        constraints=[
            ECMConstraint("has_bms", "eq", True, "Requires BMS with trend data"),
        ],
        cost_per_unit=30,
        cost_unit="m²",
        fixed_cost=50000,
        typical_savings_percent=10,
        affected_end_use="all",
        disruption_level="none",
        typical_lifetime_years=10,
    ),

    "individual_metering": ECM(
        id="individual_metering",
        name="Individual Metering & Billing",
        name_sv="Individuell mätning (IMD)",
        category=ECMCategory.CONTROLS,
        description="Apartment-level metering with billing for heating/DHW.",
        parameters=[
            ECMParameter("meter_type", ["heat", "heat_dhw", "electricity"], "", "What to meter"),
        ],
        constraints=[],
        cost_per_unit=8000,
        cost_unit="apartment",
        fixed_cost=20000,
        typical_savings_percent=15,
        affected_end_use="heating",
        disruption_level="medium",
        typical_lifetime_years=15,
    ),

    # =========================================================================
    # ADDITIONAL LIGHTING ECMs
    # =========================================================================

    "led_common_areas": ECM(
        id="led_common_areas",
        name="LED Common Area Lighting",
        name_sv="LED trapphus/källare",
        category=ECMCategory.LIGHTING,
        description="Replace lighting in stairwells, basement, garage with LED.",
        parameters=[
            ECMParameter("occupancy_control", [True, False], "", "Add occupancy sensors"),
        ],
        constraints=[],
        cost_per_unit=60,
        cost_unit="m² common",
        typical_savings_percent=60,
        affected_end_use="electricity",
        disruption_level="low",
        typical_lifetime_years=15,
    ),

    "led_outdoor": ECM(
        id="led_outdoor",
        name="LED Outdoor Lighting",
        name_sv="LED utomhusbelysning",
        category=ECMCategory.LIGHTING,
        description="Replace outdoor/parking lighting with LED.",
        parameters=[
            ECMParameter("control_type", ["photocell", "timer", "adaptive"], "", "Control method"),
        ],
        constraints=[],
        cost_per_unit=3000,
        cost_unit="fixture",
        typical_savings_percent=60,
        affected_end_use="electricity",
        disruption_level="low",
        typical_lifetime_years=20,
    ),

    # =========================================================================
    # SERVICE HOT WATER ECMs
    # =========================================================================

    "dhw_circulation_optimization": ECM(
        id="dhw_circulation_optimization",
        name="DHW Circulation Optimization",
        name_sv="VVC-optimering",
        category=ECMCategory.OPERATIONAL,
        description="Optimize circulation pump schedules and temperature setpoints.",
        parameters=[
            ECMParameter("control_type", ["timer", "demand", "temperature"], "", "Control strategy"),
        ],
        constraints=[],
        cost_per_unit=0,
        cost_unit="building",
        fixed_cost=8000,
        typical_savings_percent=15,
        affected_end_use="dhw",
        disruption_level="none",
        typical_lifetime_years=5,
    ),

    "heat_pump_water_heater": ECM(
        id="heat_pump_water_heater",
        name="Heat Pump Water Heater",
        name_sv="Varmvattenpump",
        category=ECMCategory.HVAC,
        description="Replace electric/gas water heater with heat pump unit.",
        parameters=[
            ECMParameter("cop", [2.5, 3.0, 3.5], "", "Coefficient of Performance"),
            ECMParameter("capacity_l", [200, 300, 500], "liters", "Tank capacity"),
        ],
        constraints=[
            ECMConstraint("dhw_heating", "in", ["electric", "gas"],
                         "Best for replacing electric/gas water heating"),
        ],
        cost_per_unit=35000,
        cost_unit="unit",
        typical_savings_percent=60,
        affected_end_use="dhw",
        disruption_level="medium",
        typical_lifetime_years=15,
    ),

    "dhw_tank_insulation": ECM(
        id="dhw_tank_insulation",
        name="DHW Tank Insulation",
        name_sv="Ackumulatorisolering",
        category=ECMCategory.HVAC,  # Physical measure, not operational
        description="Add insulation jacket to hot water storage tank.",
        parameters=[
            ECMParameter("insulation_mm", [50, 80, 100], "mm", "Insulation thickness"),
        ],
        constraints=[],
        cost_per_unit=2500,
        cost_unit="tank",
        typical_savings_percent=8,
        affected_end_use="dhw",
        disruption_level="low",
        typical_lifetime_years=20,
    ),

    # =========================================================================
    # ADDITIONAL OPERATIONAL ECMs
    # =========================================================================

    "recommissioning": ECM(
        id="recommissioning",
        name="Building Recommissioning",
        name_sv="Ominjustering",
        category=ECMCategory.OPERATIONAL,
        description="Comprehensive system tune-up and optimization.",
        parameters=[
            ECMParameter("scope", ["hvac", "controls", "full"], "", "Commissioning scope"),
        ],
        constraints=[],
        cost_per_unit=20,
        cost_unit="m²",
        fixed_cost=30000,
        typical_savings_percent=10,
        affected_end_use="all",
        disruption_level="none",
        typical_lifetime_years=5,
    ),

    "energy_monitoring": ECM(
        id="energy_monitoring",
        name="Energy Monitoring System",
        name_sv="Energiövervakningssystem",
        category=ECMCategory.OPERATIONAL,
        description="Install submetering and monitoring platform for visibility.",
        parameters=[
            ECMParameter("detail_level", ["building", "system", "apartment"], "", "Metering granularity"),
        ],
        constraints=[],
        cost_per_unit=25,
        cost_unit="m²",
        fixed_cost=40000,
        typical_savings_percent=8,
        affected_end_use="all",
        disruption_level="low",
        typical_lifetime_years=15,
    ),

    "pipe_insulation": ECM(
        id="pipe_insulation",
        name="Pipe Insulation",
        name_sv="Rörisolering",
        category=ECMCategory.OPERATIONAL,
        description="Insulate heating and DHW pipes in unheated spaces.",
        parameters=[
            ECMParameter("insulation_mm", [30, 50, 80], "mm", "Insulation thickness"),
        ],
        constraints=[],
        cost_per_unit=200,
        cost_unit="m pipe",
        typical_savings_percent=4,
        affected_end_use="heating",
        disruption_level="low",
        typical_lifetime_years=30,
    ),

    # =========================================================================
    # ADDITIONAL RENEWABLE ECMs
    # =========================================================================

    "battery_storage": ECM(
        id="battery_storage",
        name="Battery Energy Storage",
        name_sv="Batterilagring",
        category=ECMCategory.RENEWABLE,
        description="Battery storage for solar PV self-consumption optimization.",
        parameters=[
            ECMParameter("capacity_kwh", [10, 20, 50, 100], "kWh", "Storage capacity"),
        ],
        constraints=[
            ECMConstraint("has_solar_pv", "eq", True,
                         "Battery storage requires existing or planned solar PV"),
        ],
        cost_per_unit=7000,
        cost_unit="kWh",
        typical_savings_percent=10,  # % of electricity (increased self-consumption)
        affected_end_use="electricity",
        disruption_level="low",
        typical_lifetime_years=15,
    ),

    "solar_thermal": ECM(
        id="solar_thermal",
        name="Solar Thermal Collectors",
        name_sv="Solfångare",
        category=ECMCategory.RENEWABLE,
        description="Solar thermal collectors for DHW and space heating support.",
        parameters=[
            ECMParameter("collector_area_m2", [20, 40, 60, 100], "m²", "Collector area"),
            ECMParameter("type", ["flat_plate", "vacuum_tube"], "", "Collector type"),
        ],
        constraints=[
            ECMConstraint("roof_usable_area_m2", "gte", 20,
                         "Insufficient roof area for solar thermal"),
            ECMConstraint("roof_slope_degrees", "lte", 60,
                         "Roof too steep for solar thermal installation"),
        ],
        cost_per_unit=4500,
        cost_unit="m² collector",
        fixed_cost=30000,
        typical_savings_percent=30,  # % of DHW energy
        affected_end_use="heating",
        disruption_level="medium",
        typical_lifetime_years=25,
    ),

    "building_automation_system": ECM(
        id="building_automation_system",
        name="Integrated Building Automation",
        name_sv="Fastighetsautomation",
        category=ECMCategory.CONTROLS,
        description="Comprehensive building automation system with centralized control.",
        parameters=[
            ECMParameter("scope", ["basic", "standard", "advanced"], "", "Automation level"),
        ],
        constraints=[],
        cost_per_unit=80,
        cost_unit="m²",
        fixed_cost=100000,
        typical_savings_percent=15,
        affected_end_use="all",
        disruption_level="medium",
        typical_lifetime_years=20,
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


# =============================================================================
# ECM IDF IMPLEMENTATION STATUS REGISTRY
# =============================================================================
# This registry tracks which ECMs have actual IDF implementations vs no-ops.
# ECMs marked as 'comment_only' or 'cost_only' should be excluded from
# thermal simulation recommendations but may still be shown for cost analysis.
#
# Status levels:
#   'full' - Complete IDF modification, affects thermal simulation
#   'partial' - Some IDF changes but incomplete (e.g., missing COP modeling)
#   'comment_only' - Adds IDF comment but no thermal effect (NO-OP!)
#   'cost_only' - Affects cost calculation only, not simulation

ECM_IMPLEMENTATION_STATUS: Dict[str, str] = {
    # ENVELOPE - All fully implemented
    "wall_external_insulation": "full",
    "wall_internal_insulation": "full",
    "roof_insulation": "full",
    "window_replacement": "full",
    "air_sealing": "full",
    "basement_insulation": "full",  # Adds XPS insulation layer to floor
    "thermal_bridge_remediation": "partial",  # Minor infiltration only
    "facade_renovation": "partial",  # Calls other methods
    "entrance_door_replacement": "full",  # Reduces infiltration 8%

    # HVAC - Mixed implementation
    "ftx_upgrade": "full",
    "ftx_installation": "full",
    "ftx_overhaul": "full",
    "demand_controlled_ventilation": "full",
    "heat_pump_integration": "full",  # HR improvement + infiltration + COP documentation
    "exhaust_air_heat_pump": "partial",  # Enables HR but no COP
    "ground_source_heat_pump": "partial",  # Enables HR but no COP
    "air_source_heat_pump": "partial",  # Only infiltration
    "heat_pump_water_heater": "full",  # DHW gains + HP electricity with COP
    "vrf_system": "comment_only",  # NO VRF objects created (EXCLUDED)
    "radiator_fans": "full",  # Setpoint reduction -1°C + fan electricity
    "heat_recovery_dhw": "full",  # DWHR modeled as negative DHW gains

    # RENEWABLE
    "solar_pv": "full",  # Generator:PVWatts with ElectricLoadCenter
    "solar_thermal": "full",  # Monthly schedule for seasonal DHW savings
    "battery_storage": "full",  # ElectricLoadCenter:Storage for peak shaving

    # CONTROLS - Partial implementations
    "smart_thermostats": "partial",  # Schedule but not fully integrated
    "occupancy_sensors": "partial",  # Lighting only, not HVAC
    "daylight_sensors": "partial",  # Lighting only
    "predictive_control": "partial",  # Hardcoded setpoint reduction
    "building_automation_system": "partial",  # Hardcoded setpoint reduction
    "fault_detection": "full",  # Reduces infiltration 3% + setpoint 0.3°C
    "individual_metering": "partial",  # Reduces activity only

    # LIGHTING
    "led_lighting": "full",
    "led_common_areas": "partial",  # Pattern matching only
    "led_outdoor": "full",  # Exterior lighting with scheduled savings

    # OPERATIONAL - Mixed implementation
    "duc_calibration": "partial",  # Creates schedule
    "effektvakt_optimization": "full",  # Peak hour setpoint reduction for demand shaving
    "heating_curve_adjustment": "full",  # Monthly outdoor reset schedule
    "ventilation_schedule_optimization": "partial",  # Creates schedule
    "radiator_balancing": "full",  # Reduces setpoint 0.5°C (balanced distribution)
    "night_setback": "full",  # Schedule:Compact with day/night setpoints
    "summer_bypass": "full",  # Seasonal heating schedule (Sep-May only)
    "hot_water_temperature": "full",  # Negative internal gains for pipe loss reduction
    "pump_optimization": "full",  # VFD savings with pump affinity law
    "bms_optimization": "full",  # Setpoint correction + schedule alignment
    "district_heating_optimization": "full",  # Small setpoint reduction from better control
    "recommissioning": "partial",  # Hardcoded setpoint reduction
    "energy_monitoring": "full",  # Behavioral savings + setpoint awareness

    # DHW - Fully implemented with internal gains approach
    "dhw_circulation_optimization": "full",  # Timer control reduces circ losses
    "dhw_tank_insulation": "full",  # Reduces standby losses via negative gains
    "low_flow_fixtures": "full",  # Negative internal gains for reduced DHW use
    "pipe_insulation": "full",  # Reduces pipe heat losses
}


def get_ecm_implementation_status(ecm_id: str) -> str:
    """Get IDF implementation status for an ECM."""
    return ECM_IMPLEMENTATION_STATUS.get(ecm_id, "full")


def is_ecm_implemented(ecm_id: str) -> bool:
    """Check if ECM has thermal simulation effect (not a no-op)."""
    status = get_ecm_implementation_status(ecm_id)
    return status in ("full", "partial")


def get_implemented_ecms() -> List[ECM]:
    """Get only ECMs with actual IDF implementations (full or partial)."""
    return [
        ecm for ecm in SWEDISH_ECM_CATALOG.values()
        if is_ecm_implemented(ecm.id)
    ]


def get_unimplemented_ecm_ids() -> List[str]:
    """Get IDs of ECMs that are no-ops (comment_only or cost_only)."""
    return [
        ecm_id for ecm_id, status in ECM_IMPLEMENTATION_STATUS.items()
        if status in ("comment_only", "cost_only")
    ]


def get_full_implementation_ecms() -> List[ECM]:
    """Get only ECMs with full IDF implementation."""
    return [
        ecm for ecm in SWEDISH_ECM_CATALOG.values()
        if ECM_IMPLEMENTATION_STATUS.get(ecm.id, "full") == "full"
    ]
