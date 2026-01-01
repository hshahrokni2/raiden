"""
Swedish Cost Database - ECM costs and energy prices.

Sources:
- BeBo (Beställargrupp Bostäder) - Real retrofit project data
- SABO (Sveriges Allmännyttiga Bostadsföretag) - Public housing statistics
- Energimyndigheten - Energy efficiency program costs
- SCB Byggkostnadsindex - Construction cost indices
- Sveby - Standard values for Swedish buildings
- Wikells Sektionsfakta - Industry standard costs

Prices in SEK, 2024 levels.
Updated: 2025-12-18
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class CostCategory(Enum):
    """ECM cost categories."""
    ZERO_COST = "zero_cost"           # Operational optimization, no investment
    LOW_COST = "low_cost"             # < 100 SEK/m²
    MEDIUM_COST = "medium_cost"       # 100-500 SEK/m²
    HIGH_COST = "high_cost"           # 500-1000 SEK/m²
    MAJOR_INVESTMENT = "major"        # > 1000 SEK/m²


@dataclass
class EnergyCost:
    """Energy cost parameters."""
    price_sek_per_kwh: float
    annual_escalation: float  # Expected annual price increase
    carbon_intensity_kg_per_kwh: float  # For CO2 calculations


# Swedish energy prices (2024)
ENERGY_PRICES: Dict[str, EnergyCost] = {
    "district_heating": EnergyCost(
        price_sek_per_kwh=0.80,
        annual_escalation=0.02,
        carbon_intensity_kg_per_kwh=0.05  # Very low in Sweden
    ),
    "electricity": EnergyCost(
        price_sek_per_kwh=1.50,  # Including grid fees, taxes
        annual_escalation=0.03,
        carbon_intensity_kg_per_kwh=0.02  # Swedish grid very clean
    ),
    "electricity_spot": EnergyCost(
        price_sek_per_kwh=0.80,  # Spot price only (volatile)
        annual_escalation=0.03,
        carbon_intensity_kg_per_kwh=0.02
    ),
    "natural_gas": EnergyCost(
        price_sek_per_kwh=1.20,
        annual_escalation=0.02,
        carbon_intensity_kg_per_kwh=0.20
    ),
    "oil": EnergyCost(
        price_sek_per_kwh=1.40,
        annual_escalation=0.02,
        carbon_intensity_kg_per_kwh=0.27
    ),
    "pellets": EnergyCost(
        price_sek_per_kwh=0.60,
        annual_escalation=0.02,
        carbon_intensity_kg_per_kwh=0.02
    ),
}


@dataclass
class ECMCost:
    """ECM cost parameters."""
    cost_per_unit: float  # SEK per unit
    unit: str  # What the unit is
    fixed_cost: float = 0  # Fixed cost component
    installation_fraction: float = 0.3  # Installation as fraction of material
    lifetime_years: int = 25
    maintenance_fraction: float = 0.01  # Annual maintenance as fraction of investment
    category: CostCategory = CostCategory.MEDIUM_COST
    typical_savings_percent: float = 5.0  # Expected energy savings
    source: str = ""  # Data source
    notes: str = ""


# Swedish ECM costs (2024)
# Sources: BeBo, SABO, Energimyndigheten, Wikells
ECM_COSTS: Dict[str, ECMCost] = {

    # =========================================================================
    # ZERO-COST / OPERATIONAL MEASURES
    # These require only time for analysis and adjustment, no material cost
    # =========================================================================

    "duc_calibration": ECMCost(
        cost_per_unit=0,
        unit="building",
        fixed_cost=5000,  # Consultant time for analysis (~4-8 hours)
        installation_fraction=0,
        lifetime_years=5,  # Needs periodic re-calibration
        category=CostCategory.ZERO_COST,
        typical_savings_percent=5,
        source="BeBo",
        notes="District Heating Control Unit (DUC/UC) curve optimization. "
              "Adjusts heating curve, night setback, outdoor reset."
    ),

    "effektvakt_optimization": ECMCost(
        cost_per_unit=0,
        unit="building",
        fixed_cost=3000,  # Analysis and adjustment
        installation_fraction=0,
        lifetime_years=5,
        category=CostCategory.ZERO_COST,
        typical_savings_percent=3,
        source="Energimyndigheten",
        notes="Power guard (effektvakt) optimization. Reduces peak demand "
              "charges by better load scheduling."
    ),

    "heating_curve_adjustment": ECMCost(
        cost_per_unit=0,
        unit="building",
        fixed_cost=2000,
        installation_fraction=0,
        lifetime_years=3,
        category=CostCategory.ZERO_COST,
        typical_savings_percent=5,
        source="Sveby",
        notes="Optimize framledningstemperatur curve based on actual building "
              "response. Often set too high by default."
    ),

    "ventilation_schedule_optimization": ECMCost(
        cost_per_unit=0,
        unit="building",
        fixed_cost=2000,
        installation_fraction=0,
        lifetime_years=3,
        category=CostCategory.ZERO_COST,
        typical_savings_percent=5,
        source="BeBo",
        notes="Adjust ventilation schedules to actual occupancy. Many buildings "
              "run full ventilation 24/7 unnecessarily."
    ),

    "radiator_balancing": ECMCost(
        cost_per_unit=200,  # Per radiator
        unit="radiator",
        fixed_cost=5000,
        installation_fraction=0.8,
        lifetime_years=10,
        category=CostCategory.LOW_COST,
        typical_savings_percent=5,
        source="SABO",
        notes="Hydraulic balancing of radiator system. Ensures even heat "
              "distribution, prevents overheating in some apartments."
    ),

    "night_setback": ECMCost(
        cost_per_unit=0,
        unit="building",
        fixed_cost=1000,  # BMS programming
        installation_fraction=0,
        lifetime_years=5,
        category=CostCategory.ZERO_COST,
        typical_savings_percent=5,
        source="Sveby",
        notes="Reduce heating setpoint 2-3°C during unoccupied hours (22:00-06:00). "
              "Most buildings have this feature disabled."
    ),

    "summer_bypass": ECMCost(
        cost_per_unit=0,
        unit="building",
        fixed_cost=500,
        installation_fraction=0,
        lifetime_years=5,
        category=CostCategory.ZERO_COST,
        typical_savings_percent=3,
        source="BeBo",
        notes="Disable heating when outdoor temp > 17°C. Prevents unnecessary "
              "heating during warm periods."
    ),

    "hot_water_temperature": ECMCost(
        cost_per_unit=0,
        unit="building",
        fixed_cost=500,
        installation_fraction=0,
        lifetime_years=5,
        category=CostCategory.ZERO_COST,
        typical_savings_percent=3,
        source="Energimyndigheten",
        notes="Reduce DHW setpoint from 60°C to 55°C where safe (with circulation). "
              "Each degree saves ~3% DHW energy."
    ),

    "pump_optimization": ECMCost(
        cost_per_unit=0,
        unit="building",
        fixed_cost=2000,
        installation_fraction=0,
        lifetime_years=5,
        category=CostCategory.ZERO_COST,
        typical_savings_percent=2,
        source="Energimyndigheten",
        notes="Reduce circulation pump speeds. Many pumps run at full speed "
              "unnecessarily. Variable speed saves 30-50% pump electricity."
    ),

    "bms_optimization": ECMCost(
        cost_per_unit=0,
        unit="building",
        fixed_cost=5000,
        installation_fraction=0,
        lifetime_years=3,
        category=CostCategory.ZERO_COST,
        typical_savings_percent=5,
        source="BeBo",
        notes="Building Management System tune-up. Review all setpoints, schedules, "
              "alarms. Often finds 5-10% savings from drift and incorrect settings."
    ),

    # =========================================================================
    # LOW-COST MEASURES (< 100 SEK/m²)
    # Quick wins with short payback
    # =========================================================================

    "smart_thermostats": ECMCost(
        cost_per_unit=30,
        unit="m² floor",
        installation_fraction=0.5,
        lifetime_years=10,
        category=CostCategory.LOW_COST,
        typical_savings_percent=5,
        source="Energimyndigheten",
        notes="Individual room temperature control with night setback."
    ),

    "air_sealing": ECMCost(
        cost_per_unit=50,
        unit="m² floor",
        installation_fraction=0.7,  # Mostly labor
        lifetime_years=20,
        category=CostCategory.LOW_COST,
        typical_savings_percent=10,
        source="BeBo",
        notes="Seal air leakage paths: windows, doors, penetrations. "
              "Cost varies with building condition."
    ),

    "led_lighting": ECMCost(
        cost_per_unit=80,
        unit="m² floor",
        installation_fraction=0.4,
        lifetime_years=15,
        category=CostCategory.LOW_COST,
        typical_savings_percent=3,  # On total energy; 50% on lighting
        source="Energimyndigheten",
        notes="Replace fluorescent/incandescent with LED. Note: reduces "
              "internal heat gain, may increase heating in Nordic climates."
    ),

    # =========================================================================
    # MEDIUM-COST MEASURES (100-500 SEK/m²)
    # Significant savings with reasonable payback
    # =========================================================================

    "demand_controlled_ventilation": ECMCost(
        cost_per_unit=150,
        unit="m² floor",
        installation_fraction=0.3,
        lifetime_years=15,
        category=CostCategory.MEDIUM_COST,
        typical_savings_percent=15,
        source="BeBo",
        notes="CO2/humidity-controlled ventilation. Reduces ventilation "
              "heat losses during low occupancy."
    ),

    "ftx_upgrade": ECMCost(
        cost_per_unit=200,
        unit="m² floor",
        installation_fraction=0.4,
        lifetime_years=20,
        category=CostCategory.MEDIUM_COST,
        typical_savings_percent=10,
        source="SABO",
        notes="Upgrade existing FTX heat exchanger to higher efficiency. "
              "75% → 85% effectiveness."
    ),

    "roof_insulation": ECMCost(
        cost_per_unit=400,
        unit="m² roof",
        installation_fraction=0.3,
        lifetime_years=40,
        category=CostCategory.MEDIUM_COST,
        typical_savings_percent=5,
        source="Wikells",
        notes="Add 150-200mm insulation to attic. Easier access than walls."
    ),

    "ftx_installation": ECMCost(
        cost_per_unit=1200,
        unit="m² floor",
        installation_fraction=0.5,
        lifetime_years=25,
        category=CostCategory.HIGH_COST,
        typical_savings_percent=35,
        source="BeBo",
        notes="New FTX system with 80% heat recovery. Major intervention, "
              "best combined with other renovation."
    ),

    # =========================================================================
    # HIGH-COST MEASURES (500-1000 SEK/m²)
    # Major improvements, longer payback
    # =========================================================================

    "wall_internal_insulation": ECMCost(
        cost_per_unit=800,
        unit="m² wall",
        installation_fraction=0.5,
        lifetime_years=40,
        category=CostCategory.HIGH_COST,
        typical_savings_percent=15,
        source="Wikells",
        notes="Interior wall insulation 50-100mm. Reduces floor area slightly. "
              "Risk of moisture issues if not done correctly."
    ),

    # =========================================================================
    # MAJOR INVESTMENTS (> 1000 SEK/m²)
    # Large-scale renovation measures
    # =========================================================================

    "wall_external_insulation": ECMCost(
        cost_per_unit=1500,
        unit="m² wall",
        installation_fraction=0.4,
        lifetime_years=40,
        category=CostCategory.MAJOR_INVESTMENT,
        typical_savings_percent=20,
        source="BeBo",
        notes="ETICS external insulation system. 100-150mm. Changes building "
              "appearance, not suitable for brick/heritage facades."
    ),

    "window_replacement": ECMCost(
        cost_per_unit=6000,
        unit="m² window",
        installation_fraction=0.35,
        lifetime_years=30,
        category=CostCategory.MAJOR_INVESTMENT,
        typical_savings_percent=15,
        source="Wikells",
        notes="Replace to U=0.9 triple glazing. High cost but long lifetime."
    ),

    "heat_pump_integration": ECMCost(
        cost_per_unit=3000,
        unit="kW",
        fixed_cost=80000,
        installation_fraction=0.25,
        lifetime_years=20,
        category=CostCategory.MAJOR_INVESTMENT,
        typical_savings_percent=60,  # Primary energy, not thermal
        source="SABO",
        notes="Ground/exhaust air heat pump. Reduces purchased energy 60-70% "
              "but requires electricity. Best for non-district-heating."
    ),

    "solar_pv": ECMCost(
        cost_per_unit=12000,
        unit="kWp",
        fixed_cost=25000,
        installation_fraction=0.3,
        lifetime_years=25,
        maintenance_fraction=0.005,
        category=CostCategory.MAJOR_INVESTMENT,
        typical_savings_percent=10,  # Of total, depends on self-consumption
        source="Energimyndigheten",
        notes="Rooftop PV. ~150 kWp/m². Best economics with high self-consumption."
    ),

    # =========================================================================
    # SWEDISH-SPECIFIC QUICK WINS
    # High-impact measures common in Swedish multi-family buildings
    # =========================================================================

    "exhaust_air_heat_pump": ECMCost(
        cost_per_unit=12000,
        unit="kW",
        fixed_cost=50000,
        installation_fraction=0.25,
        lifetime_years=15,
        maintenance_fraction=0.02,
        category=CostCategory.HIGH_COST,
        typical_savings_percent=50,
        source="BeBo",
        notes="Frånluftsvärmepump (FVP). Recovers heat from exhaust air to DHW "
              "or heating. Best for buildings with F-ventilation (exhaust only). "
              "Common in 1970s-1990s buildings without FTX."
    ),

    "ground_source_heat_pump": ECMCost(
        cost_per_unit=15000,
        unit="kW",
        fixed_cost=200000,  # Borehole drilling significant cost
        installation_fraction=0.20,
        lifetime_years=25,
        maintenance_fraction=0.01,
        category=CostCategory.MAJOR_INVESTMENT,
        typical_savings_percent=65,
        source="SABO",
        notes="Bergvärmepump. High initial cost due to borehole drilling "
              "(~300 SEK/m borehole). COP 4-5. Best economics for oil/gas "
              "heated buildings or areas without district heating."
    ),

    "district_heating_optimization": ECMCost(
        cost_per_unit=0,
        unit="building",
        fixed_cost=15000,  # Consultant + equipment adjustment
        installation_fraction=0,
        lifetime_years=5,
        category=CostCategory.LOW_COST,
        typical_savings_percent=8,
        source="BeBo",
        notes="Fjärrvärmeoptimering. Substation optimization: better ΔT, "
              "reduced return temperature (bonus from energy company), "
              "weather compensation fine-tuning. Often 5-15% savings."
    ),

    "solar_thermal": ECMCost(
        cost_per_unit=8000,
        unit="m² collector",
        fixed_cost=40000,  # Storage tank, piping, controls
        installation_fraction=0.35,
        lifetime_years=25,
        maintenance_fraction=0.01,
        category=CostCategory.HIGH_COST,
        typical_savings_percent=15,  # DHW fraction
        source="Energimyndigheten",
        notes="Solfångare. Typically covers 40-60% of DHW in summer, "
              "less in winter. ~2-3 m² per apartment for multi-family. "
              "Declining market share due to PV economics."
    ),

    "low_flow_fixtures": ECMCost(
        cost_per_unit=1500,
        unit="apartment",
        fixed_cost=5000,
        installation_fraction=0.4,
        lifetime_years=15,
        category=CostCategory.LOW_COST,
        typical_savings_percent=5,  # DHW savings ~20%, total ~5%
        source="Energimyndigheten",
        notes="Snålspolande blandare och duschmunstycken. Reduces DHW "
              "consumption 20-30%. Quick payback. Include in ROT renovation."
    ),

    "imb_valves": ECMCost(
        cost_per_unit=3500,
        unit="apartment",
        fixed_cost=10000,  # System design and commissioning
        installation_fraction=0.3,
        lifetime_years=20,
        category=CostCategory.MEDIUM_COST,
        typical_savings_percent=8,
        source="BeBo",
        notes="Individuell mätning och debitering (IMD). Apartment-level "
              "metering with billing. Behavioral savings 10-20%. Required "
              "for new buildings, retrofit challenging."
    ),

    "ftx_cleaning": ECMCost(
        cost_per_unit=50,
        unit="m² floor",
        fixed_cost=10000,
        installation_fraction=0.8,
        lifetime_years=5,  # Should be done every 5 years
        category=CostCategory.LOW_COST,
        typical_savings_percent=5,
        source="BeBo",
        notes="OVK-besiktning och kanalrensning. Heat exchanger and duct "
              "cleaning restores efficiency. Dirty HRV can lose 10-20% "
              "effectiveness."
    ),

    "attic_hatch_insulation": ECMCost(
        cost_per_unit=2000,
        unit="hatch",
        fixed_cost=1000,
        installation_fraction=0.5,
        lifetime_years=30,
        category=CostCategory.LOW_COST,
        typical_savings_percent=1,
        source="Sveby",
        notes="Isolerad vindslucka. Often overlooked thermal bridge. "
              "Include during attic insulation work."
    ),

    "stairwell_door_sealing": ECMCost(
        cost_per_unit=5000,
        unit="entrance",
        fixed_cost=2000,
        installation_fraction=0.6,
        lifetime_years=15,
        category=CostCategory.LOW_COST,
        typical_savings_percent=2,
        source="BeBo",
        notes="Dörrförslutare och tätningslister på entrédörrar. Reduces "
              "cold air infiltration through stairwells. Often combined "
              "with air sealing package."
    ),

    "pipe_insulation": ECMCost(
        cost_per_unit=200,
        unit="m pipe",
        fixed_cost=5000,
        installation_fraction=0.7,
        lifetime_years=30,
        category=CostCategory.LOW_COST,
        typical_savings_percent=3,
        source="Energimyndigheten",
        notes="Rörisolering i källare och undercentral. Uninsulated pipes "
              "in basements lose 3-5% of DHW energy. Quick payback."
    ),

    # =========================================================================
    # PHASE 2 ECMs - DHW & STORAGE (Added 2025-12-26)
    # =========================================================================

    "heat_pump_water_heater": ECMCost(
        cost_per_unit=15000,
        unit="apartment",
        fixed_cost=30000,
        installation_fraction=0.35,
        lifetime_years=15,
        maintenance_fraction=0.02,
        category=CostCategory.HIGH_COST,
        typical_savings_percent=60,  # DHW portion
        source="Energimyndigheten",
        notes="Frånluftsvärmepump för varmvatten. COP 2.5-3.5. Extracts heat "
              "from exhaust air to preheat DHW. Common in Swedish multi-family."
    ),

    "heat_recovery_dhw": ECMCost(
        cost_per_unit=3000,
        unit="apartment",
        fixed_cost=10000,
        installation_fraction=0.5,
        lifetime_years=20,
        category=CostCategory.MEDIUM_COST,
        typical_savings_percent=30,  # DHW portion
        source="BeBo",
        notes="Spillvattenvärmeväxling. Recovers 30-50% of drain water heat. "
              "Best for new builds or major plumbing renovation."
    ),

    "battery_storage": ECMCost(
        cost_per_unit=6000,
        unit="kWh",
        fixed_cost=15000,
        installation_fraction=0.2,
        lifetime_years=15,
        maintenance_fraction=0.01,
        category=CostCategory.HIGH_COST,
        typical_savings_percent=20,  # Peak shaving + self-consumption
        source="Energimyndigheten",
        notes="Batterilager. Store PV for self-consumption. 10-15 kWh typical "
              "for multi-family common areas. Round-trip efficiency 85-90%."
    ),

    # =========================================================================
    # CONTROLS & MONITORING (Added 2025-12-26)
    # =========================================================================

    "fault_detection": ECMCost(
        cost_per_unit=0,
        unit="building",
        fixed_cost=25000,
        installation_fraction=0,
        lifetime_years=10,
        category=CostCategory.LOW_COST,
        typical_savings_percent=5,
        source="BeBo",
        notes="FDD-system. Automated fault detection identifies stuck valves, "
              "sensor drift, inefficient operation. ROI from reduced maintenance."
    ),

    "energy_monitoring": ECMCost(
        cost_per_unit=500,
        unit="apartment",
        fixed_cost=20000,
        installation_fraction=0.3,
        lifetime_years=15,
        category=CostCategory.LOW_COST,
        typical_savings_percent=8,
        source="Energimyndigheten",
        notes="Energivisualisering. Real-time display of consumption. "
              "Swedish studies show 5-15% behavioral savings."
    ),

    "building_automation_system": ECMCost(
        cost_per_unit=50,
        unit="m² floor",
        fixed_cost=80000,
        installation_fraction=0.4,
        lifetime_years=15,
        category=CostCategory.HIGH_COST,
        typical_savings_percent=12,
        source="BeBo",
        notes="Centralt styrsystem (BAS). Integrates HVAC, lighting, and "
              "monitoring. Enables advanced optimization and remote management."
    ),

    "occupancy_sensors": ECMCost(
        cost_per_unit=2000,
        unit="zone",
        fixed_cost=5000,
        installation_fraction=0.4,
        lifetime_years=10,
        category=CostCategory.LOW_COST,
        typical_savings_percent=5,
        source="Energimyndigheten",
        notes="Närvarodetektorer. Controls lighting and ventilation based on "
              "occupancy. Best in common areas with variable occupancy."
    ),

    "daylight_sensors": ECMCost(
        cost_per_unit=1500,
        unit="zone",
        fixed_cost=3000,
        installation_fraction=0.3,
        lifetime_years=10,
        category=CostCategory.LOW_COST,
        typical_savings_percent=3,
        source="Sveby",
        notes="Dagsljusstyrning. Dims lighting based on available daylight. "
              "Best for south-facing common areas with large windows."
    ),

    "predictive_control": ECMCost(
        cost_per_unit=0,
        unit="building",
        fixed_cost=50000,
        installation_fraction=0,
        lifetime_years=10,
        category=CostCategory.MEDIUM_COST,
        typical_savings_percent=8,
        source="Market research",
        notes="MPC/prediktiv styrning. Uses weather forecast for optimal "
              "heating. Reduces peak loads and improves comfort."
    ),

    # =========================================================================
    # LIGHTING (Added 2025-12-26)
    # =========================================================================

    "led_common_areas": ECMCost(
        cost_per_unit=100,
        unit="m² common area",
        fixed_cost=5000,
        installation_fraction=0.4,
        lifetime_years=15,
        category=CostCategory.LOW_COST,
        typical_savings_percent=60,  # Of lighting, ~3% total
        source="Energimyndigheten",
        notes="LED i trapphus, källare, tvättstuga. Often combined with "
              "occupancy sensors for additional savings."
    ),

    "led_outdoor": ECMCost(
        cost_per_unit=3000,
        unit="fixture",
        fixed_cost=2000,
        installation_fraction=0.5,
        lifetime_years=15,
        category=CostCategory.LOW_COST,
        typical_savings_percent=60,  # Of outdoor lighting
        source="Energimyndigheten",
        notes="LED utomhusbelysning. Parking, entrances, facades. "
              "Often combined with photocell control for dusk-to-dawn."
    ),

    # =========================================================================
    # DHW OPTIMIZATION (Added 2025-12-26)
    # =========================================================================

    "dhw_circulation_optimization": ECMCost(
        cost_per_unit=0,
        unit="building",
        fixed_cost=8000,
        installation_fraction=0,
        lifetime_years=5,
        category=CostCategory.ZERO_COST,
        typical_savings_percent=15,  # Of circulation losses
        source="BeBo",
        notes="VVC-optimering. Timer + temp sensor reduces circulation when "
              "not needed. Simple payback under 1 year."
    ),

    "dhw_tank_insulation": ECMCost(
        cost_per_unit=0,
        unit="tank",
        fixed_cost=5000,
        installation_fraction=0.6,
        lifetime_years=20,
        category=CostCategory.LOW_COST,
        typical_savings_percent=3,  # Of DHW losses
        source="Sveby",
        notes="Isolering av varmvattenberedare. Adds jacket to existing tank. "
              "Reduces standby losses 20-30%."
    ),

    # =========================================================================
    # ENVELOPE - ADDITIONAL (Added 2025-12-26)
    # =========================================================================

    "basement_insulation": ECMCost(
        cost_per_unit=400,
        unit="m² floor",
        fixed_cost=10000,
        installation_fraction=0.5,
        lifetime_years=40,
        category=CostCategory.MEDIUM_COST,
        typical_savings_percent=5,
        source="Wikells",
        notes="Källarisolering (golv eller tak). XPS or EPS insulation. "
              "Best combined with other basement renovation."
    ),

    "entrance_door_replacement": ECMCost(
        cost_per_unit=25000,
        unit="door",
        fixed_cost=5000,
        installation_fraction=0.3,
        lifetime_years=30,
        category=CostCategory.MEDIUM_COST,
        typical_savings_percent=2,
        source="Wikells",
        notes="Entréportbyte. Improves air sealing 8-10%. Often includes "
              "automatic closer and weatherstripping."
    ),

    "thermal_bridge_remediation": ECMCost(
        cost_per_unit=500,
        unit="m²",
        fixed_cost=20000,
        installation_fraction=0.6,
        lifetime_years=40,
        category=CostCategory.HIGH_COST,
        typical_savings_percent=5,
        source="BeBo",
        notes="Köldbryggeåtgärder. Balcony connections, window reveals, "
              "foundation junction. Best combined with facade renovation."
    ),

    "facade_renovation": ECMCost(
        cost_per_unit=2000,
        unit="m² wall",
        fixed_cost=100000,
        installation_fraction=0.45,
        lifetime_years=40,
        category=CostCategory.MAJOR_INVESTMENT,
        typical_savings_percent=25,
        source="BeBo",
        notes="Fasadrenovering komplett. Includes insulation, windows, "
              "thermal bridges. Major intervention but comprehensive."
    ),

    # =========================================================================
    # HVAC - ADDITIONAL (Added 2025-12-26)
    # =========================================================================

    "ftx_overhaul": ECMCost(
        cost_per_unit=80,
        unit="m² floor",
        fixed_cost=25000,
        installation_fraction=0.5,
        lifetime_years=10,
        category=CostCategory.MEDIUM_COST,
        typical_savings_percent=8,
        source="BeBo",
        notes="FTX-renovering. Heat exchanger cleaning, new fans, controls "
              "upgrade. Restores efficiency after 15-20 years."
    ),

    "air_source_heat_pump": ECMCost(
        cost_per_unit=8000,
        unit="kW",
        fixed_cost=50000,
        installation_fraction=0.25,
        lifetime_years=15,
        maintenance_fraction=0.02,
        category=CostCategory.HIGH_COST,
        typical_savings_percent=50,
        source="Energimyndigheten",
        notes="Luft-vattenvärmepump. COP 3-4. Best for replacing oil/gas. "
              "Lower initial cost than ground source but lower COP in winter."
    ),

    "radiator_fans": ECMCost(
        cost_per_unit=800,
        unit="radiator",
        fixed_cost=2000,
        installation_fraction=0.3,
        lifetime_years=10,
        category=CostCategory.LOW_COST,
        typical_savings_percent=5,
        source="Market research",
        notes="Radiatorfläktar. Improves heat transfer, allows 1-2°C lower "
              "supply temp. Adds ~5W electricity per radiator."
    ),

    # =========================================================================
    # OPERATIONAL (Added 2025-12-26)
    # =========================================================================

    "recommissioning": ECMCost(
        cost_per_unit=0,
        unit="building",
        fixed_cost=30000,
        installation_fraction=0,
        lifetime_years=5,
        category=CostCategory.LOW_COST,
        typical_savings_percent=10,
        source="BeBo",
        notes="Funktionskontroll. Systematic check of all building systems. "
              "Often finds 10-15% savings from drift and incorrect settings."
    ),

    "vrf_system": ECMCost(
        cost_per_unit=3500,
        unit="kW",
        fixed_cost=100000,
        installation_fraction=0.35,
        lifetime_years=20,
        category=CostCategory.MAJOR_INVESTMENT,
        typical_savings_percent=40,
        source="Market research",
        notes="VRF-system. Variable Refrigerant Flow for heating/cooling. "
              "Complex installation, not common in Swedish residential."
    ),
}


# =========================================================================
# PACKAGE COST ADJUSTMENTS
# When combining ECMs, some costs reduce due to shared scaffolding, etc.
# =========================================================================

PACKAGE_SYNERGIES: Dict[tuple, float] = {
    # (ecm1, ecm2): cost_multiplier
    # Shared scaffolding
    ("wall_external_insulation", "window_replacement"): 0.90,
    ("wall_external_insulation", "roof_insulation"): 0.95,
    # Shared HVAC contractor
    ("ftx_upgrade", "demand_controlled_ventilation"): 0.85,
    ("ftx_installation", "demand_controlled_ventilation"): 0.80,
    # Shared electrical work
    ("smart_thermostats", "led_lighting"): 0.90,
    # Swedish-specific synergies
    ("district_heating_optimization", "radiator_balancing"): 0.85,
    ("air_sealing", "stairwell_door_sealing"): 0.80,
    ("roof_insulation", "attic_hatch_insulation"): 0.70,
    ("low_flow_fixtures", "pipe_insulation"): 0.90,
    ("ftx_cleaning", "ftx_upgrade"): 0.75,
    # Heat pump combinations
    ("exhaust_air_heat_pump", "solar_pv"): 0.95,
    ("ground_source_heat_pump", "solar_pv"): 0.95,
}


class SwedishCosts:
    """
    Access Swedish cost database.

    Usage:
        costs = SwedishCosts()

        # Get energy price
        elec_price = costs.energy_price('electricity')

        # Get ECM cost
        window_cost = costs.ecm_cost('window_replacement', quantity_m2=150)
    """

    def __init__(
        self,
        energy_prices: Dict[str, EnergyCost] = None,
        ecm_costs: Dict[str, ECMCost] = None
    ):
        self.energy_prices = energy_prices or ENERGY_PRICES
        self.ecm_costs = ecm_costs or ECM_COSTS

    def energy_price(self, energy_type: str) -> EnergyCost:
        """Get energy cost parameters."""
        return self.energy_prices.get(energy_type, ENERGY_PRICES['electricity'])

    def ecm_cost(
        self,
        ecm_id: str,
        quantity: float = 1.0
    ) -> float:
        """
        Calculate total ECM cost.

        Args:
            ecm_id: ECM identifier
            quantity: Quantity in appropriate units (m², kW, kWp)

        Returns:
            Total cost in SEK
        """
        cost_data = self.ecm_costs.get(ecm_id)
        if not cost_data:
            return 0.0

        material_cost = cost_data.cost_per_unit * quantity
        installation_cost = material_cost * cost_data.installation_fraction
        total = cost_data.fixed_cost + material_cost + installation_cost

        return total

    def annual_energy_cost(
        self,
        energy_type: str,
        annual_kwh: float
    ) -> float:
        """Calculate annual energy cost."""
        price = self.energy_price(energy_type)
        return annual_kwh * price.price_sek_per_kwh

    def annual_savings(
        self,
        energy_type: str,
        baseline_kwh: float,
        ecm_kwh: float
    ) -> float:
        """Calculate annual cost savings from ECM."""
        price = self.energy_price(energy_type)
        savings_kwh = baseline_kwh - ecm_kwh
        return savings_kwh * price.price_sek_per_kwh
