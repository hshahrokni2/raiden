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
