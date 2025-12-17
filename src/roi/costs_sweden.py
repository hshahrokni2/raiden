"""
Swedish Cost Database - ECM costs and energy prices.

Based on:
- Swedish construction cost indices (SCB)
- Energimyndigheten energy price statistics
- Industry quotes and studies

Prices in SEK, 2024 levels.
"""

from dataclasses import dataclass
from typing import Dict


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


# Swedish ECM costs (2024)
ECM_COSTS: Dict[str, ECMCost] = {
    # Envelope
    "wall_external_insulation": ECMCost(
        cost_per_unit=1500,  # SEK per m² wall
        unit="m² wall",
        installation_fraction=0.4,
        lifetime_years=40
    ),
    "wall_internal_insulation": ECMCost(
        cost_per_unit=800,
        unit="m² wall",
        installation_fraction=0.5,  # More labor intensive
        lifetime_years=40
    ),
    "roof_insulation": ECMCost(
        cost_per_unit=400,
        unit="m² roof",
        installation_fraction=0.3,
        lifetime_years=40
    ),
    "window_replacement": ECMCost(
        cost_per_unit=6000,
        unit="m² window",
        installation_fraction=0.35,
        lifetime_years=30
    ),
    "air_sealing": ECMCost(
        cost_per_unit=50,
        unit="m² floor",
        installation_fraction=0.7,  # Mostly labor
        lifetime_years=20
    ),

    # HVAC
    "ftx_upgrade": ECMCost(
        cost_per_unit=200,
        unit="m² floor",
        installation_fraction=0.4,
        lifetime_years=20
    ),
    "ftx_installation": ECMCost(
        cost_per_unit=400,
        unit="m² floor",
        installation_fraction=0.5,
        lifetime_years=25
    ),
    "demand_controlled_ventilation": ECMCost(
        cost_per_unit=150,
        unit="m² floor",
        installation_fraction=0.3,
        lifetime_years=15
    ),
    "heat_pump_integration": ECMCost(
        cost_per_unit=3000,
        unit="kW",
        fixed_cost=50000,  # Fixed installation cost
        installation_fraction=0.25,
        lifetime_years=20
    ),

    # Renewable
    "solar_pv": ECMCost(
        cost_per_unit=12000,
        unit="kWp",
        fixed_cost=20000,  # Fixed cost for small systems
        installation_fraction=0.3,
        lifetime_years=25,
        maintenance_fraction=0.005  # Low maintenance
    ),

    # Controls
    "smart_thermostats": ECMCost(
        cost_per_unit=30,
        unit="m² floor",
        installation_fraction=0.5,
        lifetime_years=10
    ),
    "led_lighting": ECMCost(
        cost_per_unit=100,
        unit="m² floor",
        installation_fraction=0.4,
        lifetime_years=15
    ),
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
