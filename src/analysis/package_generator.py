"""
ECM Package Generator.

Generates ranked packages (Basic/Standard/Premium) from individual ECM results.
Accounts for interaction effects between measures.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class ECMPackageItem:
    """Single ECM in a package."""
    id: str
    name: str
    individual_savings_percent: float
    estimated_cost_sek: float


@dataclass
class ECMPackage:
    """A package of combined ECMs."""
    id: str
    name: str
    description: str
    ecms: List[ECMPackageItem]
    combined_savings_percent: float
    combined_savings_kwh_m2: float
    total_cost_sek: float
    simple_payback_years: float
    annual_cost_savings_sek: float
    co2_reduction_kg_m2: float


# Swedish ECM cost estimates (SEK per m² Atemp, 2024 prices)
ECM_COSTS_PER_M2 = {
    'wall_internal_insulation': 800,    # Interior insulation
    'wall_external_insulation': 1500,   # ETICS system
    'roof_insulation': 400,             # Attic insulation
    'window_replacement': 600,          # New windows
    'air_sealing': 150,                 # Air sealing measures
    'ftx_upgrade': 200,                 # Upgrade existing FTX
    'ftx_installation': 1200,           # New FTX system
    'demand_controlled_ventilation': 300,  # DCV sensors and controls
    'heat_pump_integration': 500,       # Integration with existing
    'solar_pv': 400,                    # Per m² roof
    'smart_thermostats': 50,            # Smart controls
    'led_lighting': 100,                # LED retrofit
}

# Swedish energy prices (SEK/kWh, 2024)
ENERGY_PRICE_SEK_KWH = 1.50  # Blended district heating / electricity

# CO2 intensity (kg CO2 / kWh, Swedish grid)
CO2_INTENSITY_KG_KWH = 0.05  # Very low due to hydro/nuclear


class PackageGenerator:
    """
    Generate ECM packages from individual results.

    Packages:
    - Basic: Top 2 ECMs by ROI (quick wins)
    - Standard: Top 4 ECMs (balanced)
    - Premium: All cost-effective ECMs
    """

    def __init__(
        self,
        energy_price: float = ENERGY_PRICE_SEK_KWH,
        interaction_factor: float = 0.70,
    ):
        """
        Initialize package generator.

        Args:
            energy_price: Energy price in SEK/kWh
            interaction_factor: Multiplier for combined savings (accounts for diminishing returns)
        """
        self.energy_price = energy_price
        self.interaction_factor = interaction_factor

    def generate_packages(
        self,
        ecm_results: List[Dict],
        baseline_kwh_m2: float,
        atemp_m2: float,
    ) -> List[ECMPackage]:
        """
        Generate Basic/Standard/Premium packages from ECM results.

        Args:
            ecm_results: List of ECM results with savings info
            baseline_kwh_m2: Baseline heating in kWh/m²
            atemp_m2: Building area in m²

        Returns:
            List of ECMPackage objects (Basic, Standard, Premium)
        """
        # Filter to ECMs with positive savings and sort by ROI
        # Exclude ECMs that increase heating (e.g., LED in cold climates)
        positive_ecms = []
        for ecm in ecm_results:
            savings_pct = ecm.get('savings_percent', 0)
            if savings_pct > 1:  # At least 1% savings to be included
                ecm_id = ecm.get('id', ecm.get('ecm_id', 'unknown'))
                cost_per_m2 = ECM_COSTS_PER_M2.get(ecm_id, 300)
                savings_kwh_m2 = baseline_kwh_m2 * (savings_pct / 100)
                annual_savings_sek_m2 = savings_kwh_m2 * self.energy_price
                payback = cost_per_m2 / annual_savings_sek_m2 if annual_savings_sek_m2 > 0 else 999

                positive_ecms.append({
                    **ecm,
                    'cost_per_m2': cost_per_m2,
                    'payback_years': payback,
                    'savings_kwh_m2': savings_kwh_m2,
                })

        # Sort by payback (lowest first = best ROI)
        sorted_ecms = sorted(positive_ecms, key=lambda x: x['payback_years'])

        packages = []

        # Basic Package: Top 2 ECMs
        if len(sorted_ecms) >= 2:
            basic = self._create_package(
                package_id='basic',
                name='Grundpaket',
                description='Snabba vinster med kort återbetalningstid',
                ecms=sorted_ecms[:2],
                baseline_kwh_m2=baseline_kwh_m2,
                atemp_m2=atemp_m2,
            )
            packages.append(basic)

        # Standard Package: Top 4 ECMs
        if len(sorted_ecms) >= 4:
            standard = self._create_package(
                package_id='standard',
                name='Standardpaket',
                description='Balanserad energibesparing',
                ecms=sorted_ecms[:4],
                baseline_kwh_m2=baseline_kwh_m2,
                atemp_m2=atemp_m2,
            )
            packages.append(standard)

        # Premium Package: All cost-effective ECMs (payback < 15 years)
        cost_effective = [e for e in sorted_ecms if e['payback_years'] < 15]
        if len(cost_effective) >= 3:
            premium = self._create_package(
                package_id='premium',
                name='Premiumpaket',
                description='Maximal energibesparing',
                ecms=cost_effective,
                baseline_kwh_m2=baseline_kwh_m2,
                atemp_m2=atemp_m2,
            )
            packages.append(premium)

        return packages

    def _create_package(
        self,
        package_id: str,
        name: str,
        description: str,
        ecms: List[Dict],
        baseline_kwh_m2: float,
        atemp_m2: float,
    ) -> ECMPackage:
        """Create a package from a list of ECMs."""
        # Calculate combined savings with interaction factor
        individual_savings_sum = sum(e['savings_percent'] for e in ecms)
        combined_savings_pct = individual_savings_sum * self.interaction_factor

        # Cap at realistic maximum
        combined_savings_pct = min(combined_savings_pct, 50)

        combined_savings_kwh_m2 = baseline_kwh_m2 * (combined_savings_pct / 100)

        # Total cost
        total_cost = sum(e['cost_per_m2'] * atemp_m2 for e in ecms)

        # Annual savings
        annual_savings = combined_savings_kwh_m2 * atemp_m2 * self.energy_price

        # Simple payback
        payback = total_cost / annual_savings if annual_savings > 0 else 999

        # CO2 reduction
        co2_reduction = combined_savings_kwh_m2 * CO2_INTENSITY_KG_KWH

        # Build package items
        items = [
            ECMPackageItem(
                id=e.get('id', e.get('ecm_id', 'unknown')),
                name=e.get('name', e.get('ecm_name', 'Unknown')),
                individual_savings_percent=e['savings_percent'],
                estimated_cost_sek=e['cost_per_m2'] * atemp_m2,
            )
            for e in ecms
        ]

        return ECMPackage(
            id=package_id,
            name=name,
            description=description,
            ecms=items,
            combined_savings_percent=combined_savings_pct,
            combined_savings_kwh_m2=combined_savings_kwh_m2,
            total_cost_sek=total_cost,
            simple_payback_years=payback,
            annual_cost_savings_sek=annual_savings,
            co2_reduction_kg_m2=co2_reduction,
        )


def generate_packages(
    ecm_results: List[Dict],
    baseline_kwh_m2: float,
    atemp_m2: float,
) -> List[ECMPackage]:
    """
    Convenience function to generate packages.

    Args:
        ecm_results: List of ECM results with 'id', 'name', 'savings_percent'
        baseline_kwh_m2: Baseline heating in kWh/m²
        atemp_m2: Building area in m²

    Returns:
        List of ECMPackage objects
    """
    generator = PackageGenerator()
    return generator.generate_packages(ecm_results, baseline_kwh_m2, atemp_m2)
