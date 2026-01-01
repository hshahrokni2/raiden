"""
ECM Package Generator.

Generates ranked packages (Basic/Standard/Premium) from individual ECM results.
Accounts for interaction effects between measures.

Uses V2 cost database with:
- Regional cost multipliers (Stockholm +18%, etc.)
- Size scaling (economies of scale)
- Accurate energy prices by region
- ROT deduction calculation (for private owners)
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import logging

from ..ecm.catalog import (
    SWEDISH_ECM_CATALOG,
    is_ecm_implemented,
    get_ecm_implementation_status,
    get_unimplemented_ecm_ids,
)
from ..ecm.dependencies import adjust_package_savings, get_package_synergy
from ..roi.costs_sweden_v2 import (
    SwedishCostCalculatorV2,
    ECM_COSTS_V2,
    Region,
    OwnerType,
    DISTRICT_HEATING_PRICES,
    ELECTRICITY_PRICES,
    quick_estimate,
)

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


# Legacy fallback costs - only used if V2 database doesn't have the ECM
ECM_COSTS_PER_M2_FALLBACK = {
    'wall_internal_insulation': 800,
    'wall_external_insulation': 1500,
    'roof_insulation': 400,
    'window_replacement': 600,
    'air_sealing': 150,
    'ftx_upgrade': 200,
    'ftx_installation': 1200,
    'demand_controlled_ventilation': 300,
    'heat_pump_integration': 500,
    'solar_pv': 400,
    'smart_thermostats': 50,
    'led_lighting': 100,
}


def get_ecm_cost_v2(
    ecm_id: str,
    atemp_m2: float,
    region: str = "stockholm",
    owner_type: str = "brf",
    num_floors: int = 4,
    num_apartments: int = 0,
) -> float:
    """
    Get ECM cost using V2 cost database with regional multipliers.

    Args:
        ecm_id: ECM identifier
        atemp_m2: Building floor area in m²
        region: Swedish region (stockholm, gothenburg, malmo, medium_city, rural, norrland)
        owner_type: Owner type - "brf", "private", "rental", "commercial"
        num_floors: Number of floors (for roof area calculation)
        num_apartments: Number of apartments (for per-apartment scaling)

    Returns:
        Total cost in SEK (after any applicable deductions)
    """
    # Estimate apartments if not provided
    if num_apartments <= 0:
        num_apartments = max(1, int(atemp_m2 / 60))  # ~60 m²/apartment

    footprint_m2 = atemp_m2 / num_floors

    # Check if ECM exists in V2 database
    if ecm_id in ECM_COSTS_V2:
        model = ECM_COSTS_V2[ecm_id]
        scales_with = model.scales_with

        # Determine quantity based on scales_with attribute
        if scales_with == "floor_area":
            quantity = atemp_m2
        elif scales_with == "wall_area":
            # Wall area ≈ 50% of Atemp for typical multi-family
            quantity = atemp_m2 * 0.5
        elif scales_with == "roof_area":
            quantity = footprint_m2
        elif scales_with == "window_area":
            # Window area ≈ 15% of Atemp (typical WWR)
            quantity = atemp_m2 * 0.15
        elif scales_with in ("unit", "per_building"):
            quantity = 1
        elif scales_with == "per_apartment":
            quantity = num_apartments
        elif scales_with == "capacity":
            # For heat pumps, solar, etc: estimate kW based on size
            if "solar" in ecm_id or "pv" in ecm_id:
                quantity = footprint_m2 * 0.15  # ~15% of roof for solar
            else:
                quantity = atemp_m2 * 0.04  # ~40 W/m² heating capacity
        else:
            # Default to floor area
            quantity = atemp_m2

        # Use V2 quick_estimate for accurate cost
        cost = quick_estimate(
            ecm_id=ecm_id,
            quantity=quantity,
            region=region,
            floor_area_m2=atemp_m2,
            owner_type=owner_type,
        )
        if cost > 0:
            return cost

    # Fallback to legacy per-m² costs
    cost_per_m2 = ECM_COSTS_PER_M2_FALLBACK.get(ecm_id, 300)
    return cost_per_m2 * atemp_m2


def get_energy_price(
    region: str = "stockholm",
    heating_type: str = "district_heating",
) -> float:
    """
    Get energy price for region and heating type.

    Args:
        region: Swedish region or city
        heating_type: district_heating, electricity, heat_pump

    Returns:
        Energy price in SEK/kWh
    """
    # Map region strings to price keys
    region_key = region.lower().replace(" ", "_")
    if "stockholm" in region_key:
        region_key = "stockholm"
    elif "göteborg" in region_key or "gothenburg" in region_key:
        region_key = "gothenburg"
    elif "malmö" in region_key or "malmo" in region_key:
        region_key = "malmo"
    elif any(x in region_key for x in ["norra", "norrland", "luleå", "umeå"]):
        region_key = "se1"  # Northern Sweden - cheap electricity
    else:
        region_key = "medium_city"

    if heating_type == "district_heating":
        prices = DISTRICT_HEATING_PRICES
        price_entry = prices.get(region_key, prices.get("medium_city"))
        return price_entry.total_price_sek_kwh
    elif heating_type in ["electricity", "electric", "heat_pump"]:
        # Use SE3 for Stockholm, SE4 for Malmö, else average
        if region_key == "stockholm":
            price_entry = ELECTRICITY_PRICES.get("se3")
        elif region_key == "malmo":
            price_entry = ELECTRICITY_PRICES.get("se4")
        elif region_key == "se1":
            price_entry = ELECTRICITY_PRICES.get("se1")
        else:
            price_entry = ELECTRICITY_PRICES.get("average")
        return price_entry.total_price_sek_kwh
    else:
        # Blended default
        return 1.20


# CO2 intensity (kg CO2 / kWh, Swedish grid - very clean)
CO2_INTENSITY_KG_KWH = 0.02  # Updated to reflect Swedish mix (hydro/nuclear)


class PackageGenerator:
    """
    Generate ECM packages from individual results.

    Packages:
    - Basic: Top 2 ECMs by ROI (quick wins)
    - Standard: Top 4 ECMs (balanced)
    - Premium: All cost-effective ECMs

    Uses V2 cost database with regional pricing and size scaling.
    """

    def __init__(
        self,
        region: str = "stockholm",
        owner_type: str = "brf",
        heating_type: str = "district_heating",
        energy_price: Optional[float] = None,
        interaction_factor: float = 0.70,
    ):
        """
        Initialize package generator with V2 cost database.

        Args:
            region: Swedish region (stockholm, gothenburg, malmo, medium_city, rural, norrland)
            owner_type: Owner type - "brf", "private", "rental", "commercial"
            heating_type: Heating type for energy price (district_heating, electricity)
            energy_price: Override energy price in SEK/kWh (uses regional price if None)
            interaction_factor: Multiplier for combined savings (accounts for diminishing returns)
        """
        self.region = region
        self.owner_type = owner_type
        self.heating_type = heating_type
        # Use regional energy price if not overridden
        self.energy_price = energy_price or get_energy_price(region, heating_type)
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
        # CRITICAL: Exclude no-op ECMs that have no thermal simulation effect
        positive_ecms = []
        excluded_noop_ecms = []

        for ecm in ecm_results:
            savings_pct = ecm.get('savings_percent', 0)
            ecm_id = ecm.get('id', ecm.get('ecm_id', 'unknown'))

            # Check if ECM is actually implemented in IDF modifier
            if not is_ecm_implemented(ecm_id):
                impl_status = get_ecm_implementation_status(ecm_id)
                excluded_noop_ecms.append((ecm_id, impl_status))
                logger.warning(
                    f"Excluding ECM '{ecm_id}' from packages - "
                    f"IDF implementation: {impl_status} (no thermal effect)"
                )
                continue

            if savings_pct > 1:  # At least 1% savings to be included
                # Use V2 cost database with regional multipliers
                total_cost = get_ecm_cost_v2(
                    ecm_id=ecm_id,
                    atemp_m2=atemp_m2,
                    region=self.region,
                    owner_type=self.owner_type,
                )
                cost_per_m2 = total_cost / atemp_m2 if atemp_m2 > 0 else 0
                savings_kwh_m2 = baseline_kwh_m2 * (savings_pct / 100)
                annual_savings_sek = savings_kwh_m2 * atemp_m2 * self.energy_price
                payback = total_cost / annual_savings_sek if annual_savings_sek > 0 else 999

                positive_ecms.append({
                    **ecm,
                    'cost_per_m2': cost_per_m2,
                    'total_cost_sek': total_cost,
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
        """Create a package from a list of ECMs using V2 costs."""
        # Get ECM IDs for synergy calculation
        ecm_ids = [e.get('id', e.get('ecm_id', 'unknown')) for e in ecms]

        # Calculate synergy factor from dependency matrix
        synergy_factor = get_package_synergy(ecm_ids)

        # Calculate combined savings with synergy factor
        individual_savings_sum = sum(e['savings_percent'] for e in ecms)
        combined_savings_pct = individual_savings_sum * synergy_factor

        # Cap at realistic maximum
        combined_savings_pct = min(combined_savings_pct, 60)

        combined_savings_kwh_m2 = baseline_kwh_m2 * (combined_savings_pct / 100)

        # Total cost - use pre-calculated V2 costs
        total_cost = sum(e.get('total_cost_sek', e['cost_per_m2'] * atemp_m2) for e in ecms)

        # Annual savings
        annual_savings = combined_savings_kwh_m2 * atemp_m2 * self.energy_price

        # Simple payback
        payback = total_cost / annual_savings if annual_savings > 0 else 999

        # CO2 reduction
        co2_reduction = combined_savings_kwh_m2 * CO2_INTENSITY_KG_KWH

        # Build package items with V2 costs
        items = [
            ECMPackageItem(
                id=e.get('id', e.get('ecm_id', 'unknown')),
                name=e.get('name', e.get('ecm_name', 'Unknown')),
                individual_savings_percent=e['savings_percent'],
                estimated_cost_sek=e.get('total_cost_sek', e['cost_per_m2'] * atemp_m2),
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
    region: str = "stockholm",
    owner_type: str = "brf",
    heating_type: str = "district_heating",
) -> List[ECMPackage]:
    """
    Convenience function to generate packages with V2 costs.

    Args:
        ecm_results: List of ECM results with 'id', 'name', 'savings_percent'
        baseline_kwh_m2: Baseline heating in kWh/m²
        atemp_m2: Building area in m²
        region: Swedish region (stockholm, gothenburg, malmo, medium_city, rural)
        owner_type: Owner type - "brf", "private", "rental", "commercial"
        heating_type: Heating type for energy pricing

    Returns:
        List of ECMPackage objects with regional cost estimates
    """
    generator = PackageGenerator(
        region=region,
        owner_type=owner_type,
        heating_type=heating_type,
    )
    return generator.generate_packages(ecm_results, baseline_kwh_m2, atemp_m2)


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================
# These maintain compatibility with code using the old V1 API

# Default energy price for backward compatibility (Stockholm district heating)
ENERGY_PRICE_SEK_KWH = 0.85  # Stockholm district heating price


def get_ecm_cost_per_m2(ecm_id: str, atemp_m2: float = 1000) -> float:
    """
    Backward-compatible wrapper for V1 API.

    Returns cost per m² (not total cost like get_ecm_cost_v2).
    Uses Stockholm region and BRF owner type as defaults.

    Args:
        ecm_id: ECM identifier
        atemp_m2: Building floor area (default 1000 for per-m² calculation)

    Returns:
        Cost per m² in SEK
    """
    total_cost = get_ecm_cost_v2(
        ecm_id=ecm_id,
        atemp_m2=atemp_m2,
        region="stockholm",
        owner_type="brf",
    )
    return total_cost / atemp_m2 if atemp_m2 > 0 else 0
