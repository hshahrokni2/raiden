"""
Swedish Cost Database V2 - Production-grade cost modeling.

Improvements over V1:
- Separate material/labor costs for ROT calculation
- Source tracking with confidence levels
- Automatic inflation adjustment
- Swedish tax deductions (ROT, green tech, grants)
- Building size scaling factors
- Regional cost adjustments (Stockholm premium)
- ECM dependency/synergy handling

Sources:
- Wikells Sektionsfakta 2024 (industry standard)
- BeBo Lönsamhetskalkyl 2023 (multi-family retrofit)
- BeBo Typkostnader 2023 (detailed cost breakdowns)
- Energimyndigheten (heat pumps, solar)
- SCB Byggkostnadsindex (inflation)

Prices in SEK, base year as specified per entry.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
from datetime import date
import math
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class CostSource(Enum):
    """Source of cost data for traceability."""
    WIKELLS_2024 = "wikells_sektionsfakta_2024"
    BEBO_LONSAMHET_2023 = "bebo_lonsamhetskalkyl_2023"
    BEBO_TYPKOSTNADER_2023 = "bebo_typkostnader_2023"
    ENERGIMYNDIGHETEN_2024 = "energimyndigheten_2024"
    SABO_2024 = "sabo_2024"
    SVEBY_2023 = "sveby_2023"
    MARKET_RESEARCH_2025 = "market_research_2025"
    SCB_BKI = "scb_byggkostnadsindex"
    USER_INPUT = "user_input"
    ESTIMATED = "estimated"


class CostCategory(Enum):
    """Cost magnitude categories."""
    ZERO_COST = "zero_cost"       # Operational optimization only
    LOW_COST = "low_cost"         # < 100 SEK/m²
    MEDIUM_COST = "medium_cost"   # 100-500 SEK/m²
    HIGH_COST = "high_cost"       # 500-1500 SEK/m²
    MAJOR = "major"               # > 1500 SEK/m²


class Region(Enum):
    """Swedish regions with cost multipliers."""
    STOCKHOLM = "stockholm"       # +15-20% premium
    GOTHENBURG = "gothenburg"     # +5-10%
    MALMO = "malmo"               # +5%
    MEDIUM_CITY = "medium_city"   # Base
    RURAL = "rural"               # -5-10%
    NORRLAND = "norrland"         # +10-15% (logistics)


class OwnerType(Enum):
    """Building owner type - affects tax deduction eligibility."""
    PRIVATE = "private"           # Villa, bostadsrätt - ROT eligible
    BRF = "brf"                   # Bostadsrättsförening - NO ROT (building owner)
    RENTAL = "rental"             # Hyresfastighet - NO ROT
    COMMERCIAL = "commercial"     # Kommersiell fastighet - NO ROT


# Regional cost multipliers relative to medium city baseline
REGIONAL_MULTIPLIERS: Dict[Region, float] = {
    Region.STOCKHOLM: 1.18,
    Region.GOTHENBURG: 1.08,
    Region.MALMO: 1.05,
    Region.MEDIUM_CITY: 1.00,
    Region.RURAL: 0.92,
    Region.NORRLAND: 1.12,
}


# Building size scaling (economies of scale)
# Larger buildings get better unit prices
def size_scaling_factor(floor_area_m2: float) -> float:
    """
    Calculate size-based cost scaling.

    Baseline: 1000 m² = 1.0
    Smaller buildings: premium (more overhead per m²)
    Larger buildings: discount (volume pricing)
    """
    if floor_area_m2 <= 0:
        return 1.0

    # Log scaling: doubling area reduces unit cost by ~10%
    baseline = 1000
    factor = 1.0 - 0.1 * math.log2(floor_area_m2 / baseline)

    # Clamp to reasonable range [0.7, 1.4]
    return max(0.7, min(1.4, factor))


# Inflation rates from SCB Byggkostnadsindex
ANNUAL_INFLATION_RATE = 0.04  # 4% average construction cost inflation


# =============================================================================
# ENERGY PRICES (for ROI calculations)
# =============================================================================

@dataclass
class EnergyPrice:
    """Energy price with regional variations."""
    base_price_sek_kwh: float  # Base price
    network_fee_sek_kwh: float = 0  # Grid fee (electricity)
    tax_sek_kwh: float = 0  # Energy tax
    annual_escalation: float = 0.02  # Expected annual increase
    co2_kg_per_kwh: float = 0.05  # Carbon intensity

    @property
    def total_price_sek_kwh(self) -> float:
        """Total price including all components."""
        return self.base_price_sek_kwh + self.network_fee_sek_kwh + self.tax_sek_kwh


# District heating prices vary significantly by city/provider
DISTRICT_HEATING_PRICES: Dict[str, EnergyPrice] = {
    "stockholm": EnergyPrice(  # Fortum/Stockholm Exergi
        base_price_sek_kwh=0.85,
        annual_escalation=0.03,
        co2_kg_per_kwh=0.04,
    ),
    "gothenburg": EnergyPrice(  # Göteborg Energi
        base_price_sek_kwh=0.75,
        annual_escalation=0.02,
        co2_kg_per_kwh=0.06,
    ),
    "malmo": EnergyPrice(  # E.ON
        base_price_sek_kwh=0.80,
        annual_escalation=0.02,
        co2_kg_per_kwh=0.05,
    ),
    "medium_city": EnergyPrice(  # Average medium city
        base_price_sek_kwh=0.70,
        annual_escalation=0.02,
        co2_kg_per_kwh=0.05,
    ),
    "rural": EnergyPrice(  # Smaller providers
        base_price_sek_kwh=0.65,
        annual_escalation=0.02,
        co2_kg_per_kwh=0.06,
    ),
}

# Electricity prices (2025 estimates, including all components)
ELECTRICITY_PRICES: Dict[str, EnergyPrice] = {
    "se1": EnergyPrice(  # Norra Sverige - lowest
        base_price_sek_kwh=0.50,
        network_fee_sek_kwh=0.35,
        tax_sek_kwh=0.36,  # Energiskatt
        annual_escalation=0.03,
        co2_kg_per_kwh=0.02,  # Swedish grid very clean
    ),
    "se2": EnergyPrice(  # Mellannorra
        base_price_sek_kwh=0.55,
        network_fee_sek_kwh=0.35,
        tax_sek_kwh=0.36,
        annual_escalation=0.03,
        co2_kg_per_kwh=0.02,
    ),
    "se3": EnergyPrice(  # Stockholm/Mellansverige
        base_price_sek_kwh=0.70,
        network_fee_sek_kwh=0.40,
        tax_sek_kwh=0.36,
        annual_escalation=0.03,
        co2_kg_per_kwh=0.02,
    ),
    "se4": EnergyPrice(  # Södra Sverige - highest
        base_price_sek_kwh=0.80,
        network_fee_sek_kwh=0.40,
        tax_sek_kwh=0.36,
        annual_escalation=0.03,
        co2_kg_per_kwh=0.03,
    ),
    "average": EnergyPrice(  # National average
        base_price_sek_kwh=0.65,
        network_fee_sek_kwh=0.38,
        tax_sek_kwh=0.36,
        annual_escalation=0.03,
        co2_kg_per_kwh=0.02,
    ),
}

# Other energy sources
OTHER_ENERGY_PRICES: Dict[str, EnergyPrice] = {
    "natural_gas": EnergyPrice(
        base_price_sek_kwh=1.20,
        annual_escalation=0.02,
        co2_kg_per_kwh=0.20,
    ),
    "oil": EnergyPrice(
        base_price_sek_kwh=1.40,
        annual_escalation=0.02,
        co2_kg_per_kwh=0.27,
    ),
    "pellets": EnergyPrice(
        base_price_sek_kwh=0.55,
        annual_escalation=0.02,
        co2_kg_per_kwh=0.02,
    ),
}


def get_energy_price(
    energy_type: str,
    region: str = "medium_city"
) -> EnergyPrice:
    """Get energy price for a specific type and region."""
    if energy_type == "district_heating":
        return DISTRICT_HEATING_PRICES.get(region, DISTRICT_HEATING_PRICES["medium_city"])
    elif energy_type == "electricity":
        # Map region to price zone
        zone_map = {
            "stockholm": "se3",
            "gothenburg": "se3",
            "malmo": "se4",
            "medium_city": "average",
            "rural": "se2",
            "norrland": "se1",
        }
        zone = zone_map.get(region, "average")
        return ELECTRICITY_PRICES.get(zone, ELECTRICITY_PRICES["average"])
    else:
        return OTHER_ENERGY_PRICES.get(energy_type, ELECTRICITY_PRICES["average"])


# =============================================================================
# EFFEKTTARIFF (Power Demand Tariff) - New from 2025
# =============================================================================
# Based on Ellevio's new tariff structure effective 2025-01-01
# ~45% of grid fee now comes from peak power demand, not energy consumption

@dataclass
class EffektTariff:
    """
    Swedish power demand tariff structure (effekttariff).

    Based on Ellevio 2025 model, now standard across most Swedish grid operators.
    Peak demand is measured as average of 3 highest hourly peaks per month.
    """
    # Peak demand charges (SEK/kW/month)
    peak_charge_day_sek_kw: float = 81.25    # 06:00-22:00 all days
    peak_charge_night_sek_kw: float = 40.63  # 22:00-06:00 (half price)

    # Fixed monthly fee (replaces old säkringsavgift)
    fixed_fee_sek_month: float = 200.0       # Typical for BRF/företag

    # Energy transfer fee (much smaller now, ~5% of bill)
    transfer_fee_sek_kwh: float = 0.05       # Överföringsavgift

    # Calculation parameters
    peaks_per_month: int = 3                  # Average of top 3 peaks
    one_peak_per_day: bool = True            # Only count highest per day

    # Future: 15-minute intervals from Sept 2025
    interval_minutes: int = 60               # Currently hourly, 15 from 2025-09

    def calculate_monthly_effektavgift(
        self,
        peak_kw: float,
        is_daytime: bool = True
    ) -> float:
        """Calculate monthly power demand charge."""
        rate = self.peak_charge_day_sek_kw if is_daytime else self.peak_charge_night_sek_kw
        return peak_kw * rate

    def calculate_annual_effektavgift(
        self,
        winter_peak_kw: float,
        summer_peak_kw: float,
        winter_months: int = 6,
        summer_months: int = 6
    ) -> float:
        """
        Calculate annual power demand charge with seasonal variation.

        Args:
            winter_peak_kw: Peak demand Oct-Mar (heating season)
            summer_peak_kw: Peak demand Apr-Sep (cooling/base load)

        Returns:
            Annual effektavgift in SEK
        """
        winter_charge = winter_peak_kw * self.peak_charge_day_sek_kw * winter_months
        summer_charge = summer_peak_kw * self.peak_charge_day_sek_kw * summer_months
        return winter_charge + summer_charge


# Default tariff (Ellevio 2025)
ELLEVIO_EFFEKTTARIFF = EffektTariff(
    peak_charge_day_sek_kw=81.25,
    peak_charge_night_sek_kw=40.63,
    fixed_fee_sek_month=200.0,
    transfer_fee_sek_kwh=0.05,
)

# Regional variations (some operators have different rates)
EFFEKT_TARIFFS: Dict[str, EffektTariff] = {
    "ellevio": ELLEVIO_EFFEKTTARIFF,
    "vattenfall": EffektTariff(
        peak_charge_day_sek_kw=75.00,
        peak_charge_night_sek_kw=37.50,
        fixed_fee_sek_month=180.0,
    ),
    "eon": EffektTariff(
        peak_charge_day_sek_kw=78.00,
        peak_charge_night_sek_kw=39.00,
        fixed_fee_sek_month=190.0,
    ),
    "goteborg_energi": EffektTariff(
        peak_charge_day_sek_kw=0.0,  # Free during summer!
        peak_charge_night_sek_kw=0.0,
        fixed_fee_sek_month=250.0,
        # Note: Göteborg has 0 kr effekttariff Apr-Sep
    ),
    "default": ELLEVIO_EFFEKTTARIFF,
}


def get_effekt_tariff(grid_operator: str = "ellevio") -> EffektTariff:
    """Get effekt tariff for a specific grid operator."""
    return EFFEKT_TARIFFS.get(grid_operator.lower(), ELLEVIO_EFFEKTTARIFF)


# =============================================================================
# BUILDING PEAK POWER ESTIMATION
# =============================================================================
# Estimate peak electrical demand based on building characteristics

@dataclass
class BuildingPeakEstimate:
    """Estimated peak power demand for a building."""
    total_peak_kw: float
    winter_peak_kw: float
    summer_peak_kw: float

    # Component breakdown
    heat_pump_kw: float = 0.0
    ventilation_kw: float = 0.0
    lighting_kw: float = 0.0
    elevators_kw: float = 0.0
    common_equipment_kw: float = 0.0
    ev_charging_kw: float = 0.0

    # Diversity factors applied
    diversity_factor: float = 0.7  # Not all loads peak simultaneously


def estimate_building_peak_power(
    atemp_m2: float,
    num_floors: int = 5,
    num_apartments: int = 0,
    has_heat_pump: bool = False,
    heat_pump_capacity_kw: float = 0,
    has_ev_charging: bool = False,
    num_ev_chargers: int = 0,
    ev_charger_kw: float = 11.0,
    has_elevator: bool = True,
    num_elevators: int = 2,
) -> BuildingPeakEstimate:
    """
    Estimate peak electrical demand for a Swedish multi-family building.

    Based on typical Swedish load profiles for flerbostadshus.

    Args:
        atemp_m2: Heated floor area
        num_floors: Number of floors (affects elevator/ventilation)
        num_apartments: Number of apartments (affects diversity)
        has_heat_pump: Building heated by heat pump
        heat_pump_capacity_kw: If known, else estimated
        has_ev_charging: EV charging infrastructure
        num_ev_chargers: Number of charging points
        has_elevator: Building has elevator(s)
        num_elevators: Number of elevators

    Returns:
        BuildingPeakEstimate with component breakdown
    """
    # Ventilation (FTX fans): ~0.5-1.0 W/m² for modern FTX
    ventilation_kw = atemp_m2 * 0.8 / 1000  # 0.8 W/m² average

    # Common area lighting: ~2-5 W/m² of common areas (~15% of Atemp)
    common_area_m2 = atemp_m2 * 0.15
    lighting_kw = common_area_m2 * 4.0 / 1000  # 4 W/m² for older lighting

    # Elevators: ~15-30 kW each during operation
    elevator_kw_each = 20.0 if num_floors > 5 else 15.0
    elevators_kw = num_elevators * elevator_kw_each if has_elevator else 0

    # Common equipment (pumps, garage ventilation, etc.): ~0.5 W/m²
    common_equipment_kw = atemp_m2 * 0.5 / 1000

    # Heat pump sizing (if present)
    if has_heat_pump:
        if heat_pump_capacity_kw > 0:
            heat_pump_kw = heat_pump_capacity_kw
        else:
            # Estimate: ~30-50 W/m² heating capacity, COP 3.5 → electrical
            heating_capacity_kw = atemp_m2 * 40 / 1000  # 40 W/m²
            heat_pump_kw = heating_capacity_kw / 3.5  # Electrical input
    else:
        heat_pump_kw = 0

    # EV charging (worst case: all charging simultaneously)
    ev_charging_kw = 0
    if has_ev_charging:
        # With load balancing, assume 30% simultaneous usage
        ev_charging_kw = num_ev_chargers * ev_charger_kw * 0.3

    # Sum components (before diversity)
    total_connected_kw = (
        heat_pump_kw +
        ventilation_kw +
        lighting_kw +
        elevators_kw +
        common_equipment_kw +
        ev_charging_kw
    )

    # Apply diversity factor (not all loads peak simultaneously)
    # Higher for larger buildings (more averaging)
    if num_apartments > 50:
        diversity_factor = 0.65
    elif num_apartments > 20:
        diversity_factor = 0.70
    else:
        diversity_factor = 0.75

    total_peak_kw = total_connected_kw * diversity_factor

    # Winter peak: full heating + all other loads
    winter_peak_kw = total_peak_kw

    # Summer peak: no heating, but add some cooling if applicable
    summer_factor = 0.4 if has_heat_pump else 0.5  # Less load without heating
    summer_peak_kw = (total_peak_kw - heat_pump_kw * 0.7) * summer_factor + heat_pump_kw * 0.2
    summer_peak_kw = max(summer_peak_kw, total_peak_kw * 0.3)  # Minimum 30% of winter

    return BuildingPeakEstimate(
        total_peak_kw=total_peak_kw,
        winter_peak_kw=winter_peak_kw,
        summer_peak_kw=summer_peak_kw,
        heat_pump_kw=heat_pump_kw,
        ventilation_kw=ventilation_kw,
        lighting_kw=lighting_kw,
        elevators_kw=elevators_kw,
        common_equipment_kw=common_equipment_kw,
        ev_charging_kw=ev_charging_kw,
        diversity_factor=diversity_factor,
    )


# =============================================================================
# ECM PEAK POWER IMPACT
# =============================================================================
# How each ECM affects peak electrical demand (for effektavgift calculation)

@dataclass
class ECMPeakImpact:
    """Impact of an ECM on peak electrical demand."""
    peak_reduction_kw: float = 0.0      # Direct reduction in kW
    peak_reduction_percent: float = 0.0  # Percentage reduction
    affects_winter_only: bool = False    # Only reduces winter peak
    affects_summer_only: bool = False    # Only reduces summer peak
    description: str = ""


# Peak impact by ECM (relative to building characteristics)
ECM_PEAK_IMPACTS: Dict[str, Dict] = {
    # Effektvakt - direct peak shaving
    "effektvakt_optimization": {
        "peak_reduction_percent": 15.0,  # 15-25% peak reduction
        "affects_winter_only": True,
        "description": "Reduces peak by pre-heating and thermal coasting during peak hours",
    },

    # LED lighting - reduces connected load
    "led_lighting": {
        "lighting_reduction_percent": 70.0,  # LED uses 70% less than fluorescent
        "description": "Reduces lighting load from ~4 W/m² to ~1.2 W/m²",
    },
    "led_common_areas": {
        "lighting_reduction_percent": 70.0,
        "description": "Reduces common area lighting load",
    },
    "led_outdoor": {
        "lighting_reduction_percent": 60.0,  # Less savings for outdoor
        "description": "Reduces outdoor lighting load",
    },

    # Ventilation - affects fan power
    "demand_controlled_ventilation": {
        "ventilation_reduction_percent": 30.0,  # Fans run at lower speed
        "description": "VAV/DCV reduces average fan power",
    },

    # Heat pump optimizations
    "heat_pump_integration": {
        "heat_pump_reduction_percent": 10.0,  # Better COP = lower peak
        "affects_winter_only": True,
        "description": "Improved heat pump efficiency",
    },

    # Smart controls - load shifting
    "smart_thermostats": {
        "peak_reduction_percent": 5.0,  # Pre-heating capability
        "description": "Enables load shifting and pre-heating",
    },

    # BMS optimization
    "bms_optimization": {
        "peak_reduction_percent": 3.0,
        "description": "Better coordination of loads",
    },

    # Battery storage - peak shaving
    "battery_storage": {
        "peak_reduction_percent": 20.0,  # Discharge during peaks
        "description": "Discharge battery during peak demand periods",
    },

    # VFD on pumps
    "pump_optimization": {
        "pump_reduction_percent": 50.0,  # VFD on circulation pumps
        "description": "Variable speed pumps reduce peak demand",
    },

    # Solar PV - reduces purchased peak (but not always during peak hours)
    "solar_pv": {
        "peak_reduction_percent": 5.0,  # Limited effect during winter peaks
        "affects_summer_only": True,
        "description": "Reduces purchased power during sunny hours",
    },
}


def calculate_ecm_peak_savings(
    ecm_id: str,
    building_peak: BuildingPeakEstimate,
    tariff: EffektTariff = None,
) -> Tuple[float, float]:
    """
    Calculate annual effektavgift savings from an ECM.

    Args:
        ecm_id: ECM identifier
        building_peak: Current building peak estimate
        tariff: Effekt tariff (default: Ellevio 2025)

    Returns:
        Tuple of (peak_reduction_kw, annual_savings_sek)
    """
    if tariff is None:
        tariff = ELLEVIO_EFFEKTTARIFF

    impact = ECM_PEAK_IMPACTS.get(ecm_id, {})
    if not impact:
        return 0.0, 0.0

    peak_reduction_kw = 0.0

    # Percentage reduction of total peak
    if "peak_reduction_percent" in impact:
        reduction_pct = impact["peak_reduction_percent"] / 100
        peak_reduction_kw = building_peak.total_peak_kw * reduction_pct

    # Component-specific reductions
    if "lighting_reduction_percent" in impact:
        reduction_pct = impact["lighting_reduction_percent"] / 100
        peak_reduction_kw += building_peak.lighting_kw * reduction_pct

    if "ventilation_reduction_percent" in impact:
        reduction_pct = impact["ventilation_reduction_percent"] / 100
        peak_reduction_kw += building_peak.ventilation_kw * reduction_pct

    if "heat_pump_reduction_percent" in impact:
        reduction_pct = impact["heat_pump_reduction_percent"] / 100
        peak_reduction_kw += building_peak.heat_pump_kw * reduction_pct

    if "pump_reduction_percent" in impact:
        # Pumps are part of common equipment
        reduction_pct = impact["pump_reduction_percent"] / 100
        pump_load = building_peak.common_equipment_kw * 0.4  # Pumps ~40% of common
        peak_reduction_kw += pump_load * reduction_pct

    # Calculate annual savings
    affects_winter = not impact.get("affects_summer_only", False)
    affects_summer = not impact.get("affects_winter_only", False)

    annual_savings = 0.0
    if affects_winter:
        annual_savings += peak_reduction_kw * tariff.peak_charge_day_sek_kw * 6  # 6 winter months
    if affects_summer:
        annual_savings += peak_reduction_kw * tariff.peak_charge_day_sek_kw * 6  # 6 summer months

    return peak_reduction_kw, annual_savings


def calculate_combined_peak_savings(
    ecm_ids: List[str],
    building_peak: BuildingPeakEstimate,
    tariff: EffektTariff = None,
    interaction_factor: float = 0.85,
) -> Tuple[float, float]:
    """
    Calculate combined effektavgift savings from multiple ECMs.

    Applies interaction factor to avoid double-counting.

    Args:
        ecm_ids: List of ECM identifiers
        building_peak: Current building peak estimate
        tariff: Effekt tariff
        interaction_factor: Discount for overlapping effects (0.85 = 15% overlap)

    Returns:
        Tuple of (total_peak_reduction_kw, annual_savings_sek)
    """
    if tariff is None:
        tariff = ELLEVIO_EFFEKTTARIFF

    total_reduction = 0.0

    for i, ecm_id in enumerate(ecm_ids):
        reduction_kw, _ = calculate_ecm_peak_savings(ecm_id, building_peak, tariff)

        # Apply interaction factor for subsequent ECMs
        if i > 0:
            reduction_kw *= interaction_factor ** i

        total_reduction += reduction_kw

    # Cap at realistic maximum (can't reduce peak below base load)
    min_base_load = building_peak.total_peak_kw * 0.2
    total_reduction = min(total_reduction, building_peak.total_peak_kw - min_base_load)

    annual_savings = total_reduction * tariff.peak_charge_day_sek_kw * 12

    return total_reduction, annual_savings


# =============================================================================
# BATTERY STORAGE VIABILITY (Sweden 2025 market conditions)
# =============================================================================

@dataclass
class BatteryViabilityResult:
    """Assessment of battery storage viability for a building."""
    viable: bool
    roi_rating: str  # "good", "marginal", "poor"
    simple_payback_years: float
    npv_10yr_sek: float
    warnings: List[str]
    recommendations: List[str]

    # Savings breakdown
    effekt_savings_sek: float
    arbitrage_savings_sek: float
    self_consumption_savings_sek: float
    total_annual_savings_sek: float

    # Costs
    investment_sek: float
    annual_maintenance_sek: float


def evaluate_battery_viability(
    building_peak: BuildingPeakEstimate,
    has_solar_pv: bool = False,
    existing_solar_kwp: float = 0,
    flexibility_market: str = "poor",  # "poor", "moderate", "good"
    tariff: EffektTariff = None,
) -> BatteryViabilityResult:
    """
    Evaluate battery storage viability for Swedish BRF (2025 conditions).

    Sweden's flexibility markets are currently immature:
    - FCR-N/D: Requires 1MW minimum, prequalification
    - FFR: Limited availability
    - mFRR: Complex bidding
    - Spot arbitrage: ~0.50 SEK/kWh spread after losses

    Battery ROI primarily depends on:
    1. Effekttariff savings (if on Ellevio-type tariff)
    2. Solar self-consumption (if has PV)
    3. Spot arbitrage (minimal)

    Args:
        building_peak: Building peak power estimate
        has_solar_pv: Whether building has solar PV
        existing_solar_kwp: Existing PV capacity
        flexibility_market: Market quality ("poor", "moderate", "good")
        tariff: Effekt tariff (default: Ellevio 2025)

    Returns:
        BatteryViabilityResult with ROI assessment and recommendations
    """
    if tariff is None:
        tariff = ELLEVIO_EFFEKTTARIFF

    warnings = []
    recommendations = []

    # Size battery for peak shaving (2 hours of peak reduction)
    peak_reduction_kw = building_peak.total_peak_kw * 0.20  # 20% peak reduction
    battery_kwh = peak_reduction_kw * 2  # 2 hours discharge
    battery_kwh = max(10, min(battery_kwh, 100))  # Clamp to 10-100 kWh

    # Investment cost (from ECM_COSTS_V2)
    material_per_kwh = 5000  # SEK/kWh
    labor_per_kwh = 1500  # SEK/kWh
    fixed_cost = 20000  # SEK
    investment_sek = (material_per_kwh + labor_per_kwh) * battery_kwh + fixed_cost

    # Annual maintenance
    maintenance_per_kwh = 50  # SEK/kWh/year
    annual_maintenance_sek = maintenance_per_kwh * battery_kwh

    # ═══════════════════════════════════════════════════════════════════════
    # SAVINGS CALCULATION
    # ═══════════════════════════════════════════════════════════════════════

    # 1. Effekttariff savings (the main value driver in 2025)
    effekt_savings_sek = peak_reduction_kw * tariff.peak_charge_day_sek_kw * 12

    # 2. Arbitrage savings (minimal in Sweden without flexibility markets)
    cycles_per_year = 300  # Peak shaving cycles
    if flexibility_market == "poor":
        arbitrage_per_cycle = 0.50  # SEK/kWh (spot spread after losses)
        warnings.append("Sweden's flexibility markets are immature (2025)")
    elif flexibility_market == "moderate":
        arbitrage_per_cycle = 1.50  # SEK/kWh
    else:
        arbitrage_per_cycle = 3.00  # SEK/kWh
    arbitrage_savings_sek = battery_kwh * cycles_per_year * arbitrage_per_cycle * 0.9  # 90% DoD

    # 3. Solar self-consumption savings (only if has PV)
    self_consumption_savings_sek = 0
    if has_solar_pv and existing_solar_kwp > 0:
        # Estimate self-consumption increase
        # Without battery: ~30% self-consumption
        # With battery: ~60% self-consumption (varies by size ratio)
        battery_to_pv_ratio = min(battery_kwh / existing_solar_kwp, 2.0)
        self_consumption_increase = 0.20 + 0.10 * battery_to_pv_ratio  # 20-40% increase

        # PV production ~900 kWh/kWp in Sweden
        annual_pv_kwh = existing_solar_kwp * 900
        additional_self_consumed = annual_pv_kwh * self_consumption_increase

        # Value: difference between retail price and feed-in tariff
        retail_price = 2.00  # SEK/kWh (incl moms, nätavgift, elhandel)
        feed_in_price = 0.80  # SEK/kWh (typical spot + certificate)
        self_consumption_value = retail_price - feed_in_price

        self_consumption_savings_sek = additional_self_consumed * self_consumption_value
    else:
        warnings.append("No solar PV - battery value limited to peak shaving")
        recommendations.append("Install solar PV before battery for better ROI")

    # Total annual savings
    total_annual_savings_sek = (
        effekt_savings_sek +
        arbitrage_savings_sek +
        self_consumption_savings_sek -
        annual_maintenance_sek
    )

    # ═══════════════════════════════════════════════════════════════════════
    # ROI CALCULATION
    # ═══════════════════════════════════════════════════════════════════════

    if total_annual_savings_sek > 0:
        simple_payback_years = investment_sek / total_annual_savings_sek
    else:
        simple_payback_years = 999  # No payback

    # NPV over 10 years (battery lifespan ~12-15 years but use 10 for conservatism)
    discount_rate = 0.05
    npv_10yr_sek = -investment_sek
    for year in range(1, 11):
        npv_10yr_sek += total_annual_savings_sek / ((1 + discount_rate) ** year)

    # ═══════════════════════════════════════════════════════════════════════
    # VIABILITY ASSESSMENT
    # ═══════════════════════════════════════════════════════════════════════

    battery_lifespan = 12  # years

    if simple_payback_years > battery_lifespan:
        viable = False
        roi_rating = "poor"
        warnings.append(f"Payback ({simple_payback_years:.0f} years) exceeds battery lifespan ({battery_lifespan} years)")
    elif simple_payback_years > 8:
        viable = True
        roi_rating = "marginal"
        warnings.append("Long payback period - consider waiting for price drops")
    elif simple_payback_years > 5:
        viable = True
        roi_rating = "marginal"
    else:
        viable = True
        roi_rating = "good"

    # Additional recommendations
    if not has_solar_pv:
        recommendations.append("Solar PV + battery is ~2x more profitable than battery alone")

    if flexibility_market == "poor":
        recommendations.append("Monitor Svenska Kraftnät flexibility markets for future opportunities")

    if effekt_savings_sek < investment_sek * 0.05:
        warnings.append("Low effektavgift benefit - check if building is on effekttariff")

    return BatteryViabilityResult(
        viable=viable,
        roi_rating=roi_rating,
        simple_payback_years=simple_payback_years,
        npv_10yr_sek=npv_10yr_sek,
        warnings=warnings,
        recommendations=recommendations,
        effekt_savings_sek=effekt_savings_sek,
        arbitrage_savings_sek=arbitrage_savings_sek,
        self_consumption_savings_sek=self_consumption_savings_sek,
        total_annual_savings_sek=total_annual_savings_sek + annual_maintenance_sek,  # Gross
        investment_sek=investment_sek,
        annual_maintenance_sek=annual_maintenance_sek,
    )


# =============================================================================
# PACKAGE COST SYNERGIES (shared scaffolding, contractors, etc.)
# =============================================================================

# When combining ECMs, some costs reduce
# Key: (ecm_a, ecm_b), Value: cost multiplier (0.85 = 15% savings)
PACKAGE_COST_SYNERGIES: Dict[Tuple[str, str], float] = {
    # Shared scaffolding (ställning) - major savings!
    ("wall_external_insulation", "window_replacement"): 0.85,
    ("wall_external_insulation", "roof_insulation"): 0.92,
    ("wall_external_insulation", "facade_renovation"): 0.80,
    ("window_replacement", "roof_insulation"): 0.95,

    # Shared HVAC contractor
    ("ftx_installation", "demand_controlled_ventilation"): 0.80,
    ("ftx_upgrade", "demand_controlled_ventilation"): 0.85,
    ("exhaust_air_heat_pump", "ftx_installation"): 0.90,

    # Shared electrical work
    ("smart_thermostats", "led_lighting"): 0.90,
    ("smart_thermostats", "occupancy_sensors"): 0.88,
    ("solar_pv", "battery_storage"): 0.92,
    ("solar_pv", "ev_charging"): 0.95,

    # Shared plumbing
    ("low_flow_fixtures", "pipe_insulation"): 0.90,
    ("heat_pump_water_heater", "dhw_circulation_optimization"): 0.88,

    # Control system integration
    ("building_automation_system", "energy_monitoring"): 0.85,
    ("building_automation_system", "smart_thermostats"): 0.80,
    ("building_automation_system", "demand_controlled_ventilation"): 0.85,
}


def get_package_cost_multiplier(ecm_ids: List[str]) -> float:
    """
    Calculate total cost multiplier for a package of ECMs.

    Returns a multiplier < 1.0 if there are cost synergies.
    """
    if len(ecm_ids) < 2:
        return 1.0

    # Find all applicable synergies
    total_discount = 0.0

    for i, ecm_a in enumerate(ecm_ids):
        for ecm_b in ecm_ids[i+1:]:
            # Check both orderings
            synergy = PACKAGE_COST_SYNERGIES.get((ecm_a, ecm_b))
            if synergy is None:
                synergy = PACKAGE_COST_SYNERGIES.get((ecm_b, ecm_a))

            if synergy is not None:
                # Accumulate discounts (1 - synergy = discount fraction)
                total_discount += (1 - synergy)

    # Cap at 30% maximum discount
    return max(0.70, 1.0 - total_discount)


# =============================================================================
# SCAFFOLDING COSTS (critical for facade work)
# =============================================================================

SCAFFOLDING_COST_SEK_M2 = 150  # Per m² facade, rental + setup + removal
SCAFFOLDING_MIN_WEEKS = 6  # Minimum rental period


def estimate_scaffolding_cost(
    facade_area_m2: float,
    building_height_floors: int = 4,
    duration_weeks: int = 8,
) -> float:
    """
    Estimate scaffolding cost for facade work.

    Args:
        facade_area_m2: Total facade area to be accessed
        building_height_floors: Number of floors (affects complexity)
        duration_weeks: Expected duration of work

    Returns:
        Estimated scaffolding cost in SEK
    """
    base_cost = facade_area_m2 * SCAFFOLDING_COST_SEK_M2

    # Height premium (high-rise more expensive)
    if building_height_floors > 6:
        base_cost *= 1.25
    elif building_height_floors > 4:
        base_cost *= 1.10

    # Duration premium if longer than minimum
    if duration_weeks > SCAFFOLDING_MIN_WEEKS:
        extra_weeks = duration_weeks - SCAFFOLDING_MIN_WEEKS
        weekly_rental = facade_area_m2 * 15  # ~15 SEK/m²/week rental
        base_cost += extra_weeks * weekly_rental

    return base_cost


# =============================================================================
# PROJECT OVERHEAD AND CONTINGENCY
# =============================================================================

def calculate_project_overhead(
    base_cost: float,
    building_age_years: int = 40,
    scope: str = "medium",  # small, medium, large
) -> Dict[str, float]:
    """
    Calculate project overhead costs.

    Returns breakdown of indirect costs.
    """
    overhead = {}

    # Project management (projektledning)
    pm_rates = {"small": 0.05, "medium": 0.08, "large": 0.12}
    overhead["project_management"] = base_cost * pm_rates.get(scope, 0.08)

    # Design/engineering (projektering)
    design_rates = {"small": 0.03, "medium": 0.05, "large": 0.08}
    overhead["design_engineering"] = base_cost * design_rates.get(scope, 0.05)

    # Contingency (oförutsett) - higher for older buildings
    if building_age_years > 60:
        contingency_rate = 0.15  # Pre-1965: many unknowns
    elif building_age_years > 40:
        contingency_rate = 0.10  # 1965-1985: concrete panel era
    else:
        contingency_rate = 0.05  # Modern buildings

    overhead["contingency"] = base_cost * contingency_rate

    # Building permit if required (estimate)
    overhead["permits_inspections"] = min(50000, base_cost * 0.02)

    overhead["total"] = sum(overhead.values())

    return overhead


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class CostEntry:
    """A single cost data point with metadata."""

    value_sek: float
    unit: str  # "SEK/m²", "SEK/kW", "SEK/unit", etc.
    source: CostSource
    year: int
    confidence: float = 0.8  # 0-1, reliability of this data
    notes: Optional[str] = None

    def inflate_to(self, target_year: int, annual_rate: float = ANNUAL_INFLATION_RATE) -> float:
        """Inflate cost to target year using compound growth."""
        years = target_year - self.year
        if years == 0:
            return self.value_sek
        return self.value_sek * (1 + annual_rate) ** years

    def with_confidence_adjustment(self) -> float:
        """
        Adjust value based on confidence (for uncertainty modeling).

        Low confidence → add buffer for safety margin.
        """
        if self.confidence >= 0.8:
            return self.value_sek
        elif self.confidence >= 0.6:
            return self.value_sek * 1.1  # 10% buffer
        else:
            return self.value_sek * 1.2  # 20% buffer


@dataclass
class ECMCostModel:
    """
    Complete cost model for an ECM.

    Separates material and labor for proper ROT calculation.
    Includes maintenance, lifetime, and Swedish deductions.
    """

    ecm_id: str
    name_sv: str  # Swedish name

    # Core costs
    material_cost: CostEntry
    labor_cost: CostEntry
    fixed_cost: Optional[CostEntry] = None  # Per-project overhead

    # Lifecycle
    lifetime_years: int = 25
    annual_maintenance: Optional[CostEntry] = None

    # Swedish deductions
    rot_eligible: bool = False  # 50% labor deduction (max 50k SEK/person/year)
    green_tech_eligible: bool = False  # 15% (from July 2025) or 20% (solar PV)
    energy_grant_eligible: bool = False  # Boverket/Energimyndigheten grants

    # Scaling behavior
    # Options: floor_area, wall_area, roof_area, window_area, capacity, unit, per_apartment, per_building
    scales_with: str = "floor_area"
    has_economies_of_scale: bool = True

    # Category
    category: CostCategory = CostCategory.MEDIUM_COST

    def calculate_cost(
        self,
        quantity: float,
        year: int = 2025,
        region: Region = Region.MEDIUM_CITY,
        floor_area_m2: float = 1000,
        owner_type: "OwnerType" = None,
        include_maintenance: bool = False,
        analysis_period_years: int = 25,
    ) -> "CostBreakdown":
        """
        Calculate total cost with all adjustments.

        Args:
            quantity: Quantity in appropriate units
            year: Target year for inflation
            region: Swedish region for cost adjustment
            floor_area_m2: Building size for scaling
            owner_type: Building owner type (affects tax deductions)
                - PRIVATE: ROT + green tech eligible
                - BRF/RENTAL/COMMERCIAL: No ROT, no green tech (use grants instead)
            include_maintenance: Whether to include LCC maintenance
            analysis_period_years: Years for maintenance calculation

        Returns:
            CostBreakdown with all cost components
        """
        # Default to BRF (multi-family) since that's the target use case
        if owner_type is None:
            owner_type = OwnerType.BRF

        # Base costs inflated to target year
        material = self.material_cost.inflate_to(year) * quantity
        labor = self.labor_cost.inflate_to(year) * quantity
        fixed = self.fixed_cost.inflate_to(year) if self.fixed_cost else 0

        # Regional adjustment
        regional_mult = REGIONAL_MULTIPLIERS.get(region, 1.0)
        material *= regional_mult
        labor *= regional_mult
        fixed *= regional_mult

        # Size scaling (only for variable costs, not fixed)
        if self.has_economies_of_scale:
            scale = size_scaling_factor(floor_area_m2)
            material *= scale
            labor *= scale

        # Swedish deductions - ONLY for private individuals!
        rot_deduction = 0
        green_deduction = 0

        if owner_type == OwnerType.PRIVATE:
            # ROT: 50% of labor, max 50,000 SEK per person per year
            # Only for private homeowners (villa, bostadsrätt)
            if self.rot_eligible:
                rot_deduction = min(labor * 0.5, 50000)

            # Green tech: 15% of material+labor (solar, battery, EV charging)
            # Only for private individuals
            if self.green_tech_eligible:
                green_deduction = (material + labor) * 0.15

        # Note: BRF/commercial can apply for:
        # - Energimyndigheten grants (case by case)
        # - Klimatklivet (industrial scale)
        # - Boverket renovationsstöd
        # These are NOT automatic deductions, so not included here

        subtotal = material + labor + fixed - rot_deduction - green_deduction

        # Maintenance costs (present value)
        maintenance_total = 0
        if include_maintenance and self.annual_maintenance:
            annual = self.annual_maintenance.inflate_to(year) * quantity
            # Simple sum (could use NPV with discount rate)
            maintenance_total = annual * min(analysis_period_years, self.lifetime_years)

        return CostBreakdown(
            ecm_id=self.ecm_id,
            material_cost=material,
            labor_cost=labor,
            fixed_cost=fixed,
            rot_deduction=rot_deduction,
            green_tech_deduction=green_deduction,
            maintenance_cost=maintenance_total,
            total_before_deductions=material + labor + fixed,
            total_after_deductions=subtotal + maintenance_total,
            quantity=quantity,
            unit=self.material_cost.unit,
            year=year,
            region=region,
            owner_type=owner_type,
        )


@dataclass
class CostBreakdown:
    """Detailed cost breakdown for transparency."""

    ecm_id: str
    material_cost: float
    labor_cost: float
    fixed_cost: float
    rot_deduction: float
    green_tech_deduction: float
    maintenance_cost: float
    total_before_deductions: float
    total_after_deductions: float
    quantity: float
    unit: str
    year: int
    region: Region
    owner_type: "OwnerType" = None  # Added to track deduction eligibility

    def to_dict(self) -> Dict:
        """Export as dictionary."""
        return {
            "ecm_id": self.ecm_id,
            "material_cost_sek": round(self.material_cost),
            "labor_cost_sek": round(self.labor_cost),
            "fixed_cost_sek": round(self.fixed_cost),
            "rot_deduction_sek": round(self.rot_deduction),
            "green_tech_deduction_sek": round(self.green_tech_deduction),
            "maintenance_cost_sek": round(self.maintenance_cost),
            "total_before_deductions_sek": round(self.total_before_deductions),
            "total_after_deductions_sek": round(self.total_after_deductions),
            "quantity": self.quantity,
            "unit": self.unit,
            "year": self.year,
            "region": self.region.value,
            "owner_type": self.owner_type.value if self.owner_type else "brf",
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Cost Breakdown: {self.ecm_id}",
            f"  Quantity: {self.quantity:.1f} {self.unit}",
            f"  Material: {self.material_cost:,.0f} SEK",
            f"  Labor: {self.labor_cost:,.0f} SEK",
        ]
        if self.fixed_cost > 0:
            lines.append(f"  Fixed: {self.fixed_cost:,.0f} SEK")
        if self.rot_deduction > 0:
            lines.append(f"  ROT deduction: -{self.rot_deduction:,.0f} SEK")
        if self.green_tech_deduction > 0:
            lines.append(f"  Green tech deduction: -{self.green_tech_deduction:,.0f} SEK")
        if self.maintenance_cost > 0:
            lines.append(f"  Maintenance (LCC): {self.maintenance_cost:,.0f} SEK")
        lines.extend([
            f"  ─────────────────────",
            f"  Total before deductions: {self.total_before_deductions:,.0f} SEK",
            f"  Total after deductions: {self.total_after_deductions:,.0f} SEK",
        ])
        return "\n".join(lines)


# =============================================================================
# ECM COST DATABASE
# =============================================================================

ECM_COSTS_V2: Dict[str, ECMCostModel] = {

    # =========================================================================
    # ENVELOPE MEASURES
    # =========================================================================

    "wall_external_insulation": ECMCostModel(
        ecm_id="wall_external_insulation",
        name_sv="Tilläggsisolering fasad (utvändig)",
        material_cost=CostEntry(
            value_sek=800,
            unit="SEK/m² wall",
            source=CostSource.BEBO_TYPKOSTNADER_2023,
            year=2023,
            confidence=0.75,
            notes="100mm mineral wool + rendering/facade boards"
        ),
        labor_cost=CostEntry(
            value_sek=700,
            unit="SEK/m² wall",
            source=CostSource.BEBO_TYPKOSTNADER_2023,
            year=2023,
            confidence=0.75,
            notes="Includes scaffolding labor"
        ),
        fixed_cost=CostEntry(
            value_sek=80000,
            unit="SEK/building",
            source=CostSource.BEBO_LONSAMHET_2023,
            year=2023,
            confidence=0.7,
            notes="Scaffolding setup, project management"
        ),
        lifetime_years=40,
        rot_eligible=True,
        scales_with="wall_area",
        category=CostCategory.MAJOR,
    ),

    "wall_internal_insulation": ECMCostModel(
        ecm_id="wall_internal_insulation",
        name_sv="Tilläggsisolering fasad (invändig)",
        material_cost=CostEntry(
            value_sek=400,
            unit="SEK/m² wall",
            source=CostSource.WIKELLS_2024,
            year=2024,
            confidence=0.8,
            notes="50-80mm insulation + gypsum board"
        ),
        labor_cost=CostEntry(
            value_sek=400,
            unit="SEK/m² wall",
            source=CostSource.WIKELLS_2024,
            year=2024,
            confidence=0.75,
        ),
        lifetime_years=40,
        rot_eligible=True,
        scales_with="wall_area",
        category=CostCategory.HIGH_COST,
    ),

    "roof_insulation": ECMCostModel(
        ecm_id="roof_insulation",
        name_sv="Vindsisolering",
        material_cost=CostEntry(
            value_sek=250,
            unit="SEK/m² roof",
            source=CostSource.BEBO_TYPKOSTNADER_2023,
            year=2023,
            confidence=0.8,
            notes="200mm loose-fill or batts"
        ),
        labor_cost=CostEntry(
            value_sek=150,
            unit="SEK/m² roof",
            source=CostSource.BEBO_TYPKOSTNADER_2023,
            year=2023,
            confidence=0.8,
        ),
        lifetime_years=40,
        rot_eligible=True,
        scales_with="roof_area",
        category=CostCategory.MEDIUM_COST,
    ),

    "window_replacement": ECMCostModel(
        ecm_id="window_replacement",
        name_sv="Fönsterbyte",
        material_cost=CostEntry(
            value_sek=4000,
            unit="SEK/m² window",
            source=CostSource.WIKELLS_2024,
            year=2024,
            confidence=0.8,
            notes="Triple glazing U=0.9-1.0"
        ),
        labor_cost=CostEntry(
            value_sek=2000,
            unit="SEK/m² window",
            source=CostSource.WIKELLS_2024,
            year=2024,
            confidence=0.75,
        ),
        fixed_cost=CostEntry(
            value_sek=20000,
            unit="SEK/building",
            source=CostSource.ESTIMATED,
            year=2024,
            confidence=0.6,
            notes="Project overhead"
        ),
        lifetime_years=30,
        rot_eligible=True,
        scales_with="window_area",
        category=CostCategory.MAJOR,
    ),

    "air_sealing": ECMCostModel(
        ecm_id="air_sealing",
        name_sv="Tätning luftläckage",
        material_cost=CostEntry(
            value_sek=15,
            unit="SEK/m² floor",
            source=CostSource.BEBO_LONSAMHET_2023,
            year=2023,
            confidence=0.7,
            notes="Sealants, gaskets, caulk"
        ),
        labor_cost=CostEntry(
            value_sek=35,
            unit="SEK/m² floor",
            source=CostSource.BEBO_LONSAMHET_2023,
            year=2023,
            confidence=0.7,
        ),
        fixed_cost=CostEntry(
            value_sek=10000,
            unit="SEK/building",
            source=CostSource.BEBO_LONSAMHET_2023,
            year=2023,
            confidence=0.65,
            notes="Blower door test before/after"
        ),
        lifetime_years=20,
        rot_eligible=True,
        scales_with="floor_area",
        category=CostCategory.LOW_COST,
    ),

    # =========================================================================
    # HVAC - HEAT PUMPS
    # =========================================================================

    "exhaust_air_heat_pump": ECMCostModel(
        ecm_id="exhaust_air_heat_pump",
        name_sv="Frånluftsvärmepump (FVP)",
        material_cost=CostEntry(
            value_sek=60000,
            unit="SEK/unit",
            source=CostSource.MARKET_RESEARCH_2025,
            year=2025,
            confidence=0.8,
            notes="NIBE F470/F750 or equivalent, 8-12 kW"
        ),
        labor_cost=CostEntry(
            value_sek=30000,
            unit="SEK/unit",
            source=CostSource.MARKET_RESEARCH_2025,
            year=2025,
            confidence=0.75,
            notes="Installation, electrical, commissioning"
        ),
        fixed_cost=CostEntry(
            value_sek=15000,
            unit="SEK/unit",
            source=CostSource.ESTIMATED,
            year=2025,
            confidence=0.6,
            notes="DHW tank if needed"
        ),
        annual_maintenance=CostEntry(
            value_sek=2000,
            unit="SEK/year",
            source=CostSource.ENERGIMYNDIGHETEN_2024,
            year=2024,
            confidence=0.7,
        ),
        lifetime_years=15,
        rot_eligible=True,
        green_tech_eligible=True,
        scales_with="unit",
        category=CostCategory.HIGH_COST,
    ),

    "ground_source_heat_pump": ECMCostModel(
        ecm_id="ground_source_heat_pump",
        name_sv="Bergvärmepump",
        material_cost=CostEntry(
            value_sek=8000,
            unit="SEK/kW",
            source=CostSource.BEBO_LONSAMHET_2023,
            year=2023,
            confidence=0.75,
            notes="Commercial GSHP ~40 W/m² sizing. Equipment + installation."
        ),
        labor_cost=CostEntry(
            value_sek=4000,
            unit="SEK/kW",
            source=CostSource.BEBO_LONSAMHET_2023,
            year=2023,
            confidence=0.7,
            notes="System integration, controls, commissioning"
        ),
        fixed_cost=CostEntry(
            value_sek=12000,
            unit="SEK/kW",
            source=CostSource.MARKET_RESEARCH_2025,
            year=2025,
            confidence=0.7,
            notes="Borehole drilling ~20m/kW @ 600 SEK/m = 12,000 SEK/kW"
        ),
        annual_maintenance=CostEntry(
            value_sek=300,
            unit="SEK/kW/year",
            source=CostSource.ENERGIMYNDIGHETEN_2024,
            year=2024,
            confidence=0.7,
        ),
        lifetime_years=25,
        rot_eligible=True,
        green_tech_eligible=True,
        scales_with="capacity",
        has_economies_of_scale=True,
        category=CostCategory.MAJOR,
    ),

    "heat_pump_integration": ECMCostModel(
        ecm_id="heat_pump_integration",
        name_sv="Värmepumpsintegration",
        material_cost=CostEntry(
            value_sek=2500,
            unit="SEK/kW",
            source=CostSource.SABO_2024,
            year=2024,
            confidence=0.7,
            notes="Generic HP integration (type unspecified)"
        ),
        labor_cost=CostEntry(
            value_sek=1000,
            unit="SEK/kW",
            source=CostSource.SABO_2024,
            year=2024,
            confidence=0.7,
        ),
        fixed_cost=CostEntry(
            value_sek=50000,
            unit="SEK/building",
            source=CostSource.SABO_2024,
            year=2024,
            confidence=0.65,
        ),
        lifetime_years=20,
        rot_eligible=True,
        green_tech_eligible=True,
        scales_with="capacity",
        category=CostCategory.MAJOR,
    ),

    # =========================================================================
    # HVAC - VENTILATION
    # =========================================================================

    "ftx_installation": ECMCostModel(
        ecm_id="ftx_installation",
        name_sv="FTX-installation",
        material_cost=CostEntry(
            value_sek=700,
            unit="SEK/m² floor",
            source=CostSource.BEBO_TYPKOSTNADER_2023,
            year=2023,
            confidence=0.7,
            notes="Central system for multi-family"
        ),
        labor_cost=CostEntry(
            value_sek=500,
            unit="SEK/m² floor",
            source=CostSource.BEBO_TYPKOSTNADER_2023,
            year=2023,
            confidence=0.7,
        ),
        fixed_cost=CostEntry(
            value_sek=150000,
            unit="SEK/building",
            source=CostSource.BEBO_LONSAMHET_2023,
            year=2023,
            confidence=0.65,
            notes="AHU, roof penetrations, controls"
        ),
        annual_maintenance=CostEntry(
            value_sek=8,
            unit="SEK/m²/year",
            source=CostSource.BEBO_LONSAMHET_2023,
            year=2023,
            confidence=0.7,
            notes="Filters, inspections, OVK"
        ),
        lifetime_years=25,
        rot_eligible=True,
        scales_with="floor_area",
        category=CostCategory.MAJOR,
    ),

    "ftx_upgrade": ECMCostModel(
        ecm_id="ftx_upgrade",
        name_sv="FTX-uppgradering",
        material_cost=CostEntry(
            value_sek=120,
            unit="SEK/m² floor",
            source=CostSource.BEBO_LONSAMHET_2023,
            year=2023,
            confidence=0.7,
            notes="New heat exchanger, EC motors"
        ),
        labor_cost=CostEntry(
            value_sek=80,
            unit="SEK/m² floor",
            source=CostSource.BEBO_LONSAMHET_2023,
            year=2023,
            confidence=0.7,
        ),
        lifetime_years=20,
        rot_eligible=True,
        scales_with="floor_area",
        category=CostCategory.MEDIUM_COST,
    ),

    "demand_controlled_ventilation": ECMCostModel(
        ecm_id="demand_controlled_ventilation",
        name_sv="Behovsstyrd ventilation (DCV)",
        material_cost=CostEntry(
            value_sek=80,
            unit="SEK/m² floor",
            source=CostSource.BEBO_LONSAMHET_2023,
            year=2023,
            confidence=0.75,
            notes="CO2/humidity sensors, dampers, controls"
        ),
        labor_cost=CostEntry(
            value_sek=70,
            unit="SEK/m² floor",
            source=CostSource.BEBO_LONSAMHET_2023,
            year=2023,
            confidence=0.75,
        ),
        lifetime_years=15,
        rot_eligible=True,
        scales_with="floor_area",
        category=CostCategory.MEDIUM_COST,
    ),

    # =========================================================================
    # HVAC - DISTRICT HEATING
    # =========================================================================

    "district_heating_optimization": ECMCostModel(
        ecm_id="district_heating_optimization",
        name_sv="Fjärrvärmeoptimering",
        material_cost=CostEntry(
            value_sek=0,
            unit="SEK/building",
            source=CostSource.BEBO_LONSAMHET_2023,
            year=2023,
            confidence=0.8,
            notes="Primarily consultant/adjustment work"
        ),
        labor_cost=CostEntry(
            value_sek=15000,
            unit="SEK/building",
            source=CostSource.BEBO_LONSAMHET_2023,
            year=2023,
            confidence=0.75,
            notes="Analysis, adjustment, commissioning"
        ),
        lifetime_years=5,
        rot_eligible=False,
        scales_with="unit",
        category=CostCategory.LOW_COST,
    ),

    # =========================================================================
    # RENEWABLES
    # =========================================================================

    "solar_pv": ECMCostModel(
        ecm_id="solar_pv",
        name_sv="Solceller (tak)",
        material_cost=CostEntry(
            value_sek=8000,
            unit="SEK/kWp",
            source=CostSource.ENERGIMYNDIGHETEN_2024,
            year=2024,
            confidence=0.85,
            notes="Panels, inverter, mounting, cables"
        ),
        labor_cost=CostEntry(
            value_sek=4000,
            unit="SEK/kWp",
            source=CostSource.ENERGIMYNDIGHETEN_2024,
            year=2024,
            confidence=0.8,
        ),
        fixed_cost=CostEntry(
            value_sek=25000,
            unit="SEK/system",
            source=CostSource.ENERGIMYNDIGHETEN_2024,
            year=2024,
            confidence=0.75,
            notes="Grid connection, permits"
        ),
        annual_maintenance=CostEntry(
            value_sek=50,
            unit="SEK/kWp/year",
            source=CostSource.ENERGIMYNDIGHETEN_2024,
            year=2024,
            confidence=0.7,
        ),
        lifetime_years=25,
        rot_eligible=True,
        green_tech_eligible=True,  # 15% from July 2025
        scales_with="capacity",
        category=CostCategory.MAJOR,
    ),

    "solar_thermal": ECMCostModel(
        ecm_id="solar_thermal",
        name_sv="Solfångare",
        material_cost=CostEntry(
            value_sek=5000,
            unit="SEK/m² collector",
            source=CostSource.ENERGIMYNDIGHETEN_2024,
            year=2024,
            confidence=0.75,
            notes="Flat plate collectors"
        ),
        labor_cost=CostEntry(
            value_sek=3000,
            unit="SEK/m² collector",
            source=CostSource.ENERGIMYNDIGHETEN_2024,
            year=2024,
            confidence=0.7,
        ),
        fixed_cost=CostEntry(
            value_sek=40000,
            unit="SEK/system",
            source=CostSource.ESTIMATED,
            year=2024,
            confidence=0.6,
            notes="Storage tank, piping, controls"
        ),
        lifetime_years=25,
        rot_eligible=True,
        green_tech_eligible=True,
        scales_with="capacity",
        category=CostCategory.HIGH_COST,
    ),

    # =========================================================================
    # CONTROLS
    # =========================================================================

    "smart_thermostats": ECMCostModel(
        ecm_id="smart_thermostats",
        name_sv="Smarta termostater",
        material_cost=CostEntry(
            value_sek=20,
            unit="SEK/m² floor",
            source=CostSource.ESTIMATED,
            year=2024,
            confidence=0.7,
            notes="~1500 SEK/apartment installed"
        ),
        labor_cost=CostEntry(
            value_sek=15,
            unit="SEK/m² floor",
            source=CostSource.ESTIMATED,
            year=2024,
            confidence=0.7,
        ),
        lifetime_years=10,
        rot_eligible=True,
        scales_with="floor_area",
        category=CostCategory.LOW_COST,
    ),

    "led_lighting": ECMCostModel(
        ecm_id="led_lighting",
        name_sv="LED-belysning",
        material_cost=CostEntry(
            value_sek=50,
            unit="SEK/m² floor",
            source=CostSource.ESTIMATED,
            year=2024,
            confidence=0.75,
            notes="Common areas"
        ),
        labor_cost=CostEntry(
            value_sek=30,
            unit="SEK/m² floor",
            source=CostSource.ESTIMATED,
            year=2024,
            confidence=0.7,
        ),
        lifetime_years=15,
        rot_eligible=True,
        scales_with="floor_area",
        category=CostCategory.LOW_COST,
    ),

    # =========================================================================
    # LOW-COST QUICK WINS
    # =========================================================================

    "low_flow_fixtures": ECMCostModel(
        ecm_id="low_flow_fixtures",
        name_sv="Snålspolande armaturer",
        material_cost=CostEntry(
            value_sek=800,
            unit="SEK/apartment",
            source=CostSource.ENERGIMYNDIGHETEN_2024,
            year=2024,
            confidence=0.8,
            notes="Showerheads, faucet aerators - per apartment"
        ),
        labor_cost=CostEntry(
            value_sek=700,
            unit="SEK/apartment",
            source=CostSource.ENERGIMYNDIGHETEN_2024,
            year=2024,
            confidence=0.75,
        ),
        lifetime_years=15,
        rot_eligible=True,
        scales_with="per_apartment",
        category=CostCategory.LOW_COST,
    ),

    "radiator_balancing": ECMCostModel(
        ecm_id="radiator_balancing",
        name_sv="Injustering av radiatorsystem",
        material_cost=CostEntry(
            value_sek=300,
            unit="SEK/apartment",
            source=CostSource.BEBO_LONSAMHET_2023,
            year=2023,
            confidence=0.7,
            notes="~3 radiators/apt × 100 SEK/radiator for valves"
        ),
        labor_cost=CostEntry(
            value_sek=600,
            unit="SEK/apartment",
            source=CostSource.BEBO_LONSAMHET_2023,
            year=2023,
            confidence=0.75,
            notes="~3 radiators/apt × 200 SEK/radiator for balancing"
        ),
        fixed_cost=CostEntry(
            value_sek=8000,
            unit="SEK/building",
            source=CostSource.BEBO_LONSAMHET_2023,
            year=2023,
            confidence=0.7,
            notes="Hydronic calculation, commissioning"
        ),
        lifetime_years=10,
        rot_eligible=True,
        scales_with="per_apartment",
        category=CostCategory.LOW_COST,
    ),

    "pipe_insulation": ECMCostModel(
        ecm_id="pipe_insulation",
        name_sv="Rörisolering",
        material_cost=CostEntry(
            value_sek=20,
            unit="SEK/m² floor",
            source=CostSource.SVEBY_2023,
            year=2023,
            confidence=0.7,
            notes="~0.1 m pipe per m² floor × 200 SEK/m = 20 SEK/m²"
        ),
        labor_cost=CostEntry(
            value_sek=30,
            unit="SEK/m² floor",
            source=CostSource.SVEBY_2023,
            year=2023,
            confidence=0.65,
        ),
        lifetime_years=30,
        rot_eligible=True,
        scales_with="floor_area",
        category=CostCategory.LOW_COST,
    ),

    # =========================================================================
    # ZERO-COST OPERATIONAL MEASURES
    # =========================================================================

    "heating_curve_adjustment": ECMCostModel(
        ecm_id="heating_curve_adjustment",
        name_sv="Framledningskurva-optimering",
        material_cost=CostEntry(
            value_sek=0,
            unit="SEK/building",
            source=CostSource.BEBO_LONSAMHET_2023,
            year=2023,
            confidence=0.9,
        ),
        labor_cost=CostEntry(
            value_sek=3000,
            unit="SEK/building",
            source=CostSource.BEBO_LONSAMHET_2023,
            year=2023,
            confidence=0.8,
            notes="2-4 hours of technician time"
        ),
        lifetime_years=3,
        rot_eligible=False,
        scales_with="unit",
        category=CostCategory.ZERO_COST,
    ),

    "bms_optimization": ECMCostModel(
        ecm_id="bms_optimization",
        name_sv="Styr- och regleropti",
        material_cost=CostEntry(
            value_sek=0,
            unit="SEK/building",
            source=CostSource.BEBO_LONSAMHET_2023,
            year=2023,
            confidence=0.9,
        ),
        labor_cost=CostEntry(
            value_sek=8000,
            unit="SEK/building",
            source=CostSource.BEBO_LONSAMHET_2023,
            year=2023,
            confidence=0.75,
            notes="Consultant review of all setpoints"
        ),
        lifetime_years=3,
        rot_eligible=False,
        scales_with="unit",
        category=CostCategory.ZERO_COST,
    ),

    # =========================================================================
    # PHASE 2 ECMs - DHW & STORAGE (Added 2025-12-26)
    # =========================================================================

    "heat_pump_water_heater": ECMCostModel(
        ecm_id="heat_pump_water_heater",
        name_sv="Frånluftsvärmepump varmvatten",
        material_cost=CostEntry(
            value_sek=45000,
            unit="SEK/unit",
            source=CostSource.MARKET_RESEARCH_2025,
            year=2025,
            confidence=0.75,
            notes="NIBE F1255 or equivalent HPWH"
        ),
        labor_cost=CostEntry(
            value_sek=25000,
            unit="SEK/unit",
            source=CostSource.MARKET_RESEARCH_2025,
            year=2025,
            confidence=0.7,
            notes="Installation, plumbing, electrical"
        ),
        fixed_cost=CostEntry(
            value_sek=10000,
            unit="SEK/building",
            source=CostSource.ESTIMATED,
            year=2025,
            confidence=0.6,
            notes="DHW storage if needed"
        ),
        annual_maintenance=CostEntry(
            value_sek=1500,
            unit="SEK/year",
            source=CostSource.ENERGIMYNDIGHETEN_2024,
            year=2024,
            confidence=0.7,
        ),
        lifetime_years=15,
        rot_eligible=True,
        green_tech_eligible=True,
        scales_with="unit",
        category=CostCategory.HIGH_COST,
    ),

    "battery_storage": ECMCostModel(
        ecm_id="battery_storage",
        name_sv="Batterilager",
        material_cost=CostEntry(
            value_sek=5000,
            unit="SEK/kWh",
            source=CostSource.MARKET_RESEARCH_2025,
            year=2025,
            confidence=0.8,
            notes="LFP or NMC battery modules"
        ),
        labor_cost=CostEntry(
            value_sek=1500,
            unit="SEK/kWh",
            source=CostSource.MARKET_RESEARCH_2025,
            year=2025,
            confidence=0.75,
            notes="Installation, inverter integration"
        ),
        fixed_cost=CostEntry(
            value_sek=20000,
            unit="SEK/system",
            source=CostSource.MARKET_RESEARCH_2025,
            year=2025,
            confidence=0.7,
            notes="Hybrid inverter, BMS, grid connection"
        ),
        annual_maintenance=CostEntry(
            value_sek=50,
            unit="SEK/kWh/year",
            source=CostSource.ESTIMATED,
            year=2025,
            confidence=0.6,
        ),
        lifetime_years=15,
        rot_eligible=True,
        green_tech_eligible=True,
        scales_with="capacity",
        category=CostCategory.HIGH_COST,
    ),

    "effektvakt_optimization": ECMCostModel(
        ecm_id="effektvakt_optimization",
        name_sv="Effektvakt-optimering",
        material_cost=CostEntry(
            value_sek=0,
            unit="SEK/building",
            source=CostSource.BEBO_LONSAMHET_2023,
            year=2023,
            confidence=0.9,
        ),
        labor_cost=CostEntry(
            value_sek=5000,
            unit="SEK/building",
            source=CostSource.BEBO_LONSAMHET_2023,
            year=2023,
            confidence=0.8,
            notes="Analysis and configuration of peak shaving"
        ),
        lifetime_years=5,
        rot_eligible=False,
        scales_with="unit",
        category=CostCategory.ZERO_COST,
    ),

    # =========================================================================
    # DHW OPTIMIZATION (Added 2025-12-26)
    # =========================================================================

    "dhw_circulation_optimization": ECMCostModel(
        ecm_id="dhw_circulation_optimization",
        name_sv="VVC-optimering",
        material_cost=CostEntry(
            value_sek=2000,
            unit="SEK/building",
            source=CostSource.BEBO_LONSAMHET_2023,
            year=2023,
            confidence=0.8,
            notes="Timer, temperature sensor"
        ),
        labor_cost=CostEntry(
            value_sek=6000,
            unit="SEK/building",
            source=CostSource.BEBO_LONSAMHET_2023,
            year=2023,
            confidence=0.75,
            notes="Installation and commissioning"
        ),
        lifetime_years=10,
        rot_eligible=False,
        scales_with="unit",
        category=CostCategory.LOW_COST,
    ),

    "dhw_tank_insulation": ECMCostModel(
        ecm_id="dhw_tank_insulation",
        name_sv="Ackumulatorisolering",
        material_cost=CostEntry(
            value_sek=3000,
            unit="SEK/tank",
            source=CostSource.SVEBY_2023,
            year=2023,
            confidence=0.8,
            notes="Insulation jacket"
        ),
        labor_cost=CostEntry(
            value_sek=2000,
            unit="SEK/tank",
            source=CostSource.SVEBY_2023,
            year=2023,
            confidence=0.75,
        ),
        lifetime_years=20,
        rot_eligible=True,
        scales_with="unit",
        category=CostCategory.LOW_COST,
    ),

    # =========================================================================
    # ADDITIONAL CONTROLS (Added 2025-12-26)
    # =========================================================================

    "fault_detection": ECMCostModel(
        ecm_id="fault_detection",
        name_sv="Feldetektering (FDD)",
        material_cost=CostEntry(
            value_sek=15000,
            unit="SEK/building",
            source=CostSource.MARKET_RESEARCH_2025,
            year=2025,
            confidence=0.7,
            notes="Software and sensors"
        ),
        labor_cost=CostEntry(
            value_sek=10000,
            unit="SEK/building",
            source=CostSource.MARKET_RESEARCH_2025,
            year=2025,
            confidence=0.65,
            notes="Integration with BMS"
        ),
        annual_maintenance=CostEntry(
            value_sek=3000,
            unit="SEK/year",
            source=CostSource.ESTIMATED,
            year=2025,
            confidence=0.6,
            notes="Software subscription"
        ),
        lifetime_years=10,
        rot_eligible=False,
        scales_with="unit",
        category=CostCategory.LOW_COST,
    ),

    "energy_monitoring": ECMCostModel(
        ecm_id="energy_monitoring",
        name_sv="Energivisualisering",
        material_cost=CostEntry(
            value_sek=400,
            unit="SEK/apartment",
            source=CostSource.ENERGIMYNDIGHETEN_2024,
            year=2024,
            confidence=0.75,
            notes="Display unit or app integration"
        ),
        labor_cost=CostEntry(
            value_sek=300,
            unit="SEK/apartment",
            source=CostSource.ENERGIMYNDIGHETEN_2024,
            year=2024,
            confidence=0.7,
        ),
        fixed_cost=CostEntry(
            value_sek=15000,
            unit="SEK/building",
            source=CostSource.ESTIMATED,
            year=2024,
            confidence=0.6,
            notes="Central system, gateway"
        ),
        lifetime_years=10,
        rot_eligible=False,
        scales_with="per_apartment",
        category=CostCategory.LOW_COST,
    ),

    # =========================================================================
    # LIGHTING (Added 2025-12-26)
    # =========================================================================

    "led_outdoor": ECMCostModel(
        ecm_id="led_outdoor",
        name_sv="LED utomhusbelysning",
        material_cost=CostEntry(
            value_sek=2000,
            unit="SEK/fixture",
            source=CostSource.MARKET_RESEARCH_2025,
            year=2025,
            confidence=0.8,
            notes="LED fixture with photocell"
        ),
        labor_cost=CostEntry(
            value_sek=1500,
            unit="SEK/fixture",
            source=CostSource.MARKET_RESEARCH_2025,
            year=2025,
            confidence=0.75,
        ),
        fixed_cost=CostEntry(
            value_sek=2000,
            unit="SEK/building",
            source=CostSource.ESTIMATED,
            year=2025,
            confidence=0.6,
        ),
        lifetime_years=15,
        rot_eligible=True,
        scales_with="unit",
        category=CostCategory.LOW_COST,
    ),

    "radiator_fans": ECMCostModel(
        ecm_id="radiator_fans",
        name_sv="Radiatorfläktar",
        material_cost=CostEntry(
            value_sek=35,  # ~500 SEK/radiator × 0.067 radiators/m² ≈ 35 SEK/m²
            unit="SEK/m²",
            source=CostSource.MARKET_RESEARCH_2025,
            year=2025,
            confidence=0.75,
            notes="Fan unit with thermostat; ~1 radiator per 15 m² floor area"
        ),
        labor_cost=CostEntry(
            value_sek=25,  # ~400 SEK/radiator × 0.067 radiators/m² ≈ 25 SEK/m²
            unit="SEK/m²",
            source=CostSource.MARKET_RESEARCH_2025,
            year=2025,
            confidence=0.7,
        ),
        lifetime_years=10,
        rot_eligible=True,
        scales_with="floor_area",  # Fixed: was "unit", radiators scale with floor area
        category=CostCategory.LOW_COST,
    ),

    # =========================================================================
    # PER-APARTMENT MEASURES
    # =========================================================================

    "individual_metering": ECMCostModel(
        ecm_id="individual_metering",
        name_sv="Individuell mätning och debitering (IMD)",
        material_cost=CostEntry(
            value_sek=8000,
            unit="SEK/apartment",
            source=CostSource.BEBO_LONSAMHET_2023,
            year=2023,
            confidence=0.75,
            notes="Smart meters for heating, water, electricity"
        ),
        labor_cost=CostEntry(
            value_sek=3000,
            unit="SEK/apartment",
            source=CostSource.BEBO_LONSAMHET_2023,
            year=2023,
            confidence=0.7,
            notes="Installation and commissioning per unit"
        ),
        fixed_cost=CostEntry(
            value_sek=50000,
            unit="SEK/building",
            source=CostSource.ESTIMATED,
            year=2023,
            confidence=0.6,
            notes="Central gateway and software setup"
        ),
        annual_maintenance=CostEntry(
            value_sek=500,
            unit="SEK/apartment/year",
            source=CostSource.ESTIMATED,
            year=2023,
            confidence=0.6,
            notes="Software subscription, calibration"
        ),
        lifetime_years=15,
        rot_eligible=False,
        scales_with="per_apartment",
        has_economies_of_scale=True,
        category=CostCategory.MEDIUM_COST,
    ),

    "water_efficient_fixtures": ECMCostModel(
        ecm_id="water_efficient_fixtures",
        name_sv="Snålspolande armaturer",
        material_cost=CostEntry(
            value_sek=2500,
            unit="SEK/apartment",
            source=CostSource.MARKET_RESEARCH_2025,
            year=2025,
            confidence=0.75,
            notes="Low-flow faucets, showerheads, toilet valves"
        ),
        labor_cost=CostEntry(
            value_sek=1500,
            unit="SEK/apartment",
            source=CostSource.ESTIMATED,
            year=2025,
            confidence=0.65,
            notes="Plumber installation time"
        ),
        lifetime_years=15,
        rot_eligible=True,
        scales_with="per_apartment",
        has_economies_of_scale=True,
        category=CostCategory.LOW_COST,
    ),

    "apartment_ventilation_units": ECMCostModel(
        ecm_id="apartment_ventilation_units",
        name_sv="Lägenhetsaggregat ventilation",
        material_cost=CostEntry(
            value_sek=25000,
            unit="SEK/apartment",
            source=CostSource.MARKET_RESEARCH_2025,
            year=2025,
            confidence=0.75,
            notes="Individual FTX unit with 80%+ HR"
        ),
        labor_cost=CostEntry(
            value_sek=15000,
            unit="SEK/apartment",
            source=CostSource.MARKET_RESEARCH_2025,
            year=2025,
            confidence=0.7,
            notes="Installation, ducting, electrical"
        ),
        annual_maintenance=CostEntry(
            value_sek=800,
            unit="SEK/apartment/year",
            source=CostSource.ESTIMATED,
            year=2025,
            confidence=0.6,
            notes="Filter changes, inspections"
        ),
        lifetime_years=20,
        rot_eligible=True,
        scales_with="per_apartment",
        has_economies_of_scale=True,
        category=CostCategory.MAJOR,
    ),

    # =========================================================================
    # ADDITIONAL HIGH-IMPACT ECMs (Added 2025-12-28)
    # =========================================================================

    "vrf_system": ECMCostModel(
        ecm_id="vrf_system",
        name_sv="VRF/VRV värmepumpsystem",
        material_cost=CostEntry(
            value_sek=500,
            unit="SEK/m² floor",
            source=CostSource.MARKET_RESEARCH_2025,
            year=2025,
            confidence=0.7,
            notes="Variable refrigerant flow system - equipment"
        ),
        labor_cost=CostEntry(
            value_sek=300,
            unit="SEK/m² floor",
            source=CostSource.MARKET_RESEARCH_2025,
            year=2025,
            confidence=0.65,
            notes="Installation, piping, controls"
        ),
        fixed_cost=CostEntry(
            value_sek=150000,
            unit="SEK/building",
            source=CostSource.ESTIMATED,
            year=2025,
            confidence=0.6,
            notes="Outdoor units, main piping runs, commissioning"
        ),
        annual_maintenance=CostEntry(
            value_sek=15,
            unit="SEK/m²/year",
            source=CostSource.ESTIMATED,
            year=2025,
            confidence=0.6,
            notes="Service contract, refrigerant checks"
        ),
        lifetime_years=20,
        rot_eligible=True,
        scales_with="floor_area",
        has_economies_of_scale=True,
        category=CostCategory.MAJOR,
    ),

    "facade_renovation": ECMCostModel(
        ecm_id="facade_renovation",
        name_sv="Fasadrenovering komplett",
        material_cost=CostEntry(
            value_sek=1200,
            unit="SEK/m² wall",
            source=CostSource.BEBO_TYPKOSTNADER_2023,
            year=2023,
            confidence=0.75,
            notes="Full facade renovation incl. insulation, rendering"
        ),
        labor_cost=CostEntry(
            value_sek=800,
            unit="SEK/m² wall",
            source=CostSource.BEBO_TYPKOSTNADER_2023,
            year=2023,
            confidence=0.7,
            notes="Scaffolding, labor, project management"
        ),
        fixed_cost=CostEntry(
            value_sek=100000,
            unit="SEK/building",
            source=CostSource.BEBO_LONSAMHET_2023,
            year=2023,
            confidence=0.65,
            notes="Scaffolding setup, permits, design"
        ),
        lifetime_years=40,
        rot_eligible=True,
        scales_with="wall_area",
        has_economies_of_scale=True,
        category=CostCategory.MAJOR,
    ),

    "air_source_heat_pump": ECMCostModel(
        ecm_id="air_source_heat_pump",
        name_sv="Luft-vatten värmepump",
        material_cost=CostEntry(
            value_sek=10000,
            unit="SEK/kW",
            source=CostSource.BEBO_LONSAMHET_2023,
            year=2023,
            confidence=0.75,
            notes="ASHP unit - ~40 W/m² typical sizing"
        ),
        labor_cost=CostEntry(
            value_sek=5000,
            unit="SEK/kW",
            source=CostSource.BEBO_LONSAMHET_2023,
            year=2023,
            confidence=0.7,
            notes="Installation, integration with existing system"
        ),
        fixed_cost=CostEntry(
            value_sek=80000,
            unit="SEK/building",
            source=CostSource.ESTIMATED,
            year=2023,
            confidence=0.6,
            notes="System integration, controls, buffer tank"
        ),
        annual_maintenance=CostEntry(
            value_sek=500,
            unit="SEK/kW/year",
            source=CostSource.ESTIMATED,
            year=2023,
            confidence=0.6,
            notes="Service, compressor checks"
        ),
        lifetime_years=18,
        rot_eligible=True,
        green_tech_eligible=True,
        scales_with="capacity",
        has_economies_of_scale=True,
        category=CostCategory.MAJOR,
    ),

    "basement_insulation": ECMCostModel(
        ecm_id="basement_insulation",
        name_sv="Källarisolering",
        material_cost=CostEntry(
            value_sek=200,
            unit="SEK/m² basement ceiling",
            source=CostSource.WIKELLS_2024,
            year=2024,
            confidence=0.75,
            notes="Basement ceiling insulation, 100-200mm. Area = footprint."
        ),
        labor_cost=CostEntry(
            value_sek=150,
            unit="SEK/m² basement ceiling",
            source=CostSource.WIKELLS_2024,
            year=2024,
            confidence=0.7,
        ),
        fixed_cost=CostEntry(
            value_sek=15000,
            unit="SEK/building",
            source=CostSource.ESTIMATED,
            year=2024,
            confidence=0.6,
            notes="Preparation, cleanup"
        ),
        lifetime_years=40,
        rot_eligible=True,
        scales_with="roof_area",  # basement ceiling = footprint = roof area
        category=CostCategory.MEDIUM_COST,
    ),

    "thermal_bridge_remediation": ECMCostModel(
        ecm_id="thermal_bridge_remediation",
        name_sv="Köldbryggsåtgärder",
        material_cost=CostEntry(
            value_sek=120,
            unit="SEK/m² floor",
            source=CostSource.BEBO_LONSAMHET_2023,
            year=2023,
            confidence=0.65,
            notes="Insulation at balconies, window reveals, junctions"
        ),
        labor_cost=CostEntry(
            value_sek=100,
            unit="SEK/m² floor",
            source=CostSource.BEBO_LONSAMHET_2023,
            year=2023,
            confidence=0.6,
        ),
        lifetime_years=40,
        rot_eligible=True,
        scales_with="floor_area",
        category=CostCategory.MEDIUM_COST,
    ),

    "building_automation_system": ECMCostModel(
        ecm_id="building_automation_system",
        name_sv="Styr- och övervakningssystem",
        material_cost=CostEntry(
            value_sek=30,
            unit="SEK/m² floor",
            source=CostSource.MARKET_RESEARCH_2025,
            year=2025,
            confidence=0.7,
            notes="BAS/BMS - sensors, controllers, gateway"
        ),
        labor_cost=CostEntry(
            value_sek=25,
            unit="SEK/m² floor",
            source=CostSource.MARKET_RESEARCH_2025,
            year=2025,
            confidence=0.65,
            notes="Installation, programming, commissioning"
        ),
        fixed_cost=CostEntry(
            value_sek=100000,
            unit="SEK/building",
            source=CostSource.ESTIMATED,
            year=2025,
            confidence=0.6,
            notes="Central controller, software licensing, integration"
        ),
        annual_maintenance=CostEntry(
            value_sek=5,
            unit="SEK/m²/year",
            source=CostSource.ESTIMATED,
            year=2025,
            confidence=0.6,
            notes="Software updates, calibration"
        ),
        lifetime_years=15,
        rot_eligible=False,
        scales_with="floor_area",
        has_economies_of_scale=True,
        category=CostCategory.MEDIUM_COST,
    ),

    "ftx_overhaul": ECMCostModel(
        ecm_id="ftx_overhaul",
        name_sv="FTX-översyn och renovering",
        material_cost=CostEntry(
            value_sek=80,
            unit="SEK/m² floor",
            source=CostSource.BEBO_TYPKOSTNADER_2023,
            year=2023,
            confidence=0.7,
            notes="New rotor/heat exchanger, filters, motors"
        ),
        labor_cost=CostEntry(
            value_sek=70,
            unit="SEK/m² floor",
            source=CostSource.BEBO_TYPKOSTNADER_2023,
            year=2023,
            confidence=0.7,
            notes="Disassembly, cleaning, reassembly, testing"
        ),
        fixed_cost=CostEntry(
            value_sek=30000,
            unit="SEK/building",
            source=CostSource.ESTIMATED,
            year=2023,
            confidence=0.6,
            notes="Crane, scaffolding if rooftop unit"
        ),
        lifetime_years=15,
        rot_eligible=True,
        scales_with="floor_area",
        category=CostCategory.MEDIUM_COST,
    ),

    "entrance_door_replacement": ECMCostModel(
        ecm_id="entrance_door_replacement",
        name_sv="Entréportbyte",
        material_cost=CostEntry(
            value_sek=20000,
            unit="SEK/door",
            source=CostSource.WIKELLS_2024,
            year=2024,
            confidence=0.8,
            notes="Insulated entrance door with closer"
        ),
        labor_cost=CostEntry(
            value_sek=8000,
            unit="SEK/door",
            source=CostSource.WIKELLS_2024,
            year=2024,
            confidence=0.75,
            notes="Removal, installation, finishing"
        ),
        lifetime_years=30,
        rot_eligible=True,
        scales_with="unit",
        category=CostCategory.MEDIUM_COST,
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # OPERATIONAL / ZERO-COST ECMs (Added to complete catalog coverage)
    # ═══════════════════════════════════════════════════════════════════════

    "duc_calibration": ECMCostModel(
        ecm_id="duc_calibration",
        name_sv="DUC-kalibrering",
        material_cost=CostEntry(
            value_sek=0,
            unit="SEK/building",
            source=CostSource.ESTIMATED,
            year=2024,
            confidence=0.9,
            notes="Software/settings adjustment, no material cost"
        ),
        labor_cost=CostEntry(
            value_sek=5000,
            unit="SEK/building",
            source=CostSource.ESTIMATED,
            year=2024,
            confidence=0.8,
            notes="Technician time for calibration, 2-4 hours"
        ),
        lifetime_years=5,
        rot_eligible=False,
        scales_with="building",
        category=CostCategory.ZERO_COST,
    ),

    "ventilation_schedule_optimization": ECMCostModel(
        ecm_id="ventilation_schedule_optimization",
        name_sv="Ventilationsschema-optimering",
        material_cost=CostEntry(
            value_sek=0,
            unit="SEK/building",
            source=CostSource.ESTIMATED,
            year=2024,
            confidence=0.9,
            notes="BMS schedule adjustment only"
        ),
        labor_cost=CostEntry(
            value_sek=3000,
            unit="SEK/building",
            source=CostSource.ESTIMATED,
            year=2024,
            confidence=0.8,
            notes="Energy consultant, 1-2 hours"
        ),
        lifetime_years=3,
        rot_eligible=False,
        scales_with="building",
        category=CostCategory.ZERO_COST,
    ),

    "night_setback": ECMCostModel(
        ecm_id="night_setback",
        name_sv="Nattsänkning",
        material_cost=CostEntry(
            value_sek=0,
            unit="SEK/building",
            source=CostSource.ESTIMATED,
            year=2024,
            confidence=0.95,
            notes="Thermostat/BMS setting change only"
        ),
        labor_cost=CostEntry(
            value_sek=2000,
            unit="SEK/building",
            source=CostSource.ESTIMATED,
            year=2024,
            confidence=0.9,
            notes="Quick configuration, 1 hour"
        ),
        lifetime_years=5,
        rot_eligible=False,
        scales_with="building",
        category=CostCategory.ZERO_COST,
    ),

    "summer_bypass": ECMCostModel(
        ecm_id="summer_bypass",
        name_sv="Sommar-bypass (VÅV)",
        material_cost=CostEntry(
            value_sek=0,
            unit="SEK/building",
            source=CostSource.ESTIMATED,
            year=2024,
            confidence=0.9,
            notes="Most FTX systems have bypass, just needs enabling"
        ),
        labor_cost=CostEntry(
            value_sek=1500,
            unit="SEK/building",
            source=CostSource.ESTIMATED,
            year=2024,
            confidence=0.85,
            notes="BMS setting adjustment"
        ),
        lifetime_years=10,
        rot_eligible=False,
        scales_with="building",
        category=CostCategory.ZERO_COST,
    ),

    "hot_water_temperature": ECMCostModel(
        ecm_id="hot_water_temperature",
        name_sv="Varmvattentemperatur-justering",
        material_cost=CostEntry(
            value_sek=0,
            unit="SEK/building",
            source=CostSource.ESTIMATED,
            year=2024,
            confidence=0.95,
            notes="Thermostat adjustment only"
        ),
        labor_cost=CostEntry(
            value_sek=1000,
            unit="SEK/building",
            source=CostSource.ESTIMATED,
            year=2024,
            confidence=0.9,
            notes="Quick adjustment, 30 min"
        ),
        lifetime_years=5,
        rot_eligible=False,
        scales_with="building",
        category=CostCategory.ZERO_COST,
    ),

    "pump_optimization": ECMCostModel(
        ecm_id="pump_optimization",
        name_sv="Pumpoptimering",
        material_cost=CostEntry(
            value_sek=0,
            unit="SEK/building",
            source=CostSource.ESTIMATED,
            year=2024,
            confidence=0.85,
            notes="Frequency adjustment if VFD exists, else minor cost"
        ),
        labor_cost=CostEntry(
            value_sek=3000,
            unit="SEK/building",
            source=CostSource.ESTIMATED,
            year=2024,
            confidence=0.8,
            notes="Balancing and speed optimization"
        ),
        lifetime_years=5,
        rot_eligible=False,
        scales_with="building",
        category=CostCategory.ZERO_COST,
    ),

    "heat_recovery_dhw": ECMCostModel(
        ecm_id="heat_recovery_dhw",
        name_sv="Spillvärmeåtervinning (tappvatten)",
        material_cost=CostEntry(
            value_sek=15000,
            unit="SEK/apartment",
            source=CostSource.MARKET_RESEARCH_2025,
            year=2024,
            confidence=0.7,
            notes="Drain water heat recovery unit per apartment/shower"
        ),
        labor_cost=CostEntry(
            value_sek=5000,
            unit="SEK/apartment",
            source=CostSource.ESTIMATED,
            year=2024,
            confidence=0.65,
            notes="Plumbing installation"
        ),
        lifetime_years=20,
        rot_eligible=True,
        scales_with="apartment",
        category=CostCategory.MEDIUM_COST,
    ),

    "occupancy_sensors": ECMCostModel(
        ecm_id="occupancy_sensors",
        name_sv="Närvarosensorer",
        material_cost=CostEntry(
            value_sek=500,
            unit="SEK/sensor",
            source=CostSource.MARKET_RESEARCH_2025,
            year=2024,
            confidence=0.85,
            notes="PIR sensor for lighting/HVAC control"
        ),
        labor_cost=CostEntry(
            value_sek=300,
            unit="SEK/sensor",
            source=CostSource.ESTIMATED,
            year=2024,
            confidence=0.8,
            notes="Installation and wiring per sensor"
        ),
        lifetime_years=10,
        rot_eligible=False,
        scales_with="common_area_m2",
        category=CostCategory.LOW_COST,
    ),

    "daylight_sensors": ECMCostModel(
        ecm_id="daylight_sensors",
        name_sv="Dagsljussensorer",
        material_cost=CostEntry(
            value_sek=800,
            unit="SEK/sensor",
            source=CostSource.MARKET_RESEARCH_2025,
            year=2024,
            confidence=0.85,
            notes="Lux sensor with dimming control"
        ),
        labor_cost=CostEntry(
            value_sek=400,
            unit="SEK/sensor",
            source=CostSource.ESTIMATED,
            year=2024,
            confidence=0.8,
            notes="Installation, calibration"
        ),
        lifetime_years=10,
        rot_eligible=False,
        scales_with="common_area_m2",
        category=CostCategory.LOW_COST,
    ),

    "predictive_control": ECMCostModel(
        ecm_id="predictive_control",
        name_sv="Prediktiv styrning (MPC)",
        material_cost=CostEntry(
            value_sek=50000,
            unit="SEK/building",
            source=CostSource.MARKET_RESEARCH_2025,
            year=2024,
            confidence=0.6,
            notes="Software license + integration (SaaS or perpetual)"
        ),
        labor_cost=CostEntry(
            value_sek=30000,
            unit="SEK/building",
            source=CostSource.ESTIMATED,
            year=2024,
            confidence=0.5,
            notes="Integration with BMS, commissioning, training"
        ),
        lifetime_years=10,
        rot_eligible=False,
        scales_with="building",
        category=CostCategory.MEDIUM_COST,
    ),

    "recommissioning": ECMCostModel(
        ecm_id="recommissioning",
        name_sv="Funktionstrimning (RCx)",
        material_cost=CostEntry(
            value_sek=0,
            unit="SEK/m²",
            source=CostSource.BEBO_LONSAMHET_2023,
            year=2023,
            confidence=0.8,
            notes="No material, just optimization of existing systems"
        ),
        labor_cost=CostEntry(
            value_sek=20,
            unit="SEK/m²",
            source=CostSource.BEBO_LONSAMHET_2023,
            year=2023,
            confidence=0.75,
            notes="Typical 15-25 SEK/m² for full recommissioning"
        ),
        lifetime_years=5,
        rot_eligible=False,
        scales_with="floor_area",
        category=CostCategory.LOW_COST,
    ),

    "led_common_areas": ECMCostModel(
        ecm_id="led_common_areas",
        name_sv="LED trapphus/gemensamma ytor",
        material_cost=CostEntry(
            value_sek=50,
            unit="SEK/m² common",
            source=CostSource.BEBO_TYPKOSTNADER_2023,
            year=2023,
            confidence=0.85,
            notes="LED fixtures for corridors, stairwells"
        ),
        labor_cost=CostEntry(
            value_sek=30,
            unit="SEK/m² common",
            source=CostSource.ESTIMATED,
            year=2024,
            confidence=0.8,
            notes="Installation per m² common area"
        ),
        lifetime_years=15,
        rot_eligible=False,
        scales_with="common_area_m2",
        category=CostCategory.LOW_COST,
    ),
}


# =============================================================================
# COST CALCULATOR CLASS
# =============================================================================

class SwedishCostCalculatorV2:
    """
    Production-grade cost calculator with Swedish-specific features.

    Usage:
        # For BRF/multi-family (default - NO ROT/green tech)
        calc = SwedishCostCalculatorV2(region=Region.STOCKHOLM)

        # For private homeowner (ROT + green tech eligible)
        calc = SwedishCostCalculatorV2(
            region=Region.STOCKHOLM,
            owner_type=OwnerType.PRIVATE
        )

        cost = calc.calculate_ecm_cost(
            ecm_id="wall_external_insulation",
            quantity=500,  # m² wall
            floor_area_m2=2000,
        )

        print(cost.summary())
        print(f"Total: {cost.total_after_deductions:,.0f} SEK")

    Note:
        - ROT deduction: Only for PRIVATE owners (50% labor, max 50k SEK)
        - Green tech: Only for PRIVATE owners (15% of total)
        - BRF/commercial: Apply for grants separately (Energimyndigheten, Klimatklivet)
    """

    def __init__(
        self,
        region: Region = Region.MEDIUM_CITY,
        year: int = 2025,
        owner_type: OwnerType = OwnerType.BRF,  # Default to BRF (multi-family)
        cost_database: Dict[str, ECMCostModel] = None,
    ):
        self.region = region
        self.year = year
        self.owner_type = owner_type
        self.cost_database = cost_database or ECM_COSTS_V2

    def calculate_ecm_cost(
        self,
        ecm_id: str,
        quantity: float,
        floor_area_m2: float = 1000,
        include_maintenance: bool = False,
        analysis_period_years: int = 25,
    ) -> CostBreakdown:
        """Calculate cost for a single ECM."""
        if ecm_id not in self.cost_database:
            raise ValueError(f"Unknown ECM: {ecm_id}")

        model = self.cost_database[ecm_id]
        return model.calculate_cost(
            quantity=quantity,
            year=self.year,
            region=self.region,
            floor_area_m2=floor_area_m2,
            owner_type=self.owner_type,
            include_maintenance=include_maintenance,
            analysis_period_years=analysis_period_years,
        )

    def calculate_package_cost(
        self,
        ecm_quantities: Dict[str, float],
        floor_area_m2: float = 1000,
        synergy_discount: float = 0.0,  # Package discount
    ) -> Dict[str, CostBreakdown]:
        """Calculate costs for multiple ECMs with potential synergy discount."""
        results = {}

        for ecm_id, quantity in ecm_quantities.items():
            if ecm_id in self.cost_database:
                cost = self.calculate_ecm_cost(
                    ecm_id=ecm_id,
                    quantity=quantity,
                    floor_area_m2=floor_area_m2,
                )
                results[ecm_id] = cost

        # Apply synergy discount to totals if applicable
        # (This is simplified - real synergies are ECM-pair specific)

        return results

    def list_ecms(self) -> List[str]:
        """List all available ECMs."""
        return list(self.cost_database.keys())

    def get_ecm_info(self, ecm_id: str) -> Optional[ECMCostModel]:
        """Get cost model for an ECM."""
        return self.cost_database.get(ecm_id)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_estimate(
    ecm_id: str,
    quantity: float,
    region: str = "medium_city",
    floor_area_m2: float = 1000,
    owner_type: str = "brf",
) -> float:
    """
    Quick cost estimate for an ECM.

    Args:
        ecm_id: ECM identifier
        quantity: Quantity in appropriate units
        region: Swedish region (stockholm, gothenburg, malmo, medium_city, rural, norrland)
        floor_area_m2: Building floor area
        owner_type: Owner type - "brf", "private", "rental", "commercial"
            - "brf"/"rental"/"commercial": Full cost (no ROT/green tech)
            - "private": ROT + green tech deductions apply

    Returns:
        Total cost in SEK (after any applicable deductions)
    """
    owner_map = {
        "private": OwnerType.PRIVATE,
        "brf": OwnerType.BRF,
        "rental": OwnerType.RENTAL,
        "commercial": OwnerType.COMMERCIAL,
    }
    calc = SwedishCostCalculatorV2(
        region=Region(region) if isinstance(region, str) else region,
        owner_type=owner_map.get(owner_type.lower(), OwnerType.BRF),
    )
    try:
        cost = calc.calculate_ecm_cost(ecm_id, quantity, floor_area_m2)
        return cost.total_after_deductions
    except ValueError:
        logger.warning(f"No cost data for ECM: {ecm_id}")
        return 0.0


def compare_costs_by_region(
    ecm_id: str,
    quantity: float,
    floor_area_m2: float = 1000,
) -> Dict[str, float]:
    """Compare costs across Swedish regions."""
    results = {}
    for region in Region:
        calc = SwedishCostCalculatorV2(region=region)
        try:
            cost = calc.calculate_ecm_cost(ecm_id, quantity, floor_area_m2)
            results[region.value] = cost.total_after_deductions
        except ValueError:
            pass
    return results
