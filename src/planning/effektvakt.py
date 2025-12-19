"""
Effektvakt (Power Guard) Optimizer.

Optimizes peak demand reduction using building thermal mass.
Swedish electricity tariffs (e.g., Ellevio) charge per kW of peak demand,
so reducing peaks saves money even without reducing total energy.

Key concepts:
- Effektavgift: Monthly charge based on highest hourly demand (kW)
- Thermal mass: Building stores heat, can pre-heat before peak hours
- Pre-heating: Run heating harder during off-peak, coast through peak
- Staggering: Don't start all heat pumps simultaneously
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class BuildingThermalProperties:
    """Building thermal properties for peak shaving analysis."""
    # From EnergyPlus model or estimates
    total_heat_capacity_kj_k: float      # Total thermal mass (kJ/K)
    heat_loss_coefficient_w_k: float     # UA-value (W/K)
    heated_floor_area_m2: float

    # Heating system
    heating_capacity_kw: float           # Maximum heating output
    heating_type: str = "district"       # district, heat_pump, electric

    # Comfort constraints
    min_indoor_temp_c: float = 20.0
    max_indoor_temp_c: float = 23.0
    target_indoor_temp_c: float = 21.0


@dataclass
class TariffPeakStructure:
    """Peak demand tariff structure."""
    # Electricity (Ellevio example)
    el_peak_sek_kw_month: float = 59.0   # Effektavgift per kW
    el_peak_hours: Tuple[int, int] = (7, 19)  # Peak hours (07:00-19:00)

    # District heating (Stockholm Exergi example)
    fv_peak_sek_kw_year: float = 400.0   # Annual peak charge
    fv_peak_months: List[int] = None     # Peak months (Nov-Mar typically)

    def __post_init__(self):
        if self.fv_peak_months is None:
            self.fv_peak_months = [11, 12, 1, 2, 3]


@dataclass
class PeakShavingResult:
    """Results from peak shaving optimization."""
    # Current state
    current_el_peak_kw: float
    current_fv_peak_kw: float

    # Optimized state
    optimized_el_peak_kw: float
    optimized_fv_peak_kw: float

    # Strategy
    pre_heat_hours: float                # Hours of pre-heating needed
    pre_heat_temp_increase_c: float      # How much to pre-heat
    coast_duration_hours: float          # How long can coast through peak
    stagger_minutes: int                 # Minutes between HP starts

    # Savings
    el_peak_reduction_kw: float
    fv_peak_reduction_kw: float
    annual_el_savings_sek: float
    annual_fv_savings_sek: float
    total_annual_savings_sek: float

    # Implementation notes
    requires_bms: bool                   # Needs BMS for automation
    manual_possible: bool                # Can do manually
    notes: List[str] = None


class EffektvaktOptimizer:
    """
    Optimize peak demand using building thermal mass.

    The building acts as a thermal battery:
    1. Pre-heat before peak hours (charge)
    2. Reduce heating during peak (discharge)
    3. Temperature floats down within comfort band

    Usage:
        optimizer = EffektvaktOptimizer()
        result = optimizer.optimize(
            thermal=building_thermal_props,
            tariff=tariff_structure,
            current_peaks=(150, 200),  # el_kW, fv_kW
        )
    """

    def __init__(self):
        self.outdoor_design_temp_c = -18  # Swedish design day

    def optimize(
        self,
        thermal: BuildingThermalProperties,
        tariff: TariffPeakStructure,
        current_el_peak_kw: float,
        current_fv_peak_kw: float,
        outdoor_temp_c: float = -10,
    ) -> PeakShavingResult:
        """
        Calculate optimal peak shaving strategy.

        Args:
            thermal: Building thermal properties
            tariff: Tariff structure with peak charges
            current_el_peak_kw: Current electricity peak demand
            current_fv_peak_kw: Current district heating peak
            outdoor_temp_c: Outdoor temperature for calculation

        Returns:
            PeakShavingResult with strategy and savings
        """
        # Calculate thermal time constant (hours)
        # τ = C / (UA) where C = heat capacity, UA = heat loss
        time_constant_hours = (
            thermal.total_heat_capacity_kj_k /
            (thermal.heat_loss_coefficient_w_k * 3.6)  # kJ to Wh
        )

        logger.info(f"Building thermal time constant: {time_constant_hours:.1f} hours")

        # Calculate temperature drop rate
        delta_t = thermal.target_indoor_temp_c - outdoor_temp_c
        heat_loss_kw = thermal.heat_loss_coefficient_w_k * delta_t / 1000

        # Pre-heating strategy
        comfort_band = thermal.max_indoor_temp_c - thermal.min_indoor_temp_c
        pre_heat_temp = thermal.max_indoor_temp_c - thermal.target_indoor_temp_c

        # Coast duration (how long can we reduce heating)
        # Using simplified exponential decay
        coast_hours = self._calculate_coast_duration(
            thermal,
            pre_heat_temp,
            outdoor_temp_c,
        )

        # Peak reduction potential
        # During coast, we can reduce heating to minimum (or zero for short periods)
        el_reduction = self._calculate_el_peak_reduction(
            current_el_peak_kw,
            thermal,
            coast_hours,
        )

        fv_reduction = self._calculate_fv_peak_reduction(
            current_fv_peak_kw,
            thermal,
            coast_hours,
        )

        # Calculate savings
        el_savings = el_reduction * tariff.el_peak_sek_kw_month * 12
        fv_savings = fv_reduction * tariff.fv_peak_sek_kw_year

        # Stagger calculation for heat pumps
        stagger_minutes = 0
        if thermal.heating_type == "heat_pump":
            # Typical: 10-15 minutes between compressor starts
            stagger_minutes = 15

        # Implementation notes
        notes = []
        requires_bms = True
        manual_possible = False

        if coast_hours >= 2:
            notes.append(f"Building kan 'segla' {coast_hours:.1f} timmar på termisk massa")
            manual_possible = True

        if pre_heat_temp > 1.5:
            notes.append(f"Förvärm till {thermal.max_indoor_temp_c}°C före höglasttimmar")

        if stagger_minutes > 0:
            notes.append(f"Fördröj värmepumpstarter med {stagger_minutes} minuter")

        if el_reduction > 20:
            notes.append(f"Potentiell effektreduktion: {el_reduction:.0f} kW el")

        return PeakShavingResult(
            current_el_peak_kw=current_el_peak_kw,
            current_fv_peak_kw=current_fv_peak_kw,
            optimized_el_peak_kw=current_el_peak_kw - el_reduction,
            optimized_fv_peak_kw=current_fv_peak_kw - fv_reduction,
            pre_heat_hours=2.0,  # Typical pre-heat duration
            pre_heat_temp_increase_c=pre_heat_temp,
            coast_duration_hours=coast_hours,
            stagger_minutes=stagger_minutes,
            el_peak_reduction_kw=el_reduction,
            fv_peak_reduction_kw=fv_reduction,
            annual_el_savings_sek=el_savings,
            annual_fv_savings_sek=fv_savings,
            total_annual_savings_sek=el_savings + fv_savings,
            requires_bms=requires_bms,
            manual_possible=manual_possible,
            notes=notes,
        )

    def _calculate_coast_duration(
        self,
        thermal: BuildingThermalProperties,
        pre_heat_delta_c: float,
        outdoor_temp_c: float,
    ) -> float:
        """Calculate how long building can coast without heating."""
        # Simplified: time to drop from max to min temp
        # T(t) = T_out + (T_0 - T_out) * exp(-t/τ)

        # Time constant in hours
        tau = thermal.total_heat_capacity_kj_k / (
            thermal.heat_loss_coefficient_w_k * 3.6
        )

        # Initial and final conditions
        t_initial = thermal.max_indoor_temp_c
        t_final = thermal.min_indoor_temp_c
        t_out = outdoor_temp_c

        # Solve for time
        # t = -τ * ln((T_final - T_out) / (T_initial - T_out))
        if t_initial <= t_out or t_final <= t_out:
            return 0.0

        ratio = (t_final - t_out) / (t_initial - t_out)
        if ratio <= 0:
            return 0.0

        coast_hours = -tau * math.log(ratio)
        return max(0, coast_hours)

    def _calculate_el_peak_reduction(
        self,
        current_peak_kw: float,
        thermal: BuildingThermalProperties,
        coast_hours: float,
    ) -> float:
        """Calculate potential electricity peak reduction."""
        if thermal.heating_type != "heat_pump":
            # Only heat pumps contribute to el peak significantly
            return 0.0

        # During coast, can reduce HP output significantly
        if coast_hours >= 1:
            # Can reduce by ~50-70% during peak hours
            reduction_factor = 0.5
        else:
            reduction_factor = 0.2

        return current_peak_kw * reduction_factor

    def _calculate_fv_peak_reduction(
        self,
        current_peak_kw: float,
        thermal: BuildingThermalProperties,
        coast_hours: float,
    ) -> float:
        """Calculate potential district heating peak reduction."""
        if thermal.heating_type != "district":
            return 0.0

        # Pre-heating spreads load, reducing peak
        if coast_hours >= 2:
            reduction_factor = 0.3
        elif coast_hours >= 1:
            reduction_factor = 0.2
        else:
            reduction_factor = 0.1

        return current_peak_kw * reduction_factor


def estimate_thermal_properties(
    atemp_m2: float,
    construction_year: int,
    heating_type: str = "district",
    num_floors: int = 4,
) -> BuildingThermalProperties:
    """
    Estimate thermal properties from basic building data.

    Args:
        atemp_m2: Heated floor area
        construction_year: Year of construction
        heating_type: Type of heating system
        num_floors: Number of floors

    Returns:
        Estimated BuildingThermalProperties
    """
    # Estimate thermal mass based on construction era
    # Swedish buildings: typically concrete/masonry = high thermal mass
    if construction_year < 1960:
        # Brick/masonry: ~400 kJ/(m²·K)
        thermal_mass_per_m2 = 400
    elif construction_year < 1975:
        # Concrete panel (miljonprogrammet): ~350 kJ/(m²·K)
        thermal_mass_per_m2 = 350
    elif construction_year < 2000:
        # Mixed construction: ~300 kJ/(m²·K)
        thermal_mass_per_m2 = 300
    else:
        # Modern lightweight: ~250 kJ/(m²·K)
        thermal_mass_per_m2 = 250

    total_heat_capacity = thermal_mass_per_m2 * atemp_m2

    # Estimate UA-value from era
    if construction_year < 1960:
        ua_per_m2 = 1.5  # W/(m²·K)
    elif construction_year < 1975:
        ua_per_m2 = 1.2
    elif construction_year < 2000:
        ua_per_m2 = 0.9
    else:
        ua_per_m2 = 0.6

    heat_loss_coefficient = ua_per_m2 * atemp_m2

    # Estimate heating capacity (typically 30-50 W/m²)
    heating_capacity = atemp_m2 * 0.040  # 40 W/m² = kW

    return BuildingThermalProperties(
        total_heat_capacity_kj_k=total_heat_capacity,
        heat_loss_coefficient_w_k=heat_loss_coefficient,
        heated_floor_area_m2=atemp_m2,
        heating_capacity_kw=heating_capacity,
        heating_type=heating_type,
    )


def analyze_effektvakt_potential(
    atemp_m2: float,
    construction_year: int,
    heating_type: str = "district",
    current_el_peak_kw: float = 100,
    current_fv_peak_kw: float = 200,
) -> PeakShavingResult:
    """
    Convenience function to analyze effektvakt potential.

    Args:
        atemp_m2: Building heated floor area
        construction_year: Year of construction
        heating_type: Heating system type
        current_el_peak_kw: Current electricity peak
        current_fv_peak_kw: Current district heating peak

    Returns:
        PeakShavingResult with potential savings
    """
    thermal = estimate_thermal_properties(
        atemp_m2=atemp_m2,
        construction_year=construction_year,
        heating_type=heating_type,
    )

    tariff = TariffPeakStructure()  # Default Swedish tariffs

    optimizer = EffektvaktOptimizer()
    return optimizer.optimize(
        thermal=thermal,
        tariff=tariff,
        current_el_peak_kw=current_el_peak_kw,
        current_fv_peak_kw=current_fv_peak_kw,
    )
