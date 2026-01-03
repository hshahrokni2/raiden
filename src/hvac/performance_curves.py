"""
Temperature-dependent HVAC performance curves for Swedish systems.

Provides COP/efficiency curves that vary with operating conditions:
- Outdoor temperature (ASHP, district heating)
- Ground temperature (GSHP)
- Exhaust air temperature (FTX-VP)
- Part-load ratio

Based on:
- Swedish Energy Agency test data
- Manufacturer specifications (Nibe, Thermia, IVT, CTC)
- SP Technical Research Institute measurements
- BBR 6:42 reference values

EnergyPlus Curve Objects:
- Curve:Biquadratic for 2-variable relationships
- Curve:Quadratic for single-variable relationships
- Curve:Cubic for part-load performance
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import math


class HeatPumpType(Enum):
    """Swedish heat pump types with performance characteristics."""
    GROUND_SOURCE = "ground_source"      # Bergvärme
    EXHAUST_AIR = "exhaust_air"          # FTX-VP
    AIR_SOURCE = "air_source"            # Luft-vatten
    AIR_AIR = "air_air"                  # Luft-luft


@dataclass
class PerformanceCurve:
    """
    Performance curve for temperature-dependent efficiency.

    Represents the relationship:
    COP = COP_nominal * f(T_source, T_sink)

    For EnergyPlus biquadratic curves:
    f(x,y) = c1 + c2*x + c3*x² + c4*y + c5*y² + c6*x*y

    Where x = source temperature (°C), y = sink temperature (°C)
    """
    name: str
    curve_type: str = "biquadratic"  # or "quadratic", "cubic"

    # Biquadratic coefficients
    c1: float = 1.0  # Constant
    c2: float = 0.0  # x coefficient
    c3: float = 0.0  # x² coefficient
    c4: float = 0.0  # y coefficient
    c5: float = 0.0  # y² coefficient
    c6: float = 0.0  # x*y coefficient

    # Valid input ranges
    x_min: float = -20.0
    x_max: float = 35.0
    y_min: float = 20.0
    y_max: float = 60.0

    # Output limits
    output_min: float = 0.5
    output_max: float = 1.5

    def evaluate(self, x: float, y: float = 0.0) -> float:
        """
        Evaluate curve at given conditions.

        Args:
            x: Primary variable (source temp for HP, outdoor temp for boiler)
            y: Secondary variable (sink temp, leaving water temp)

        Returns:
            Performance multiplier (typically 0.5 - 1.5)
        """
        # Clamp inputs to valid range
        x = max(self.x_min, min(self.x_max, x))
        y = max(self.y_min, min(self.y_max, y))

        if self.curve_type == "biquadratic":
            result = (self.c1 + self.c2*x + self.c3*x*x +
                     self.c4*y + self.c5*y*y + self.c6*x*y)
        elif self.curve_type == "quadratic":
            result = self.c1 + self.c2*x + self.c3*x*x
        elif self.curve_type == "cubic":
            result = self.c1 + self.c2*x + self.c3*x*x + self.c4*x*x*x
        else:
            result = 1.0

        # Clamp output
        return max(self.output_min, min(self.output_max, result))

    def to_idf(self) -> str:
        """Generate EnergyPlus IDF curve object."""
        if self.curve_type == "biquadratic":
            return f"""
Curve:Biquadratic,
    {self.name},                         !- Name
    {self.c1},                           !- Coefficient1 Constant
    {self.c2},                           !- Coefficient2 x
    {self.c3},                           !- Coefficient3 x**2
    {self.c4},                           !- Coefficient4 y
    {self.c5},                           !- Coefficient5 y**2
    {self.c6},                           !- Coefficient6 x*y
    {self.x_min},                        !- Minimum Value of x
    {self.x_max},                        !- Maximum Value of x
    {self.y_min},                        !- Minimum Value of y
    {self.y_max},                        !- Maximum Value of y
    {self.output_min},                   !- Minimum Curve Output
    {self.output_max};                   !- Maximum Curve Output
"""
        elif self.curve_type == "quadratic":
            return f"""
Curve:Quadratic,
    {self.name},                         !- Name
    {self.c1},                           !- Coefficient1 Constant
    {self.c2},                           !- Coefficient2 x
    {self.c3},                           !- Coefficient3 x**2
    {self.x_min},                        !- Minimum Value of x
    {self.x_max},                        !- Maximum Value of x
    {self.output_min},                   !- Minimum Curve Output
    {self.output_max};                   !- Maximum Curve Output
"""
        return ""


@dataclass
class HeatPumpPerformance:
    """
    Complete heat pump performance model.

    Combines multiple curves for accurate simulation:
    - Capacity curve: How heating capacity varies with temperature
    - COP curve: How efficiency varies with temperature
    - Part-load curve: How efficiency varies at partial load
    """
    name: str
    hp_type: HeatPumpType

    # Nominal conditions (for Swedish climate)
    nominal_cop: float = 3.5
    nominal_heating_kw: float = 10.0

    # Design temperatures (°C)
    design_source_temp: float = 0.0    # Outdoor for ASHP, ground for GSHP
    design_sink_temp: float = 35.0     # Leaving water temperature

    # Performance curves
    heating_capacity_curve: Optional[PerformanceCurve] = None
    heating_cop_curve: Optional[PerformanceCurve] = None
    part_load_curve: Optional[PerformanceCurve] = None

    def get_cop(
        self,
        source_temp_c: float,
        sink_temp_c: float = 35.0,
        part_load_ratio: float = 1.0
    ) -> float:
        """
        Calculate COP at given operating conditions.

        Args:
            source_temp_c: Source temperature (outdoor air, ground, exhaust)
            sink_temp_c: Sink temperature (leaving water/supply air)
            part_load_ratio: Fraction of full load (0.0 - 1.0)

        Returns:
            COP at these conditions
        """
        # Base COP adjustment for temperature
        cop_multiplier = 1.0
        if self.heating_cop_curve:
            cop_multiplier = self.heating_cop_curve.evaluate(source_temp_c, sink_temp_c)

        # Part-load adjustment
        plr_multiplier = 1.0
        if self.part_load_curve and part_load_ratio < 1.0:
            plr_multiplier = self.part_load_curve.evaluate(part_load_ratio)

        return self.nominal_cop * cop_multiplier * plr_multiplier

    def get_capacity(
        self,
        source_temp_c: float,
        sink_temp_c: float = 35.0
    ) -> float:
        """
        Calculate heating capacity at given conditions.

        Returns:
            Heating capacity in kW
        """
        cap_multiplier = 1.0
        if self.heating_capacity_curve:
            cap_multiplier = self.heating_capacity_curve.evaluate(source_temp_c, sink_temp_c)

        return self.nominal_heating_kw * cap_multiplier


# =============================================================================
# SWEDISH HEAT PUMP PERFORMANCE DATA
# Based on Swedish Energy Agency test standards and manufacturer data
# =============================================================================

# Ground Source Heat Pump (Bergvärme)
# Source: Ground @ 0°C (typical Swedish borehole at 100m depth)
# Sink: Radiator water @ 35°C (low-temp system) to 55°C (traditional)
GSHP_HEATING_CAPACITY_CURVE = PerformanceCurve(
    name="GSHP_HeatingCapacity_fT",
    curve_type="biquadratic",
    # Capacity increases with source temp, decreases with sink temp
    c1=1.0,
    c2=0.015,    # +1.5% per °C source increase
    c3=0.0,
    c4=-0.008,   # -0.8% per °C sink increase
    c5=0.0,
    c6=0.0,
    x_min=-5.0,   # Ground temp range
    x_max=15.0,
    y_min=25.0,   # Leaving water temp range
    y_max=55.0,
    output_min=0.7,
    output_max=1.3,
)

GSHP_HEATING_COP_CURVE = PerformanceCurve(
    name="GSHP_HeatingCOP_fT",
    curve_type="biquadratic",
    # COP increases with source temp, decreases with sink temp
    # Based on Carnot efficiency trends
    c1=1.15,
    c2=0.025,    # +2.5% per °C source increase
    c3=-0.0005,  # Small quadratic decrease at high source temps
    c4=-0.012,   # -1.2% per °C sink increase
    c5=0.0001,
    c6=0.0002,
    x_min=-5.0,
    x_max=15.0,
    y_min=25.0,
    y_max=55.0,
    output_min=0.6,
    output_max=1.4,
)

GSHP_PART_LOAD_CURVE = PerformanceCurve(
    name="GSHP_PartLoad_fPLR",
    curve_type="cubic",
    # Variable speed compressor - good part-load performance
    c1=0.1,      # Minimum efficiency at very low loads
    c2=1.2,      # Linear improvement
    c3=-0.4,     # Some degradation at mid-loads
    c4=0.1,      # Cubic correction
    x_min=0.1,
    x_max=1.0,
    output_min=0.8,
    output_max=1.1,
)


# Air Source Heat Pump (Luft-vatten)
# Source: Outdoor air (-20 to +15°C in Sweden)
# Sink: Radiator water @ 35-55°C
ASHP_HEATING_CAPACITY_CURVE = PerformanceCurve(
    name="ASHP_HeatingCapacity_fT",
    curve_type="biquadratic",
    # Significant capacity drop at low outdoor temps
    c1=0.8,
    c2=0.025,    # +2.5% per °C outdoor increase
    c3=-0.0003,  # Slight curve
    c4=-0.006,   # -0.6% per °C sink increase
    c5=0.0,
    c6=0.0001,
    x_min=-20.0,  # Swedish winter
    x_max=20.0,
    y_min=25.0,
    y_max=55.0,
    output_min=0.4,   # Significant capacity reduction at -20°C
    output_max=1.4,
)

ASHP_HEATING_COP_CURVE = PerformanceCurve(
    name="ASHP_HeatingCOP_fT",
    curve_type="biquadratic",
    # COP drops significantly at low outdoor temps
    # Based on Carnot + defrost cycles
    c1=0.75,
    c2=0.035,    # +3.5% per °C outdoor increase
    c3=-0.0008,  # Quadratic term for realistic curve
    c4=-0.010,   # -1.0% per °C sink increase
    c5=0.0001,
    c6=0.0003,
    x_min=-20.0,
    x_max=20.0,
    y_min=25.0,
    y_max=55.0,
    output_min=0.3,   # Low COP at extreme cold
    output_max=1.5,
)

ASHP_PART_LOAD_CURVE = PerformanceCurve(
    name="ASHP_PartLoad_fPLR",
    curve_type="cubic",
    # Variable speed - moderate part-load performance
    c1=0.15,
    c2=1.1,
    c3=-0.35,
    c4=0.1,
    x_min=0.1,
    x_max=1.0,
    output_min=0.75,
    output_max=1.1,
)


# Exhaust Air Heat Pump (FTX-VP)
# Source: Exhaust air @ ~21°C (indoor temp)
# Sink: Supply air or water
EXHAUST_HP_HEATING_COP_CURVE = PerformanceCurve(
    name="ExhaustHP_HeatingCOP_fT",
    curve_type="biquadratic",
    # Very stable COP since source is indoor air
    c1=1.05,
    c2=0.008,    # Small improvement with warmer exhaust
    c3=0.0,
    c4=-0.010,   # Decrease with higher supply temp
    c5=0.0,
    c6=0.0,
    x_min=18.0,   # Exhaust air temp (indoor)
    x_max=24.0,
    y_min=25.0,   # Supply water/air temp
    y_max=50.0,
    output_min=0.9,
    output_max=1.15,
)


# =============================================================================
# PRE-CONFIGURED HEAT PUMP MODELS
# =============================================================================

SWEDISH_HEAT_PUMPS: Dict[HeatPumpType, HeatPumpPerformance] = {
    HeatPumpType.GROUND_SOURCE: HeatPumpPerformance(
        name="Swedish GSHP (Bergvärme)",
        hp_type=HeatPumpType.GROUND_SOURCE,
        nominal_cop=4.2,              # SCOP for Swedish conditions
        nominal_heating_kw=10.0,      # Typical residential size
        design_source_temp=0.0,       # Ground at 100m
        design_sink_temp=35.0,        # Low-temp radiators
        heating_capacity_curve=GSHP_HEATING_CAPACITY_CURVE,
        heating_cop_curve=GSHP_HEATING_COP_CURVE,
        part_load_curve=GSHP_PART_LOAD_CURVE,
    ),

    HeatPumpType.AIR_SOURCE: HeatPumpPerformance(
        name="Swedish ASHP (Luft-vatten)",
        hp_type=HeatPumpType.AIR_SOURCE,
        nominal_cop=3.2,              # SCOP (lower than GSHP)
        nominal_heating_kw=8.0,
        design_source_temp=7.0,       # A7 rating point
        design_sink_temp=35.0,
        heating_capacity_curve=ASHP_HEATING_CAPACITY_CURVE,
        heating_cop_curve=ASHP_HEATING_COP_CURVE,
        part_load_curve=ASHP_PART_LOAD_CURVE,
    ),

    HeatPumpType.EXHAUST_AIR: HeatPumpPerformance(
        name="Swedish FTX-VP (Exhaust Air HP)",
        hp_type=HeatPumpType.EXHAUST_AIR,
        nominal_cop=3.0,              # Lower due to limited source
        nominal_heating_kw=3.0,       # Limited by exhaust flow
        design_source_temp=21.0,      # Indoor exhaust air
        design_sink_temp=35.0,
        heating_cop_curve=EXHAUST_HP_HEATING_COP_CURVE,
    ),
}


def get_heat_pump_performance(hp_type: HeatPumpType) -> HeatPumpPerformance:
    """Get pre-configured heat pump performance model."""
    return SWEDISH_HEAT_PUMPS.get(hp_type, SWEDISH_HEAT_PUMPS[HeatPumpType.GROUND_SOURCE])


def calculate_seasonal_cop(
    hp_type: HeatPumpType,
    climate_zone: str = "stockholm",
    heating_system: str = "low_temp_radiator"
) -> float:
    """
    Calculate seasonal COP (SCOP) for Swedish climate conditions.

    Args:
        hp_type: Heat pump type
        climate_zone: Swedish climate zone (stockholm, malmö, kiruna, etc.)
        heating_system: Type of heating distribution

    Returns:
        Seasonal COP for heating season
    """
    hp = get_heat_pump_performance(hp_type)

    # Swedish heating degree hours distribution (simplified)
    # Temperature bins and their frequency during heating season
    TEMP_BINS = {
        "stockholm": [
            (-15, 0.05),  # 5% of time at -15°C
            (-10, 0.10),  # 10% at -10°C
            (-5, 0.15),   # 15% at -5°C
            (0, 0.25),    # 25% at 0°C
            (5, 0.25),    # 25% at 5°C
            (10, 0.15),   # 15% at 10°C
            (15, 0.05),   # 5% at 15°C
        ],
        "malmö": [
            (-10, 0.05),
            (-5, 0.10),
            (0, 0.20),
            (5, 0.30),
            (10, 0.25),
            (15, 0.10),
        ],
        "kiruna": [
            (-25, 0.10),
            (-20, 0.15),
            (-15, 0.20),
            (-10, 0.20),
            (-5, 0.15),
            (0, 0.15),
            (5, 0.05),
        ],
    }

    # Sink temperatures for different heating systems
    SINK_TEMPS = {
        "underfloor": 35.0,
        "low_temp_radiator": 40.0,
        "standard_radiator": 50.0,
        "high_temp_radiator": 55.0,
    }

    temp_bins = TEMP_BINS.get(climate_zone, TEMP_BINS["stockholm"])
    sink_temp = SINK_TEMPS.get(heating_system, 40.0)

    # Calculate weighted average COP
    total_cop = 0.0
    total_weight = 0.0

    for outdoor_temp, weight in temp_bins:
        if hp_type == HeatPumpType.GROUND_SOURCE:
            # Ground temperature is more stable
            source_temp = 0.0  # Average ground temp at depth
        elif hp_type == HeatPumpType.EXHAUST_AIR:
            source_temp = 21.0  # Indoor exhaust
        else:
            source_temp = outdoor_temp

        cop = hp.get_cop(source_temp, sink_temp, 0.8)  # Assume 80% avg load
        total_cop += cop * weight
        total_weight += weight

    return total_cop / total_weight if total_weight > 0 else hp.nominal_cop


def generate_idf_performance_curves(hp_type: HeatPumpType) -> str:
    """
    Generate all EnergyPlus curve objects for a heat pump type.

    Returns:
        IDF string with all performance curves
    """
    hp = get_heat_pump_performance(hp_type)

    curves = []
    if hp.heating_capacity_curve:
        curves.append(hp.heating_capacity_curve.to_idf())
    if hp.heating_cop_curve:
        curves.append(hp.heating_cop_curve.to_idf())
    if hp.part_load_curve:
        curves.append(hp.part_load_curve.to_idf())

    return "\n".join(curves)


# =============================================================================
# QUICK REFERENCE: SWEDISH SCOP VALUES
# =============================================================================

# Typical SCOP values for Swedish conditions (Stockholm climate)
SWEDISH_SCOP_REFERENCE = {
    HeatPumpType.GROUND_SOURCE: {
        "low_temp_radiator": 4.5,    # 35°C supply
        "standard_radiator": 3.8,    # 50°C supply
        "high_temp_radiator": 3.2,   # 55°C supply
    },
    HeatPumpType.AIR_SOURCE: {
        "low_temp_radiator": 3.5,
        "standard_radiator": 2.8,
        "high_temp_radiator": 2.3,
    },
    HeatPumpType.EXHAUST_AIR: {
        "low_temp_radiator": 3.2,
        "standard_radiator": 2.7,
        "high_temp_radiator": 2.2,
    },
    HeatPumpType.AIR_AIR: {
        "low_temp_radiator": 3.0,
        "standard_radiator": 2.5,
        "high_temp_radiator": 2.0,
    },
}


def get_scop(hp_type: HeatPumpType, heating_system: str = "standard_radiator") -> float:
    """Get quick SCOP lookup for Swedish conditions."""
    if hp_type in SWEDISH_SCOP_REFERENCE:
        return SWEDISH_SCOP_REFERENCE[hp_type].get(heating_system, 3.0)
    return 3.0  # Default fallback
