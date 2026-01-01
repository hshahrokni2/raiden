"""
ROI Module - Calculate return on investment for ECMs.

Features:
- Swedish energy prices
- Swedish ECM costs
- Effekttariff (power demand tariff) - 2025+
- Payback period calculation
- NPV and IRR
- Package recommendations
"""

from .costs_sweden import SwedishCosts
from .calculator import ROICalculator, ROIResult
from .costs_sweden_v2 import (
    # Effekttariff (power demand tariff)
    EffektTariff,
    ELLEVIO_EFFEKTTARIFF,
    EFFEKT_TARIFFS,
    get_effekt_tariff,
    # Building peak estimation
    BuildingPeakEstimate,
    estimate_building_peak_power,
    # ECM peak impacts
    ECM_PEAK_IMPACTS,
    calculate_ecm_peak_savings,
    calculate_combined_peak_savings,
    # Battery viability (Sweden 2025 market conditions)
    BatteryViabilityResult,
    evaluate_battery_viability,
    # Cost calculator
    SwedishCostCalculatorV2,
)

__all__ = [
    'SwedishCosts',
    'ROICalculator',
    'ROIResult',
    # Effekttariff
    'EffektTariff',
    'ELLEVIO_EFFEKTTARIFF',
    'EFFEKT_TARIFFS',
    'get_effekt_tariff',
    # Peak estimation
    'BuildingPeakEstimate',
    'estimate_building_peak_power',
    # ECM peak impacts
    'ECM_PEAK_IMPACTS',
    'calculate_ecm_peak_savings',
    'calculate_combined_peak_savings',
    # Battery viability
    'BatteryViabilityResult',
    'evaluate_battery_viability',
    # V2 calculator
    'SwedishCostCalculatorV2',
]
