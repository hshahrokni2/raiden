"""
ROI Module - Calculate return on investment for ECMs.

Features:
- Swedish energy prices
- Swedish ECM costs
- Payback period calculation
- NPV and IRR
- Package recommendations
"""

from .costs_sweden import SwedishCosts
from .calculator import ROICalculator, ROIResult

__all__ = ['SwedishCosts', 'ROICalculator', 'ROIResult']
