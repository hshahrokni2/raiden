"""
Geometry Module - Calculate building geometry from public data.

Extracts and calculates:
- Wall areas per orientation (N/S/E/W) from footprint + height
- Window areas per orientation from WWR detection
- Roof area and PV-available area
- Building volume and thermal mass
- Shading factors from neighboring buildings and trees
"""

from .building_geometry import (
    BuildingGeometry,
    BuildingGeometryCalculator,
    FacadeGeometry,
    RoofGeometry,
    WallSegment,
    calculate_building_geometry,
)
from .pv_potential import PVPotentialCalculator
from .thermal_mass import ThermalMassCalculator

__all__ = [
    'BuildingGeometry',
    'BuildingGeometryCalculator',
    'FacadeGeometry',
    'RoofGeometry',
    'WallSegment',
    'calculate_building_geometry',
    'PVPotentialCalculator',
    'ThermalMassCalculator',
]
