"""
Building Geometry Calculator

Calculates physical geometry from OSM/Overture footprint data:
- Wall areas per cardinal orientation (N/S/E/W)
- Window areas per orientation (using WWR from Mapillary)
- Floor areas per level
- Envelope areas (walls, roof, ground floor)

Input: GeoJSON footprint, height, floors, WWR per facade
Output: BuildingGeometry object with all calculated areas
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math


@dataclass
class FacadeGeometry:
    """Geometry for a single facade orientation."""
    orientation: str  # 'N', 'S', 'E', 'W', or angle in degrees
    wall_area_m2: float
    window_area_m2: float
    wwr: float  # Window-to-wall ratio (0-1)
    azimuth_deg: float  # True azimuth angle
    length_m: float  # Facade length


@dataclass
class RoofGeometry:
    """Roof geometry for PV calculations."""
    total_area_m2: float
    flat_area_m2: float  # Area suitable for flat PV
    pitched_area_m2: float  # Area on pitched sections
    primary_slope_deg: float  # Main roof slope
    primary_azimuth_deg: float  # Main roof orientation
    available_pv_area_m2: float  # After setbacks, obstructions


@dataclass
class BuildingGeometry:
    """Complete building geometry."""
    # Basic dimensions
    footprint_area_m2: float
    gross_floor_area_m2: float  # Atemp
    height_m: float
    floors: int
    floor_height_m: float

    # Envelope
    facades: List[FacadeGeometry]
    roof: RoofGeometry
    ground_floor_area_m2: float

    # Totals
    total_wall_area_m2: float
    total_window_area_m2: float
    total_envelope_area_m2: float
    average_wwr: float

    # Volume
    volume_m3: float

    # Perimeter
    perimeter_m: float


class BuildingGeometryCalculator:
    """
    Calculate building geometry from GeoJSON footprint and metadata.

    Usage:
        calculator = BuildingGeometryCalculator()
        geometry = calculator.calculate(
            footprint_geojson=geojson_polygon,
            height_m=21.0,
            floors=7,
            wwr_by_orientation={'N': 0.15, 'S': 0.25, 'E': 0.20, 'W': 0.20}
        )
    """

    def __init__(self):
        pass

    def calculate(
        self,
        footprint_geojson: dict,
        height_m: float,
        floors: int,
        wwr_by_orientation: Dict[str, float],
        roof_type: str = 'flat',
        roof_slope_deg: float = 0.0
    ) -> BuildingGeometry:
        """
        Calculate complete building geometry.

        Args:
            footprint_geojson: GeoJSON Polygon or MultiPolygon
            height_m: Building height in meters
            floors: Number of floors
            wwr_by_orientation: WWR for each cardinal direction
            roof_type: 'flat', 'pitched', 'gabled'
            roof_slope_deg: Roof slope in degrees (0 for flat)

        Returns:
            BuildingGeometry with all calculated values
        """
        # TODO: Implement
        # 1. Parse GeoJSON coordinates
        # 2. Calculate footprint area using Shoelace formula
        # 3. Calculate perimeter and segment lengths
        # 4. Assign segments to cardinal orientations based on azimuth
        # 5. Calculate wall areas = segment_length × height
        # 6. Calculate window areas = wall_area × wwr
        # 7. Calculate roof geometry
        # 8. Sum up totals
        raise NotImplementedError("Implement geometry calculation")

    def _parse_footprint(self, geojson: dict) -> List[Tuple[float, float]]:
        """Extract coordinate list from GeoJSON."""
        raise NotImplementedError()

    def _calculate_polygon_area(self, coords: List[Tuple[float, float]]) -> float:
        """Calculate area using Shoelace formula."""
        raise NotImplementedError()

    def _segment_to_azimuth(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate azimuth angle of a wall segment."""
        raise NotImplementedError()

    def _azimuth_to_orientation(self, azimuth: float) -> str:
        """Convert azimuth to cardinal direction (N/S/E/W)."""
        # N: 315-45, E: 45-135, S: 135-225, W: 225-315
        raise NotImplementedError()
