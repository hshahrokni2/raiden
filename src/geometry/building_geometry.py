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

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import math


@dataclass
class FacadeGeometry:
    """Geometry for a single facade orientation."""
    orientation: str  # 'N', 'S', 'E', 'W'
    wall_area_m2: float
    window_area_m2: float
    wwr: float  # Window-to-wall ratio (0-1)
    azimuth_deg: float  # Average azimuth angle (0=N, 90=E, 180=S, 270=W)
    length_m: float  # Total facade length for this orientation
    segment_count: int = 1  # Number of wall segments in this direction


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
class WallSegment:
    """Individual wall segment from footprint."""
    start: Tuple[float, float]  # (x, y) in meters
    end: Tuple[float, float]  # (x, y) in meters
    length_m: float
    azimuth_deg: float  # Direction wall faces (outward normal)
    orientation: str  # N, S, E, W


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
    facades: Dict[str, FacadeGeometry]  # Keyed by orientation
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

    # Raw segments (for debugging/visualization)
    wall_segments: List[WallSegment] = field(default_factory=list)


class BuildingGeometryCalculator:
    """
    Calculate building geometry from GeoJSON footprint and metadata.

    Usage:
        calculator = BuildingGeometryCalculator()
        geometry = calculator.calculate(
            footprint_coords=[(lon1, lat1), (lon2, lat2), ...],
            height_m=21.0,
            floors=7,
            wwr_by_orientation={'N': 0.15, 'S': 0.25, 'E': 0.20, 'W': 0.20}
        )
    """

    # Stockholm reference point for local projection
    STOCKHOLM_REF_LAT = 59.3293
    STOCKHOLM_REF_LON = 18.0686

    def __init__(self, reference_lat: float = None, reference_lon: float = None):
        """
        Initialize calculator with optional reference point for coordinate conversion.

        Args:
            reference_lat: Latitude for local projection (default: Stockholm)
            reference_lon: Longitude for local projection (default: Stockholm)
        """
        self.ref_lat = reference_lat or self.STOCKHOLM_REF_LAT
        self.ref_lon = reference_lon or self.STOCKHOLM_REF_LON

    def calculate(
        self,
        footprint_coords: List[Tuple[float, float]],
        height_m: float,
        floors: int,
        wwr_by_orientation: Dict[str, float] = None,
        roof_type: str = 'flat',
        roof_slope_deg: float = 0.0,
    ) -> BuildingGeometry:
        """
        Calculate complete building geometry.

        Args:
            footprint_coords: List of (lon, lat) coordinates in WGS84
            height_m: Building height in meters
            floors: Number of floors
            wwr_by_orientation: WWR for each cardinal direction {'N': 0.15, ...}
            roof_type: 'flat', 'pitched', 'gabled'
            roof_slope_deg: Roof slope in degrees (0 for flat)

        Returns:
            BuildingGeometry with all calculated values
        """
        # Default WWR if not provided
        if wwr_by_orientation is None:
            wwr_by_orientation = {'N': 0.15, 'S': 0.25, 'E': 0.20, 'W': 0.20}

        # 1. Convert WGS84 to local metric coordinates
        local_coords = self._wgs84_to_local(footprint_coords)

        # 2. Ensure polygon is closed
        if local_coords[0] != local_coords[-1]:
            local_coords.append(local_coords[0])

        # 3. Calculate footprint area using Shoelace formula
        footprint_area = self._calculate_polygon_area(local_coords)

        # 4. Calculate perimeter and wall segments
        segments = self._calculate_wall_segments(local_coords)
        perimeter = sum(s.length_m for s in segments)

        # 5. Group segments by cardinal orientation
        segments_by_orientation = self._group_segments_by_orientation(segments)

        # 6. Calculate wall and window areas per orientation
        floor_height = height_m / floors
        facades = {}

        for orientation in ['N', 'S', 'E', 'W']:
            orientation_segments = segments_by_orientation.get(orientation, [])
            total_length = sum(s.length_m for s in orientation_segments)
            wall_area = total_length * height_m

            wwr = wwr_by_orientation.get(orientation, 0.15)
            window_area = wall_area * wwr

            # Calculate average azimuth for this orientation
            if orientation_segments:
                avg_azimuth = sum(s.azimuth_deg for s in orientation_segments) / len(orientation_segments)
            else:
                avg_azimuth = {'N': 0, 'E': 90, 'S': 180, 'W': 270}[orientation]

            facades[orientation] = FacadeGeometry(
                orientation=orientation,
                wall_area_m2=wall_area,
                window_area_m2=window_area,
                wwr=wwr,
                azimuth_deg=avg_azimuth,
                length_m=total_length,
                segment_count=len(orientation_segments),
            )

        # 7. Calculate totals
        total_wall_area = sum(f.wall_area_m2 for f in facades.values())
        total_window_area = sum(f.window_area_m2 for f in facades.values())
        average_wwr = total_window_area / total_wall_area if total_wall_area > 0 else 0

        # 8. Calculate roof geometry
        roof = self._calculate_roof_geometry(
            footprint_area=footprint_area,
            roof_type=roof_type,
            roof_slope_deg=roof_slope_deg,
        )

        # 9. Calculate gross floor area and volume
        gross_floor_area = footprint_area * floors
        volume = footprint_area * height_m

        # 10. Calculate total envelope area
        total_envelope_area = total_wall_area + roof.total_area_m2 + footprint_area

        return BuildingGeometry(
            footprint_area_m2=footprint_area,
            gross_floor_area_m2=gross_floor_area,
            height_m=height_m,
            floors=floors,
            floor_height_m=floor_height,
            facades=facades,
            roof=roof,
            ground_floor_area_m2=footprint_area,
            total_wall_area_m2=total_wall_area,
            total_window_area_m2=total_window_area,
            total_envelope_area_m2=total_envelope_area,
            average_wwr=average_wwr,
            volume_m3=volume,
            perimeter_m=perimeter,
            wall_segments=segments,
        )

    def calculate_from_geojson(
        self,
        geojson: dict,
        height_m: float,
        floors: int,
        wwr_by_orientation: Dict[str, float] = None,
        **kwargs,
    ) -> BuildingGeometry:
        """
        Calculate geometry from a GeoJSON Polygon or Feature.

        Args:
            geojson: GeoJSON dict (Polygon, MultiPolygon, or Feature)
            height_m: Building height
            floors: Number of floors
            wwr_by_orientation: WWR per direction

        Returns:
            BuildingGeometry
        """
        coords = self._parse_geojson(geojson)
        return self.calculate(coords, height_m, floors, wwr_by_orientation, **kwargs)

    def _parse_geojson(self, geojson: dict) -> List[Tuple[float, float]]:
        """Extract coordinate list from GeoJSON."""
        # Handle Feature wrapper
        if geojson.get('type') == 'Feature':
            geojson = geojson.get('geometry', {})

        geom_type = geojson.get('type')
        coords = geojson.get('coordinates', [])

        if geom_type == 'Polygon':
            # First ring is exterior
            if coords and len(coords) > 0:
                return [(c[0], c[1]) for c in coords[0]]
        elif geom_type == 'MultiPolygon':
            # Take first polygon
            if coords and len(coords) > 0 and len(coords[0]) > 0:
                return [(c[0], c[1]) for c in coords[0][0]]

        raise ValueError(f"Unsupported GeoJSON geometry type: {geom_type}")

    def _wgs84_to_local(
        self,
        coords: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """
        Convert WGS84 (lon, lat) coordinates to local metric (x, y) coordinates.

        Uses equirectangular projection centered on the building centroid.
        Accurate enough for building-scale calculations in Sweden.

        Args:
            coords: List of (longitude, latitude) tuples

        Returns:
            List of (x, y) tuples in meters
        """
        if not coords:
            return []

        # Calculate centroid for reference
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        center_lon = sum(lons) / len(lons)
        center_lat = sum(lats) / len(lats)

        # Earth radius in meters
        R = 6371000

        # Convert to local coordinates
        local_coords = []
        for lon, lat in coords:
            # Equirectangular projection
            x = R * math.radians(lon - center_lon) * math.cos(math.radians(center_lat))
            y = R * math.radians(lat - center_lat)
            local_coords.append((x, y))

        return local_coords

    def _calculate_polygon_area(self, coords: List[Tuple[float, float]]) -> float:
        """
        Calculate polygon area using the Shoelace formula.

        Args:
            coords: List of (x, y) coordinates in meters (must be closed)

        Returns:
            Area in square meters (absolute value)
        """
        n = len(coords)
        if n < 3:
            return 0.0

        area = 0.0
        for i in range(n - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            area += x1 * y2 - x2 * y1

        return abs(area) / 2.0

    def _calculate_wall_segments(
        self,
        coords: List[Tuple[float, float]],
    ) -> List[WallSegment]:
        """
        Calculate wall segments from polygon coordinates.

        For each edge, calculates:
        - Length
        - Azimuth (direction the wall faces, i.e., outward normal)
        - Cardinal orientation

        Args:
            coords: List of (x, y) coordinates in meters

        Returns:
            List of WallSegment objects
        """
        segments = []

        for i in range(len(coords) - 1):
            p1 = coords[i]
            p2 = coords[i + 1]

            # Calculate length
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.sqrt(dx * dx + dy * dy)

            if length < 0.1:  # Skip very short segments
                continue

            # Calculate azimuth of the wall segment direction
            # atan2(dx, dy) gives angle from north (y-axis)
            segment_azimuth = math.degrees(math.atan2(dx, dy))
            if segment_azimuth < 0:
                segment_azimuth += 360

            # The wall faces perpendicular to segment direction
            # For a counter-clockwise polygon, the outward normal is 90° clockwise
            wall_azimuth = (segment_azimuth + 90) % 360

            # Determine cardinal orientation
            orientation = self._azimuth_to_orientation(wall_azimuth)

            segments.append(WallSegment(
                start=p1,
                end=p2,
                length_m=length,
                azimuth_deg=wall_azimuth,
                orientation=orientation,
            ))

        return segments

    def _azimuth_to_orientation(self, azimuth: float) -> str:
        """
        Convert azimuth angle to cardinal direction.

        Azimuth: 0° = North, 90° = East, 180° = South, 270° = West

        Args:
            azimuth: Angle in degrees (0-360)

        Returns:
            Cardinal direction: 'N', 'E', 'S', or 'W'
        """
        # Normalize to 0-360
        azimuth = azimuth % 360

        # N: 315-45 (centered on 0)
        # E: 45-135 (centered on 90)
        # S: 135-225 (centered on 180)
        # W: 225-315 (centered on 270)
        if azimuth >= 315 or azimuth < 45:
            return 'N'
        elif 45 <= azimuth < 135:
            return 'E'
        elif 135 <= azimuth < 225:
            return 'S'
        else:  # 225 <= azimuth < 315
            return 'W'

    def _group_segments_by_orientation(
        self,
        segments: List[WallSegment],
    ) -> Dict[str, List[WallSegment]]:
        """Group wall segments by their cardinal orientation."""
        grouped = {'N': [], 'E': [], 'S': [], 'W': []}
        for segment in segments:
            grouped[segment.orientation].append(segment)
        return grouped

    def _calculate_roof_geometry(
        self,
        footprint_area: float,
        roof_type: str,
        roof_slope_deg: float,
    ) -> RoofGeometry:
        """
        Calculate roof geometry for PV potential.

        Args:
            footprint_area: Ground floor area in m²
            roof_type: 'flat', 'pitched', 'gabled'
            roof_slope_deg: Roof slope in degrees

        Returns:
            RoofGeometry object
        """
        if roof_type == 'flat' or roof_slope_deg < 5:
            return RoofGeometry(
                total_area_m2=footprint_area,
                flat_area_m2=footprint_area,
                pitched_area_m2=0,
                primary_slope_deg=0,
                primary_azimuth_deg=180,  # Doesn't matter for flat
                available_pv_area_m2=footprint_area * 0.7,  # 70% usable after setbacks
            )

        elif roof_type == 'pitched':
            # Pitched roof increases total area by cos(slope)
            slope_rad = math.radians(roof_slope_deg)
            total_area = footprint_area / math.cos(slope_rad)
            return RoofGeometry(
                total_area_m2=total_area,
                flat_area_m2=0,
                pitched_area_m2=total_area,
                primary_slope_deg=roof_slope_deg,
                primary_azimuth_deg=180,  # Assume south-facing
                available_pv_area_m2=total_area * 0.5,  # Only south-facing half
            )

        elif roof_type == 'gabled':
            # Gabled roof: two pitched surfaces
            slope_rad = math.radians(roof_slope_deg)
            total_area = footprint_area / math.cos(slope_rad)
            return RoofGeometry(
                total_area_m2=total_area,
                flat_area_m2=0,
                pitched_area_m2=total_area,
                primary_slope_deg=roof_slope_deg,
                primary_azimuth_deg=180,
                available_pv_area_m2=total_area * 0.4,  # Less usable due to orientation
            )

        else:
            # Default to flat
            return RoofGeometry(
                total_area_m2=footprint_area,
                flat_area_m2=footprint_area,
                pitched_area_m2=0,
                primary_slope_deg=0,
                primary_azimuth_deg=180,
                available_pv_area_m2=footprint_area * 0.7,
            )


def calculate_building_geometry(
    footprint_coords: List[Tuple[float, float]],
    height_m: float,
    floors: int,
    wwr_by_orientation: Dict[str, float] = None,
) -> BuildingGeometry:
    """
    Convenience function to calculate building geometry.

    Args:
        footprint_coords: List of (lon, lat) in WGS84
        height_m: Building height
        floors: Number of floors
        wwr_by_orientation: WWR per direction

    Returns:
        BuildingGeometry object
    """
    calculator = BuildingGeometryCalculator()
    return calculator.calculate(footprint_coords, height_m, floors, wwr_by_orientation)


# =============================================================================
# BUILDING COMPLEXITY DETECTION
# =============================================================================

@dataclass
class BuildingComplexity:
    """
    Building complexity assessment for multi-zone model triggering.

    Determines whether a single-zone thermal model is adequate or if
    multi-zone modeling is needed for accurate energy simulation.
    """
    # Complexity score (0-100)
    # 0-30: Simple (single-zone OK)
    # 31-60: Moderate (single-zone with caution)
    # 61-100: Complex (multi-zone recommended)
    complexity_score: float

    # Recommendation
    recommended_zones: int  # 1 = single zone, 2+ = multi-zone
    single_zone_adequate: bool
    confidence: float  # 0-1

    # Contributing factors
    aspect_ratio: float  # Length/width ratio
    num_corners: int  # Number of vertices (4 = rectangular)
    convexity: float  # 0-1 (1 = fully convex)
    floor_area_m2: float  # Per floor
    has_atrium: bool
    num_orientations: int  # How many distinct facade directions

    # Warning message
    warning: str
    warning_sv: str  # Swedish


def assess_building_complexity(
    footprint_coords: List[Tuple[float, float]],
    gross_floor_area_m2: float,
    floors: int,
) -> BuildingComplexity:
    """
    Assess building complexity for thermal zoning decisions.

    Uses multiple heuristics to determine if single-zone modeling is adequate:
    1. Aspect ratio: Long/thin buildings need multiple zones
    2. Corners: Non-rectangular shapes need zone decomposition
    3. Convexity: L/U/T shapes have internal corners
    4. Floor area: Large floors have temperature gradients
    5. Orientations: Multiple distinct facade directions

    Args:
        footprint_coords: List of (lon, lat) in WGS84
        gross_floor_area_m2: Total floor area (Atemp)
        floors: Number of floors

    Returns:
        BuildingComplexity assessment
    """
    # Convert to local meters
    calc = BuildingGeometryCalculator()
    local_coords = calc._wgs84_to_local(footprint_coords)

    # Basic metrics
    num_corners = len(local_coords) - 1  # Subtract 1 if closed polygon
    if local_coords[0] == local_coords[-1]:
        num_corners = len(local_coords) - 1

    floor_area_m2 = gross_floor_area_m2 / floors if floors > 0 else gross_floor_area_m2

    # Calculate bounding box for aspect ratio
    xs = [p[0] for p in local_coords]
    ys = [p[1] for p in local_coords]
    width = max(xs) - min(xs)
    length = max(ys) - min(ys)
    aspect_ratio = max(width, length) / max(min(width, length), 0.1)

    # Calculate convexity (area of convex hull / actual area)
    footprint_area = calc._calculate_polygon_area(local_coords)
    convex_hull_area = _calculate_convex_hull_area(local_coords)
    convexity = footprint_area / convex_hull_area if convex_hull_area > 0 else 1.0

    # Check for atrium (internal void)
    has_atrium = convexity < 0.8  # Significant interior void

    # Count distinct orientations
    segments = calc._calculate_wall_segments(local_coords)
    orientations = set(s.orientation for s in segments)
    num_orientations = len(orientations)

    # Calculate complexity score
    score = 0.0

    # Aspect ratio contribution (0-25 points)
    # Rectangular = 1.0, Long = 3+
    if aspect_ratio > 3.0:
        score += 25
    elif aspect_ratio > 2.0:
        score += 15
    elif aspect_ratio > 1.5:
        score += 5

    # Corners contribution (0-25 points)
    # 4 = rectangular, 6+ = complex shape
    if num_corners > 8:
        score += 25
    elif num_corners > 6:
        score += 20
    elif num_corners > 4:
        score += 10

    # Convexity contribution (0-25 points)
    # 1.0 = convex, <0.8 = significant concavity (L/U/T shape)
    if convexity < 0.7:
        score += 25
    elif convexity < 0.8:
        score += 15
    elif convexity < 0.9:
        score += 5

    # Floor area contribution (0-25 points)
    # Small floors = uniform temp, Large floors = gradients
    if floor_area_m2 > 3000:
        score += 25
    elif floor_area_m2 > 1500:
        score += 15
    elif floor_area_m2 > 800:
        score += 5

    # Clamp to 0-100
    score = max(0, min(100, score))

    # Determine recommendation
    if score <= 30:
        single_zone_adequate = True
        recommended_zones = 1
        confidence = 0.9
        warning = ""
        warning_sv = ""
    elif score <= 60:
        single_zone_adequate = True
        recommended_zones = 1
        confidence = 0.7
        warning = (
            f"Moderate building complexity (score: {score:.0f}/100). "
            f"Single-zone model may underestimate orientation effects. "
            f"Consider multi-zone for detailed analysis."
        )
        warning_sv = (
            f"Måttlig byggnadskomplexitet (poäng: {score:.0f}/100). "
            f"Enzonmodell kan underskatta orienteringseffekter. "
            f"Överväg flerzonmodell för detaljerad analys."
        )
    else:
        single_zone_adequate = False
        recommended_zones = max(2, min(floors, 4))  # At least 2, max 4
        confidence = 0.5
        warning = (
            f"High building complexity (score: {score:.0f}/100). "
            f"Single-zone model may have ±15-25% error. "
            f"Factors: aspect ratio {aspect_ratio:.1f}, {num_corners} corners, "
            f"convexity {convexity:.2f}, floor area {floor_area_m2:.0f} m². "
            f"Multi-zone modeling recommended for accurate results."
        )
        warning_sv = (
            f"Hög byggnadskomplexitet (poäng: {score:.0f}/100). "
            f"Enzonmodell kan ha ±15-25% fel. "
            f"Faktorer: sidoförhållande {aspect_ratio:.1f}, {num_corners} hörn, "
            f"konvexitet {convexity:.2f}, golvyta {floor_area_m2:.0f} m². "
            f"Flerzonmodellering rekommenderas för noggranna resultat."
        )

    return BuildingComplexity(
        complexity_score=score,
        recommended_zones=recommended_zones,
        single_zone_adequate=single_zone_adequate,
        confidence=confidence,
        aspect_ratio=aspect_ratio,
        num_corners=num_corners,
        convexity=convexity,
        floor_area_m2=floor_area_m2,
        has_atrium=has_atrium,
        num_orientations=num_orientations,
        warning=warning,
        warning_sv=warning_sv,
    )


def _calculate_convex_hull_area(coords: List[Tuple[float, float]]) -> float:
    """
    Calculate area of convex hull using Graham scan.

    Simple implementation for polygon convex hull.
    """
    if len(coords) < 3:
        return 0.0

    # Find lowest point
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    points = sorted(set(coords))
    if len(points) < 3:
        return 0.0

    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenate hulls
    hull = lower[:-1] + upper[:-1]

    # Calculate area using shoelace formula
    n = len(hull)
    if n < 3:
        return 0.0

    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += hull[i][0] * hull[j][1]
        area -= hull[j][0] * hull[i][1]

    return abs(area) / 2.0
