"""
Coordinate transformation utilities.

Handles conversion between:
- SWEREF99 TM (EPSG:3006) - Swedish national grid, used in Lantmäteriet data
- WGS84 (EPSG:4326) - GPS coordinates, used for web mapping
- Local coordinates - For EnergyPlus (origin at building centroid)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from pyproj import CRS, Transformer
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import transform
import numpy as np


@dataclass
class BoundingBox:
    """Bounding box in any coordinate system."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.min_x + self.max_x) / 2, (self.min_y + self.max_y) / 2)

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    def to_tuple(self) -> tuple[float, float, float, float]:
        """Return as (min_x, min_y, max_x, max_y)."""
        return (self.min_x, self.min_y, self.max_x, self.max_y)

    def to_overture_bbox(self) -> str:
        """Format for Overture Maps CLI (west,south,east,north in WGS84)."""
        return f"{self.min_x},{self.min_y},{self.max_x},{self.max_y}"


class CoordinateTransformer:
    """
    Transform coordinates between Swedish and global coordinate systems.

    Primary systems:
    - SWEREF99 TM (EPSG:3006): Swedish national grid
    - WGS84 (EPSG:4326): Global lat/lon for web maps
    - Local: Relative to building centroid for EnergyPlus
    """

    # Common CRS definitions
    SWEREF99_TM = "EPSG:3006"
    WGS84 = "EPSG:4326"

    def __init__(self):
        # Pre-create common transformers for efficiency
        self._sweref_to_wgs84 = Transformer.from_crs(
            self.SWEREF99_TM, self.WGS84, always_xy=True
        )
        self._wgs84_to_sweref = Transformer.from_crs(
            self.WGS84, self.SWEREF99_TM, always_xy=True
        )

    def sweref_to_wgs84(
        self, x: float | Sequence[float], y: float | Sequence[float]
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        """
        Convert SWEREF99 TM to WGS84.

        Args:
            x: Easting(s) in SWEREF99 TM (meters)
            y: Northing(s) in SWEREF99 TM (meters)

        Returns:
            (longitude, latitude) in WGS84 (degrees)
        """
        return self._sweref_to_wgs84.transform(x, y)

    def wgs84_to_sweref(
        self, lon: float | Sequence[float], lat: float | Sequence[float]
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        """
        Convert WGS84 to SWEREF99 TM.

        Args:
            lon: Longitude(s) in WGS84 (degrees)
            lat: Latitude(s) in WGS84 (degrees)

        Returns:
            (easting, northing) in SWEREF99 TM (meters)
        """
        return self._wgs84_to_sweref.transform(lon, lat)

    def coords_3d_to_wgs84(
        self, coords_3d: list[list[float]]
    ) -> list[tuple[float, float, float]]:
        """
        Convert 3D coordinates from SWEREF99 TM to WGS84.

        Args:
            coords_3d: List of [x, y, z] in SWEREF99 TM

        Returns:
            List of (lon, lat, z) in WGS84
        """
        result = []
        for coord in coords_3d:
            x, y, z = coord[0], coord[1], coord[2]
            lon, lat = self.sweref_to_wgs84(x, y)
            result.append((lon, lat, z))
        return result

    def coords_3d_to_2d_wgs84(
        self, coords_3d: list[list[float]]
    ) -> list[tuple[float, float]]:
        """
        Convert 3D SWEREF99 TM coordinates to 2D WGS84 (drop z).

        Args:
            coords_3d: List of [x, y, z] in SWEREF99 TM

        Returns:
            List of (lon, lat) in WGS84
        """
        result = []
        for coord in coords_3d:
            x, y = coord[0], coord[1]
            lon, lat = self.sweref_to_wgs84(x, y)
            result.append((lon, lat))
        return result

    def coords_to_local(
        self,
        coords: list[list[float]] | list[tuple[float, float]],
        origin: tuple[float, float] | None = None,
    ) -> list[tuple[float, float]]:
        """
        Convert coordinates to local system (relative to origin).

        For EnergyPlus, we need coordinates relative to building origin.

        Args:
            coords: List of [x, y] or [x, y, z] coordinates
            origin: (x, y) of local origin. If None, uses centroid.

        Returns:
            List of (local_x, local_y) coordinates
        """
        # Extract x, y from coords (may be 2D or 3D)
        xy_coords = [(c[0], c[1]) for c in coords]

        if origin is None:
            # Calculate centroid as origin
            xs = [c[0] for c in xy_coords]
            ys = [c[1] for c in xy_coords]
            origin = (sum(xs) / len(xs), sum(ys) / len(ys))

        # Translate to local coordinates
        return [(x - origin[0], y - origin[1]) for x, y in xy_coords]

    def get_bounding_box_sweref(
        self, coords: list[list[float]]
    ) -> BoundingBox:
        """Get bounding box in SWEREF99 TM."""
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        return BoundingBox(
            min_x=min(xs), min_y=min(ys), max_x=max(xs), max_y=max(ys)
        )

    def get_bounding_box_wgs84(
        self, coords: list[list[float]], buffer_meters: float = 100
    ) -> BoundingBox:
        """
        Get bounding box in WGS84 with optional buffer.

        Args:
            coords: Coordinates in SWEREF99 TM
            buffer_meters: Buffer around bbox in meters

        Returns:
            BoundingBox in WGS84 (lon/lat)
        """
        bbox_sweref = self.get_bounding_box_sweref(coords)

        # Apply buffer in meters (before transformation)
        min_x = bbox_sweref.min_x - buffer_meters
        min_y = bbox_sweref.min_y - buffer_meters
        max_x = bbox_sweref.max_x + buffer_meters
        max_y = bbox_sweref.max_y + buffer_meters

        # Transform corners
        min_lon, min_lat = self.sweref_to_wgs84(min_x, min_y)
        max_lon, max_lat = self.sweref_to_wgs84(max_x, max_y)

        return BoundingBox(
            min_x=min_lon, min_y=min_lat, max_x=max_lon, max_y=max_lat
        )

    def polygon_sweref_to_wgs84(self, polygon: Polygon) -> Polygon:
        """Transform a Shapely polygon from SWEREF99 TM to WGS84."""
        return transform(self._sweref_to_wgs84.transform, polygon)

    def polygon_wgs84_to_sweref(self, polygon: Polygon) -> Polygon:
        """Transform a Shapely polygon from WGS84 to SWEREF99 TM."""
        return transform(self._wgs84_to_sweref.transform, polygon)

    def create_polygon_from_coords(
        self, coords: list[list[float]], close: bool = True
    ) -> Polygon:
        """
        Create a Shapely Polygon from coordinate list.

        Args:
            coords: List of [x, y] or [x, y, z] coordinates
            close: Whether to close the polygon if not already closed

        Returns:
            Shapely Polygon
        """
        xy_coords = [(c[0], c[1]) for c in coords]

        # Ensure polygon is closed
        if close and xy_coords[0] != xy_coords[-1]:
            xy_coords.append(xy_coords[0])

        return Polygon(xy_coords)

    def calculate_area_sqm(self, coords: list[list[float]]) -> float:
        """
        Calculate polygon area in square meters.

        Coordinates should be in SWEREF99 TM (already in meters).
        """
        polygon = self.create_polygon_from_coords(coords)
        return polygon.area

    def calculate_perimeter_m(self, coords: list[list[float]]) -> float:
        """Calculate polygon perimeter in meters."""
        polygon = self.create_polygon_from_coords(coords)
        return polygon.length

    def calculate_centroid(
        self, coords: list[list[float]]
    ) -> tuple[float, float]:
        """Calculate centroid of polygon in same coordinate system."""
        polygon = self.create_polygon_from_coords(coords)
        return (polygon.centroid.x, polygon.centroid.y)

    def calculate_centroid_wgs84(
        self, coords: list[list[float]]
    ) -> tuple[float, float]:
        """Calculate centroid and return in WGS84."""
        centroid = self.calculate_centroid(coords)
        return self.sweref_to_wgs84(centroid[0], centroid[1])

    def get_facade_orientations(
        self, coords: list[list[float]]
    ) -> list[dict]:
        """
        Calculate facade orientations for each edge of the building footprint.

        Returns:
            List of dicts with 'start', 'end', 'azimuth', 'length', 'direction'
        """
        facades = []
        xy_coords = [(c[0], c[1]) for c in coords]

        for i in range(len(xy_coords) - 1):
            start = xy_coords[i]
            end = xy_coords[i + 1]

            # Calculate vector
            dx = end[0] - start[0]
            dy = end[1] - start[1]

            # Calculate length
            length = math.sqrt(dx * dx + dy * dy)

            if length < 0.1:  # Skip tiny edges
                continue

            # Calculate azimuth (bearing from north, clockwise)
            # Facade normal is perpendicular to edge
            azimuth = math.degrees(math.atan2(dx, dy))

            # Normalize to 0-360
            if azimuth < 0:
                azimuth += 360

            # Determine cardinal direction for the facade normal
            # (perpendicular to edge, pointing outward)
            facade_azimuth = (azimuth + 90) % 360
            direction = self._azimuth_to_direction(facade_azimuth)

            facades.append({
                "start": start,
                "end": end,
                "edge_azimuth": azimuth,
                "facade_azimuth": facade_azimuth,
                "length_m": length,
                "direction": direction,
            })

        return facades

    @staticmethod
    def _azimuth_to_direction(azimuth: float) -> str:
        """Convert azimuth angle to cardinal direction."""
        # Azimuth is degrees from north, clockwise
        if azimuth >= 337.5 or azimuth < 22.5:
            return "north"
        elif azimuth < 67.5:
            return "northeast"
        elif azimuth < 112.5:
            return "east"
        elif azimuth < 157.5:
            return "southeast"
        elif azimuth < 202.5:
            return "south"
        elif azimuth < 247.5:
            return "southwest"
        elif azimuth < 292.5:
            return "west"
        else:
            return "northwest"


def coords_to_geojson_polygon(
    coords: list[list[float]], crs: str = "EPSG:3006"
) -> dict:
    """
    Convert coordinate list to GeoJSON Polygon.

    Args:
        coords: List of [x, y] or [x, y, z] coordinates
        crs: Coordinate reference system

    Returns:
        GeoJSON dict
    """
    transformer = CoordinateTransformer()

    if crs == "EPSG:3006":
        # Transform to WGS84 for GeoJSON
        wgs84_coords = transformer.coords_3d_to_2d_wgs84(coords)
    else:
        wgs84_coords = [(c[0], c[1]) for c in coords]

    # Ensure closed
    if wgs84_coords[0] != wgs84_coords[-1]:
        wgs84_coords.append(wgs84_coords[0])

    return {
        "type": "Polygon",
        "coordinates": [wgs84_coords],
    }


def coords_to_threejs_vertices(
    coords: list[list[float]], height: float, local_origin: tuple[float, float] | None = None
) -> dict:
    """
    Convert building coordinates to Three.js compatible vertex data.

    Args:
        coords: Footprint coordinates [x, y, z] in SWEREF99 TM
        height: Building height in meters
        local_origin: Optional origin for local coordinates

    Returns:
        Dict with 'vertices', 'faces' for Three.js BufferGeometry
    """
    transformer = CoordinateTransformer()

    # Convert to local coordinates (centered at building)
    local_coords = transformer.coords_to_local(coords, local_origin)

    # Remove last coord if it's a duplicate (closed polygon)
    if local_coords[0] == local_coords[-1]:
        local_coords = local_coords[:-1]

    n_vertices = len(local_coords)

    # Create vertices: bottom ring + top ring
    vertices = []
    for x, y in local_coords:
        vertices.extend([x, 0, y])  # Bottom (y=0, using z for depth in Three.js)
    for x, y in local_coords:
        vertices.extend([x, height, y])  # Top

    # Create faces (triangles)
    # Bottom face (reversed winding for outward normals)
    bottom_indices = list(range(n_vertices))
    # Top face
    top_indices = list(range(n_vertices, 2 * n_vertices))

    # Wall faces
    wall_faces = []
    for i in range(n_vertices):
        next_i = (i + 1) % n_vertices
        # Two triangles per wall segment
        # Bottom-left, bottom-right, top-right
        wall_faces.extend([i, next_i, n_vertices + next_i])
        # Bottom-left, top-right, top-left
        wall_faces.extend([i, n_vertices + next_i, n_vertices + i])

    return {
        "vertices": vertices,
        "wall_indices": wall_faces,
        "n_base_vertices": n_vertices,
        "height": height,
        "local_origin": local_origin,
    }
