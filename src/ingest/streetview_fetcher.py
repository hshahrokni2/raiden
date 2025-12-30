"""
Google Street View Facade Fetcher

Fetches Street View images for each building facade using the footprint
to calculate optimal camera positions.

Algorithm:
1. Parse building footprint polygon
2. For each cardinal direction (N/E/S/W), find facade segments
3. Calculate camera position: offset from facade centroid, looking toward building
4. Fetch Street View image with calculated heading
"""

import os
import math
import requests
from io import BytesIO
from PIL import Image
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import json

from rich.console import Console

console = Console()


@dataclass
class FacadeViewpoint:
    """Camera viewpoint for a facade."""
    orientation: str  # N, E, S, W
    camera_lat: float
    camera_lon: float
    heading: float  # Compass heading camera should point
    facade_centroid: Tuple[float, float]  # (lat, lon) of facade center
    facade_length_m: float


@dataclass
class StreetViewImage:
    """Street View image with metadata."""
    orientation: str
    image: Image.Image
    camera_lat: float
    camera_lon: float
    heading: float
    pano_id: Optional[str] = None
    # For geometric height estimation
    pitch: float = 0.0  # Camera tilt: positive = looking up, negative = looking down
    fov: float = 90.0   # Field of view in degrees (default 90°)


class StreetViewFacadeFetcher:
    """
    Fetch Street View images for each building facade.

    Uses the building footprint to calculate optimal camera positions
    for viewing each facade direction.
    """

    # Distance from facade to place camera (meters)
    CAMERA_OFFSET_M = 30.0

    # Earth radius for coordinate calculations
    EARTH_RADIUS_M = 6371000

    def __init__(self, api_key: str = None):
        """
        Initialize with Google API key.

        Args:
            api_key: Google Cloud API key with Street View Static API enabled
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("BRF_GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key required. Set GOOGLE_API_KEY env var.")

    def fetch_facade_images(
        self,
        footprint: dict,
        image_size: str = "640x480",
        fov: int = 90,
        pitch: int = 25,  # Look up at building (higher = more upward tilt)
    ) -> Dict[str, StreetViewImage]:
        """
        Fetch Street View images for all building facades.

        Args:
            footprint: GeoJSON geometry (Polygon or Feature)
            image_size: Image dimensions (e.g., "640x480")
            fov: Field of view in degrees
            pitch: Camera pitch in degrees (positive = up)

        Returns:
            Dict mapping orientation ('N', 'E', 'S', 'W') to StreetViewImage
        """
        # Calculate viewpoints from footprint
        viewpoints = self._calculate_facade_viewpoints(footprint)

        # Fetch images for each viewpoint
        images = {}
        for viewpoint in viewpoints:
            console.print(f"  Fetching {viewpoint.orientation} facade (heading={viewpoint.heading:.0f}°)...")

            img = self._fetch_streetview_image(
                lat=viewpoint.camera_lat,
                lon=viewpoint.camera_lon,
                heading=viewpoint.heading,
                size=image_size,
                fov=fov,
                pitch=pitch,
            )

            if img:
                images[viewpoint.orientation] = StreetViewImage(
                    orientation=viewpoint.orientation,
                    image=img,
                    camera_lat=viewpoint.camera_lat,
                    camera_lon=viewpoint.camera_lon,
                    heading=viewpoint.heading,
                    pitch=float(pitch),  # Store for geometric height estimation
                    fov=float(fov),
                )
                console.print(f"    ✓ Got image")
            else:
                console.print(f"    ✗ No coverage")

        return images

    def fetch_multi_facade_images(
        self,
        footprint: dict,
        images_per_facade: int = 3,
        image_size: str = "640x480",
        fov: int = 90,
        pitch: int = 25,  # Look up at building (higher = more upward tilt)
        multi_pitch: bool = True,  # Fetch at multiple pitch angles
    ) -> Dict[str, List[StreetViewImage]]:
        """
        Fetch multiple Street View images per facade for higher confidence analysis.

        Samples multiple positions along each facade to get different viewpoints.

        Args:
            footprint: GeoJSON geometry (Polygon or Feature)
            images_per_facade: Number of images to fetch per facade direction
            image_size: Image dimensions
            fov: Field of view in degrees
            pitch: Camera pitch in degrees

        Returns:
            Dict mapping orientation to List[StreetViewImage]
        """
        # Parse footprint and analyze wall segments
        coords = self._parse_footprint(footprint)
        if not coords:
            return {}

        # Calculate centroid
        centroid_lon = sum(c[0] for c in coords) / len(coords)
        centroid_lat = sum(c[1] for c in coords) / len(coords)

        # Analyze wall segments
        segments_by_orientation = self._analyze_wall_segments(coords, centroid_lat)

        images = {'N': [], 'E': [], 'S': [], 'W': []}

        for orientation in ['N', 'E', 'S', 'W']:
            segments = segments_by_orientation.get(orientation, [])

            if not segments:
                # Use fallback viewpoints from centroid
                viewpoints = self._generate_fallback_viewpoints(
                    orientation, centroid_lat, centroid_lon, images_per_facade
                )
            else:
                # Sample viewpoints along the facade segments
                viewpoints = self._sample_facade_viewpoints(
                    orientation, segments, images_per_facade
                )

            # Define pitch angles to capture full building height
            if multi_pitch:
                pitch_angles = [5, 25, 45]  # Ground floor, middle, upper
                console.print(f"  Fetching {orientation} facade ({len(viewpoints)} pos × {len(pitch_angles)} pitches)...")
            else:
                pitch_angles = [pitch]
                console.print(f"  Fetching {orientation} facade ({len(viewpoints)} positions)...")

            for i, vp in enumerate(viewpoints):
                for p in pitch_angles:
                    img = self._fetch_streetview_image(
                        lat=vp['camera_lat'],
                        lon=vp['camera_lon'],
                        heading=vp['heading'],
                        size=image_size,
                        fov=fov,
                        pitch=p,
                    )

                    if img:
                        images[orientation].append(StreetViewImage(
                            orientation=orientation,
                            image=img,
                            camera_lat=vp['camera_lat'],
                            camera_lon=vp['camera_lon'],
                            heading=vp['heading'],
                            pitch=float(p),  # Store for geometric height estimation
                            fov=float(fov),
                        ))

            expected = len(viewpoints) * len(pitch_angles)
            console.print(f"    ✓ Got {len(images[orientation])}/{expected} images")

        return images

    def _sample_facade_viewpoints(
        self,
        orientation: str,
        segments: List[dict],
        num_samples: int,
    ) -> List[dict]:
        """Sample viewpoints along facade segments."""
        viewpoints = []

        # Sort segments by length (descending)
        sorted_segments = sorted(segments, key=lambda s: s['length_m'], reverse=True)
        total_length = sum(s['length_m'] for s in sorted_segments)

        if total_length < 5:  # Too short
            return []

        # Distribute samples proportionally to segment length
        samples_remaining = num_samples
        for seg in sorted_segments:
            if samples_remaining <= 0:
                break

            # Number of samples for this segment (at least 1)
            seg_samples = max(1, int(num_samples * seg['length_m'] / total_length))
            seg_samples = min(seg_samples, samples_remaining)

            # Sample positions along segment
            for i in range(seg_samples):
                # Position along segment (0.2 to 0.8 to avoid corners)
                t = 0.2 + 0.6 * (i + 0.5) / seg_samples

                # Interpolate position
                start_lon, start_lat = seg['start_wgs']
                end_lon, end_lat = seg['end_wgs']
                sample_lon = start_lon + t * (end_lon - start_lon)
                sample_lat = start_lat + t * (end_lat - start_lat)

                # Calculate camera position (offset from facade)
                wall_azimuth = seg['wall_azimuth']
                camera_lat, camera_lon = self._offset_position(
                    sample_lat, sample_lon,
                    wall_azimuth,
                    self.CAMERA_OFFSET_M
                )

                # Camera looks back at facade
                camera_heading = (wall_azimuth + 180) % 360

                viewpoints.append({
                    'camera_lat': camera_lat,
                    'camera_lon': camera_lon,
                    'heading': camera_heading,
                })

                samples_remaining -= 1
                if samples_remaining <= 0:
                    break

        return viewpoints

    def _generate_fallback_viewpoints(
        self,
        orientation: str,
        centroid_lat: float,
        centroid_lon: float,
        num_samples: int,
    ) -> List[dict]:
        """Generate fallback viewpoints when no segments found."""
        headings = {'N': 0, 'E': 90, 'S': 180, 'W': 270}
        heading = headings[orientation]
        offset_azimuth = (heading + 180) % 360

        viewpoints = []
        # Create viewpoints with slight lateral offsets
        for i in range(num_samples):
            # Offset perpendicular to viewing direction
            lateral_offset = (i - num_samples // 2) * 15  # 15m spacing
            lateral_azimuth = (heading + 90) % 360

            # Start from offset position
            lat1, lon1 = self._offset_position(
                centroid_lat, centroid_lon,
                lateral_azimuth, lateral_offset
            )

            # Then offset away from building
            camera_lat, camera_lon = self._offset_position(
                lat1, lon1,
                offset_azimuth,
                self.CAMERA_OFFSET_M
            )

            viewpoints.append({
                'camera_lat': camera_lat,
                'camera_lon': camera_lon,
                'heading': heading,
            })

        return viewpoints

    def _calculate_facade_viewpoints(self, footprint: dict) -> List[FacadeViewpoint]:
        """
        Calculate camera viewpoints for each facade direction.

        Args:
            footprint: GeoJSON geometry

        Returns:
            List of FacadeViewpoint objects
        """
        # Parse footprint coordinates
        coords = self._parse_footprint(footprint)
        if not coords:
            return []

        # Calculate building centroid
        centroid_lon = sum(c[0] for c in coords) / len(coords)
        centroid_lat = sum(c[1] for c in coords) / len(coords)

        # Calculate wall segments and group by orientation
        segments_by_orientation = self._analyze_wall_segments(coords, centroid_lat)

        viewpoints = []

        for orientation in ['N', 'E', 'S', 'W']:
            segments = segments_by_orientation.get(orientation, [])

            if not segments:
                # No facade in this direction - use centroid with offset
                viewpoint = self._create_fallback_viewpoint(
                    orientation, centroid_lat, centroid_lon
                )
            else:
                # Calculate facade centroid and camera position
                viewpoint = self._calculate_viewpoint_from_segments(
                    orientation, segments, centroid_lat, centroid_lon
                )

            viewpoints.append(viewpoint)

        return viewpoints

    def _parse_footprint(self, footprint: dict) -> List[Tuple[float, float]]:
        """Parse GeoJSON to coordinate list."""
        # Handle string input
        if isinstance(footprint, str):
            footprint = json.loads(footprint)

        # Handle Feature wrapper
        if footprint.get('type') == 'Feature':
            footprint = footprint.get('geometry', {})

        geom_type = footprint.get('type')
        coords = footprint.get('coordinates', [])

        if geom_type == 'Polygon':
            if coords and len(coords) > 0:
                return [(c[0], c[1]) for c in coords[0]]
        elif geom_type == 'MultiPolygon':
            if coords and len(coords) > 0 and len(coords[0]) > 0:
                return [(c[0], c[1]) for c in coords[0][0]]

        return []

    def _analyze_wall_segments(
        self,
        coords: List[Tuple[float, float]],
        ref_lat: float,
    ) -> Dict[str, List[dict]]:
        """
        Analyze wall segments and group by cardinal direction.

        Returns dict with keys N/E/S/W, values are lists of segment info.
        """
        # Convert to local metric coordinates for length calculation
        local_coords = []
        for lon, lat in coords:
            x = self.EARTH_RADIUS_M * math.radians(lon - coords[0][0]) * math.cos(math.radians(ref_lat))
            y = self.EARTH_RADIUS_M * math.radians(lat - coords[0][1])
            local_coords.append((x, y))

        segments_by_orientation = {'N': [], 'E': [], 'S': [], 'W': []}

        for i in range(len(coords) - 1):
            p1_wgs = coords[i]
            p2_wgs = coords[i + 1]
            p1_local = local_coords[i]
            p2_local = local_coords[i + 1]

            # Calculate segment properties
            dx = p2_local[0] - p1_local[0]
            dy = p2_local[1] - p1_local[1]
            length = math.sqrt(dx * dx + dy * dy)

            if length < 0.5:  # Skip tiny segments
                continue

            # Calculate segment direction and wall facing azimuth
            segment_azimuth = math.degrees(math.atan2(dx, dy)) % 360
            wall_azimuth = (segment_azimuth + 90) % 360  # Normal to segment

            # Determine cardinal direction
            orientation = self._azimuth_to_cardinal(wall_azimuth)

            # Store segment info
            segment_info = {
                'start_wgs': p1_wgs,
                'end_wgs': p2_wgs,
                'length_m': length,
                'wall_azimuth': wall_azimuth,
                'midpoint_lon': (p1_wgs[0] + p2_wgs[0]) / 2,
                'midpoint_lat': (p1_wgs[1] + p2_wgs[1]) / 2,
            }

            segments_by_orientation[orientation].append(segment_info)

        return segments_by_orientation

    def _azimuth_to_cardinal(self, azimuth: float) -> str:
        """Convert azimuth to cardinal direction."""
        azimuth = azimuth % 360
        if azimuth >= 315 or azimuth < 45:
            return 'N'
        elif 45 <= azimuth < 135:
            return 'E'
        elif 135 <= azimuth < 225:
            return 'S'
        else:
            return 'W'

    def _calculate_viewpoint_from_segments(
        self,
        orientation: str,
        segments: List[dict],
        centroid_lat: float,
        centroid_lon: float,
    ) -> FacadeViewpoint:
        """Calculate camera viewpoint from facade segments."""
        # Weight segments by length to find facade centroid
        total_length = sum(s['length_m'] for s in segments)

        weighted_lat = sum(s['midpoint_lat'] * s['length_m'] for s in segments) / total_length
        weighted_lon = sum(s['midpoint_lon'] * s['length_m'] for s in segments) / total_length
        avg_azimuth = sum(s['wall_azimuth'] * s['length_m'] for s in segments) / total_length

        # Camera heading: opposite of wall azimuth (looking AT the wall)
        camera_heading = (avg_azimuth + 180) % 360

        # Camera position: offset from facade in direction wall is facing
        offset_lat, offset_lon = self._offset_position(
            weighted_lat, weighted_lon,
            avg_azimuth,  # Move away from building in direction wall faces
            self.CAMERA_OFFSET_M
        )

        return FacadeViewpoint(
            orientation=orientation,
            camera_lat=offset_lat,
            camera_lon=offset_lon,
            heading=camera_heading,
            facade_centroid=(weighted_lat, weighted_lon),
            facade_length_m=total_length,
        )

    def _create_fallback_viewpoint(
        self,
        orientation: str,
        centroid_lat: float,
        centroid_lon: float,
    ) -> FacadeViewpoint:
        """Create fallback viewpoint when no facade segments found."""
        # Direction to look based on orientation
        headings = {'N': 0, 'E': 90, 'S': 180, 'W': 270}
        heading = headings[orientation]

        # Offset camera position opposite to viewing direction
        offset_azimuth = (heading + 180) % 360
        camera_lat, camera_lon = self._offset_position(
            centroid_lat, centroid_lon,
            offset_azimuth,
            self.CAMERA_OFFSET_M
        )

        return FacadeViewpoint(
            orientation=orientation,
            camera_lat=camera_lat,
            camera_lon=camera_lon,
            heading=heading,
            facade_centroid=(centroid_lat, centroid_lon),
            facade_length_m=0,
        )

    def _offset_position(
        self,
        lat: float,
        lon: float,
        azimuth_deg: float,
        distance_m: float,
    ) -> Tuple[float, float]:
        """
        Calculate new position offset from original by distance and azimuth.

        Args:
            lat, lon: Starting position
            azimuth_deg: Direction to offset (0=N, 90=E, etc.)
            distance_m: Distance to offset

        Returns:
            (new_lat, new_lon)
        """
        # Convert to radians
        lat_rad = math.radians(lat)
        azimuth_rad = math.radians(azimuth_deg)

        # Angular distance
        angular_dist = distance_m / self.EARTH_RADIUS_M

        # New position using spherical geometry
        new_lat_rad = math.asin(
            math.sin(lat_rad) * math.cos(angular_dist) +
            math.cos(lat_rad) * math.sin(angular_dist) * math.cos(azimuth_rad)
        )

        new_lon_rad = math.radians(lon) + math.atan2(
            math.sin(azimuth_rad) * math.sin(angular_dist) * math.cos(lat_rad),
            math.cos(angular_dist) - math.sin(lat_rad) * math.sin(new_lat_rad)
        )

        return math.degrees(new_lat_rad), math.degrees(new_lon_rad)

    def _fetch_streetview_image(
        self,
        lat: float,
        lon: float,
        heading: float,
        size: str = "640x480",
        fov: int = 90,
        pitch: int = 25,  # Look up at building (higher = more upward tilt)
    ) -> Optional[Image.Image]:
        """Fetch a single Street View image."""
        url = "https://maps.googleapis.com/maps/api/streetview"
        params = {
            "size": size,
            "location": f"{lat},{lon}",
            "heading": heading,
            "pitch": pitch,
            "fov": fov,
            "radius": 100,  # Search radius for nearest panorama
            "source": "outdoor",
            "key": self.api_key,
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                # Check if we got an actual image (not error image)
                content_type = response.headers.get('content-type', '')
                if 'image' in content_type:
                    return Image.open(BytesIO(response.content))
            return None
        except Exception as e:
            console.print(f"[red]Street View fetch error: {e}[/red]")
            return None

    def check_coverage(self, lat: float, lon: float) -> bool:
        """Check if Street View coverage exists at location."""
        url = "https://maps.googleapis.com/maps/api/streetview/metadata"
        params = {
            "location": f"{lat},{lon}",
            "radius": 100,
            "key": self.api_key,
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get('status') == 'OK'
            return False
        except:
            return False


def fetch_building_facades(
    footprint: dict,
    api_key: str = None,
) -> Dict[str, StreetViewImage]:
    """
    Convenience function to fetch facade images.

    Args:
        footprint: GeoJSON geometry of building
        api_key: Google API key (optional, uses env var)

    Returns:
        Dict of orientation -> StreetViewImage
    """
    fetcher = StreetViewFacadeFetcher(api_key)
    return fetcher.fetch_facade_images(footprint)


# ═══════════════════════════════════════════════════════════════════════════════
# GEOMETRIC HEIGHT ESTIMATION FROM STREET VIEW
# Uses camera position, pitch, and roof edge detection for accurate height
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GeometricHeightEstimate:
    """Building height estimate from geometric analysis."""
    height_m: float  # Estimated building height
    confidence: float  # 0-1 confidence score
    method: str  # "geometric", "floor_count", "combined"
    camera_distance_m: float  # Distance from camera to building
    roof_angle_deg: float  # Vertical angle to roof from camera
    floor_count: int  # Estimated floor count (height / 3.0)
    has_attic: bool
    has_basement: bool
    notes: List[str]  # Estimation notes


class GeometricHeightEstimator:
    """
    Estimate building height using street view geometry.

    Method:
    1. Calculate horizontal distance from camera to building facade
    2. Use LLM to detect roof edge position in image (% from bottom)
    3. Convert image position to vertical angle using camera FOV and pitch
    4. Calculate height: H = distance * tan(angle) + camera_height

    Reference heights for validation:
    - Standard residential floor: 2.7-3.0m
    - Commercial ground floor: 3.5-4.5m
    - Swedish MFH: 3-8 floors typical (9-24m)
    """

    # Standard values
    CAMERA_HEIGHT_M = 2.5  # Google Street View camera height (on car roof)
    EARTH_RADIUS_M = 6371000

    def __init__(self):
        self.console = Console()

    def estimate_height(
        self,
        camera_lat: float,
        camera_lon: float,
        facade_lat: float,
        facade_lon: float,
        camera_pitch_deg: float,
        camera_fov_deg: float,
        roof_position_pct: float,  # 0-1, where in image is roof (0=bottom, 1=top)
        image_height_px: int = 480,
    ) -> GeometricHeightEstimate:
        """
        Estimate building height from camera geometry and roof position.

        Args:
            camera_lat, camera_lon: Camera position
            facade_lat, facade_lon: Building facade position
            camera_pitch_deg: Camera tilt (positive = looking up)
            camera_fov_deg: Vertical field of view
            roof_position_pct: Where roof edge appears in image (0=bottom, 1=top)
            image_height_px: Image height in pixels

        Returns:
            GeometricHeightEstimate with height and confidence
        """
        notes = []

        # 1. Calculate horizontal distance to facade
        distance_m = self._haversine_distance(
            camera_lat, camera_lon, facade_lat, facade_lon
        )
        notes.append(f"Camera distance: {distance_m:.1f}m")

        # 2. Calculate vertical angle to roof
        # Image center corresponds to pitch angle
        # Each pixel above/below center adds/subtracts from angle
        center_pct = 0.5
        angle_per_pct = camera_fov_deg  # Full FOV across image height

        # Roof position relative to center (positive = above center)
        roof_offset_pct = roof_position_pct - center_pct
        roof_angle_from_center = roof_offset_pct * angle_per_pct

        # Total vertical angle to roof (pitch + offset)
        roof_angle_deg = camera_pitch_deg + roof_angle_from_center
        notes.append(f"Roof angle: {roof_angle_deg:.1f}° (pitch={camera_pitch_deg}°, offset={roof_angle_from_center:.1f}°)")

        # 3. Calculate height using trigonometry
        # height = distance * tan(angle) + camera_height
        roof_angle_rad = math.radians(roof_angle_deg)
        height_above_camera = distance_m * math.tan(roof_angle_rad)
        total_height_m = height_above_camera + self.CAMERA_HEIGHT_M

        # Sanity check
        if total_height_m < 3:
            notes.append("Warning: Height < 3m, may be obstructed view")
            total_height_m = max(3.0, total_height_m)
        elif total_height_m > 100:
            notes.append("Warning: Height > 100m, likely error")
            total_height_m = min(100.0, total_height_m)

        notes.append(f"Raw height: {height_above_camera:.1f}m above camera")

        # 4. Estimate floor count
        floor_height = 2.9  # Swedish residential average
        floor_count = int(round(total_height_m / floor_height))
        floor_count = max(1, min(floor_count, 30))

        # 5. Confidence based on geometry quality
        confidence = 0.8
        if distance_m < 15:
            confidence -= 0.2  # Too close, perspective distortion
            notes.append("Low confidence: camera too close")
        elif distance_m > 100:
            confidence -= 0.1  # Far away, less precise
            notes.append("Reduced confidence: camera far from building")

        if roof_angle_deg < 5:
            confidence -= 0.2  # Roof barely visible
            notes.append("Low confidence: roof near horizon")
        elif roof_angle_deg > 70:
            confidence -= 0.2  # Looking almost straight up
            notes.append("Low confidence: extreme upward angle")

        confidence = max(0.3, min(1.0, confidence))

        return GeometricHeightEstimate(
            height_m=round(total_height_m, 1),
            confidence=confidence,
            method="geometric",
            camera_distance_m=round(distance_m, 1),
            roof_angle_deg=round(roof_angle_deg, 1),
            floor_count=floor_count,
            has_attic=False,  # Would need LLM to detect
            has_basement=False,
            notes=notes,
        )

    def estimate_from_floor_count(
        self,
        floor_count: int,
        building_form: str = "lamellhus",
        has_commercial_ground: bool = False,
        has_attic: bool = False,
    ) -> GeometricHeightEstimate:
        """
        Estimate height from LLM-detected floor count (backup method).

        Args:
            floor_count: Number of floors detected
            building_form: Swedish building form for floor height adjustment
            has_commercial_ground: Commercial ground floor (higher ceiling)
            has_attic: Has visible attic/mansard

        Returns:
            GeometricHeightEstimate
        """
        # Base floor height by building type
        floor_heights = {
            "lamellhus": 2.8,  # 1945-1975 slab blocks
            "skivhus": 2.7,   # Miljonprogrammet high-rises
            "punkthus": 2.8,  # Tower blocks
            "pre_1930": 3.2,  # Old buildings, higher ceilings
            "modern": 2.6,    # 2000+ buildings
            "radhus": 2.6,    # Row houses
        }

        floor_height = floor_heights.get(building_form, 2.8)

        # Calculate height
        height_m = floor_count * floor_height

        # Adjust for commercial ground floor (typically 3.5-4.5m)
        if has_commercial_ground:
            height_m += 1.0  # Extra height for ground floor

        # Add for attic
        if has_attic:
            height_m += 2.5

        notes = [
            f"Floor count method: {floor_count} floors × {floor_height}m",
            f"Building form: {building_form}",
        ]
        if has_commercial_ground:
            notes.append("Commercial ground floor: +1.0m")
        if has_attic:
            notes.append("Attic: +2.5m")

        return GeometricHeightEstimate(
            height_m=round(height_m, 1),
            confidence=0.7,  # Floor count less precise than geometric
            method="floor_count",
            camera_distance_m=0,
            roof_angle_deg=0,
            floor_count=floor_count,
            has_attic=has_attic,
            has_basement=False,
            notes=notes,
        )

    def combine_estimates(
        self,
        geometric: Optional[GeometricHeightEstimate],
        floor_based: Optional[GeometricHeightEstimate],
    ) -> GeometricHeightEstimate:
        """
        Combine geometric and floor-count estimates.

        IMPORTANT: Multi-position geometric is a DIRECT PHYSICAL MEASUREMENT.
        When multiple camera positions agree, this is MORE RELIABLE than
        floor count × assumed floor height (which requires era assumptions).

        HOWEVER: We add sanity checks to reject wildly unrealistic geometric estimates
        (e.g., 100m for a 5-floor building is obviously wrong).

        Priority for HEIGHT:
        1. Multi-position geometric (conf ≥ 0.85): USE AS PRIMARY (physical measurement)
        2. Multi-position geometric (conf 0.70-0.85): Weight 3× vs floor-based
        3. Single-position geometric: Weight by confidence
        4. Floor-based: Fallback (relies on floor height assumptions)

        Floor count from declarations is used for VALIDATION, not as primary height source.
        """
        # ═══════════════════════════════════════════════════════════════════════
        # SANITY CHECK: Reject unrealistic geometric estimates
        # Swedish buildings: max ~80m (typical high-rise), 3-4m per floor
        # ═══════════════════════════════════════════════════════════════════════
        MAX_REALISTIC_HEIGHT_M = 80.0  # Tallest Swedish residential ~70-80m
        MAX_HEIGHT_PER_FLOOR_M = 5.0   # Commercial ground floor max
        MIN_HEIGHT_PER_FLOOR_M = 2.2   # Minimum realistic

        if geometric and floor_based:
            # Check if geometric is physically impossible
            implied_floor_height = geometric.height_m / max(floor_based.floor_count, 1)

            if geometric.height_m > MAX_REALISTIC_HEIGHT_M:
                # Reject: Taller than any Swedish residential building
                geometric = GeometricHeightEstimate(
                    height_m=geometric.height_m,
                    confidence=0.15,  # Very low confidence
                    method=f"{geometric.method}_rejected",
                    camera_distance_m=geometric.camera_distance_m,
                    roof_angle_deg=geometric.roof_angle_deg,
                    floor_count=geometric.floor_count,
                    has_attic=geometric.has_attic,
                    has_basement=geometric.has_basement,
                    notes=[f"REJECTED: {geometric.height_m:.0f}m exceeds max realistic height ({MAX_REALISTIC_HEIGHT_M}m)"],
                )

            elif implied_floor_height > MAX_HEIGHT_PER_FLOOR_M:
                # Reject: Implies impossibly tall floors
                geometric = GeometricHeightEstimate(
                    height_m=geometric.height_m,
                    confidence=0.15,  # Very low confidence
                    method=f"{geometric.method}_rejected",
                    camera_distance_m=geometric.camera_distance_m,
                    roof_angle_deg=geometric.roof_angle_deg,
                    floor_count=geometric.floor_count,
                    has_attic=geometric.has_attic,
                    has_basement=geometric.has_basement,
                    notes=[f"REJECTED: {geometric.height_m:.0f}m ÷ {floor_based.floor_count} floors = {implied_floor_height:.1f}m/floor (impossible)"],
                )

            elif implied_floor_height < MIN_HEIGHT_PER_FLOOR_M:
                # Reject: Implies impossibly short floors
                geometric = GeometricHeightEstimate(
                    height_m=geometric.height_m,
                    confidence=0.15,
                    method=f"{geometric.method}_rejected",
                    camera_distance_m=geometric.camera_distance_m,
                    roof_angle_deg=geometric.roof_angle_deg,
                    floor_count=geometric.floor_count,
                    has_attic=geometric.has_attic,
                    has_basement=geometric.has_basement,
                    notes=[f"REJECTED: {geometric.height_m:.0f}m ÷ {floor_based.floor_count} floors = {implied_floor_height:.1f}m/floor (too short)"],
                )

        # If geometric was rejected, just use floor-based
        if geometric and "_rejected" in geometric.method:
            if floor_based:
                floor_based.notes = floor_based.notes + geometric.notes
                return floor_based
            # No floor-based either - use capped geometric
            return GeometricHeightEstimate(
                height_m=min(geometric.height_m, MAX_REALISTIC_HEIGHT_M),
                confidence=0.30,
                method="capped_fallback",
                camera_distance_m=geometric.camera_distance_m,
                roof_angle_deg=geometric.roof_angle_deg,
                floor_count=geometric.floor_count,
                has_attic=geometric.has_attic,
                has_basement=geometric.has_basement,
                notes=geometric.notes + ["Using capped value as fallback"],
            )

        if geometric and not floor_based:
            # Even without floor-based, cap at max realistic
            if geometric.height_m > MAX_REALISTIC_HEIGHT_M:
                geometric = GeometricHeightEstimate(
                    height_m=MAX_REALISTIC_HEIGHT_M,
                    confidence=0.50,
                    method=f"{geometric.method}_capped",
                    camera_distance_m=geometric.camera_distance_m,
                    roof_angle_deg=geometric.roof_angle_deg,
                    floor_count=geometric.floor_count,
                    has_attic=geometric.has_attic,
                    has_basement=geometric.has_basement,
                    notes=[f"Capped from {geometric.height_m:.0f}m to max realistic ({MAX_REALISTIC_HEIGHT_M}m)"],
                )
            return geometric
        if floor_based and not geometric:
            return floor_based
        if not geometric and not floor_based:
            return GeometricHeightEstimate(
                height_m=12.0,  # Swedish MFH default (4 floors)
                confidence=0.3,
                method="default",
                camera_distance_m=0,
                roof_angle_deg=0,
                floor_count=4,
                has_attic=False,
                has_basement=False,
                notes=["No data available, using default"],
            )

        # Check reliability of geometric estimate
        is_multi_geometric = geometric.method == "multi_geometric"
        geometric_very_reliable = is_multi_geometric and geometric.confidence >= 0.85
        geometric_reliable = is_multi_geometric and geometric.confidence >= 0.70

        height_diff = abs(geometric.height_m - floor_based.height_m)
        notes = [
            f"Geometric: {geometric.height_m}m ({geometric.method}, conf={geometric.confidence:.2f})",
            f"Floor-based: {floor_based.height_m}m (conf={floor_based.confidence:.2f})",
        ]

        # ═══════════════════════════════════════════════════════════════════════
        # CASE 1: Very high confidence multi-position (≥0.85)
        # This is a DIRECT PHYSICAL MEASUREMENT - trust it as primary for HEIGHT
        # Floor count is only used for validation/confidence boost
        # ═══════════════════════════════════════════════════════════════════════
        if geometric_very_reliable:
            weighted_height = geometric.height_m  # Use geometric directly
            confidence = geometric.confidence

            # Floor count validates but doesn't override height
            geo_floors = int(round(geometric.height_m / 2.9))
            if abs(geo_floors - floor_based.floor_count) <= 1:
                confidence = min(0.98, confidence + 0.05)
                notes.append("✓ Floor count validates geometric (direct measurement)")
            elif height_diff > 3:
                # Disagreement - note it but still trust geometric for HEIGHT
                notes.append(f"⚠ Floor-based differs by {height_diff:.1f}m - geometric is physical measurement, trusting it")
                # Slight confidence reduction due to disagreement
                confidence = max(0.85, confidence - 0.05)
            else:
                notes.append("Multi-position geometric used as primary (physical measurement)")

        # ═══════════════════════════════════════════════════════════════════════
        # CASE 2: Reliable multi-position (0.70-0.85)
        # Weight geometric 3× vs floor-based
        # ═══════════════════════════════════════════════════════════════════════
        elif geometric_reliable:
            geo_weight = geometric.confidence * 3.0  # 3× weight for multi-position
            floor_weight = floor_based.confidence
            total_weight = geo_weight + floor_weight

            weighted_height = (
                geometric.height_m * geo_weight +
                floor_based.height_m * floor_weight
            ) / total_weight

            confidence = min(0.92, geometric.confidence + 0.05)
            notes.append("Multi-position geometric weighted 3× (multiple camera agreement)")

            geo_floors = int(round(geometric.height_m / 2.9))
            if abs(geo_floors - floor_based.floor_count) <= 1:
                confidence = min(0.95, confidence + 0.05)
                notes.append("✓ Floor count validates geometric")

        # ═══════════════════════════════════════════════════════════════════════
        # CASE 3: Single-position or low-confidence geometric
        # Standard weighted average
        # ═══════════════════════════════════════════════════════════════════════
        else:
            geo_weight = geometric.confidence
            floor_weight = floor_based.confidence
            total_weight = geo_weight + floor_weight

            weighted_height = (
                geometric.height_m * geo_weight +
                floor_based.height_m * floor_weight
            ) / total_weight

            confidence = (geometric.confidence + floor_based.confidence) / 2
            if height_diff > 5:
                notes.append(f"Warning: estimates differ by {height_diff:.1f}m")
                confidence *= 0.8
            elif height_diff < 2:
                confidence = min(0.90, confidence + 0.10)
                notes.append("Good agreement between methods")

        # Floor count: use geometric-derived if multi-position reliable, else floor-based
        final_floor_count = (
            geometric.floor_count if (geometric_very_reliable or geometric_reliable)
            else floor_based.floor_count
        )

        return GeometricHeightEstimate(
            height_m=round(weighted_height, 1),
            confidence=round(confidence, 2),
            method="combined",
            camera_distance_m=geometric.camera_distance_m,
            roof_angle_deg=geometric.roof_angle_deg,
            floor_count=final_floor_count,
            has_attic=floor_based.has_attic,
            has_basement=floor_based.has_basement,
            notes=notes,
        )

    def _haversine_distance(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float,
    ) -> float:
        """Calculate distance between two points in meters."""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return self.EARTH_RADIUS_M * c

    def estimate_from_multiple_positions(
        self,
        images: List[Any],  # List of StreetViewImage with camera_lat, camera_lon, pitch, fov
        facade_lat: float,
        facade_lon: float,
        roof_position_pct: float,
        reference_floor_count: int = 0,  # If known, use for cross-validation
    ) -> GeometricHeightEstimate:
        """
        Estimate height using multiple camera positions for triangulation.

        IMPROVED METHOD (v2):
        The key insight is that roof_position_pct from LLM is valid for ONE viewing
        condition, not all. We use it to establish a reference height, then validate
        against other camera positions and floor count.

        Method:
        1. Find "reference image" closest to optimal distance (30-40m)
        2. Calculate height from reference using LLM's roof_position_pct
        3. For other images, calculate EXPECTED roof position given reference height
        4. Compare expected vs what roof_position would need to be - reject inconsistent
        5. Use floor count for cross-validation if available

        Args:
            images: List of StreetViewImage with camera metadata
            facade_lat, facade_lon: Building centroid
            roof_position_pct: Where roof appears in image (from LLM analysis)
            reference_floor_count: Floor count from LLM for cross-validation

        Returns:
            GeometricHeightEstimate with high confidence if multiple positions agree
        """
        notes = []

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 1: Find reference image (optimal distance 25-45m)
        # ═══════════════════════════════════════════════════════════════════════
        valid_images = []
        for img in images:
            if not hasattr(img, 'camera_lat') or not hasattr(img, 'pitch'):
                continue

            pitch = getattr(img, 'pitch', 0)
            if pitch < -10 or pitch > 60:  # Allow slight downward pitch
                continue

            dist = self._haversine_distance(
                img.camera_lat, img.camera_lon, facade_lat, facade_lon
            )
            if 10 < dist < 150:  # Reasonable range
                valid_images.append({
                    'img': img,
                    'dist': dist,
                    'pitch': pitch,
                    'fov': getattr(img, 'fov', 90),
                })

        if not valid_images:
            return GeometricHeightEstimate(
                height_m=12.0,
                confidence=0.3,
                method="multi_geometric_failed",
                camera_distance_m=0,
                roof_angle_deg=0,
                floor_count=4,
                has_attic=False,
                has_basement=False,
                notes=["No valid images with camera metadata"],
            )

        # Sort by distance from optimal (35m)
        valid_images.sort(key=lambda x: abs(x['dist'] - 35))
        reference = valid_images[0]
        notes.append(f"Reference image: {reference['dist']:.0f}m, pitch={reference['pitch']}°")

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 2: Calculate height from reference image using LLM's roof position
        # ═══════════════════════════════════════════════════════════════════════
        ref_estimate = self.estimate_height(
            camera_lat=reference['img'].camera_lat,
            camera_lon=reference['img'].camera_lon,
            facade_lat=facade_lat,
            facade_lon=facade_lon,
            camera_pitch_deg=reference['pitch'],
            camera_fov_deg=reference['fov'],
            roof_position_pct=roof_position_pct,
        )
        reference_height = ref_estimate.height_m
        notes.append(f"Reference height: {reference_height:.1f}m (from LLM roof position)")

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 3: Cross-validate with floor count (if available)
        # ═══════════════════════════════════════════════════════════════════════
        floor_based_height = None
        if reference_floor_count > 0:
            floor_based_height = reference_floor_count * 2.9  # Swedish average
            floor_height_ratio = reference_height / floor_based_height if floor_based_height > 0 else 1
            notes.append(f"Floor-based: {floor_based_height:.1f}m ({reference_floor_count} floors)")

            # If geometric is wildly off from floor-based, trust floor-based more
            if floor_height_ratio > 2.0:  # Geometric >2x floor-based
                notes.append(f"⚠ Geometric {floor_height_ratio:.1f}x floor-based - likely bad roof detection")
                reference_height = floor_based_height * 1.2  # Use floor-based + 20%
            elif floor_height_ratio < 0.5:  # Geometric <0.5x floor-based
                notes.append(f"⚠ Geometric only {floor_height_ratio:.1f}x floor-based - adjusting")
                reference_height = floor_based_height * 0.9  # Use floor-based - 10%

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 4: Validate with other camera positions
        # For each image, calculate what roof_position WOULD be if our height is correct
        # ═══════════════════════════════════════════════════════════════════════
        consistent_count = 0
        total_validated = 0

        for vi in valid_images[1:5]:  # Check up to 4 other images
            # Expected roof angle given our reference height
            height_above_camera = reference_height - self.CAMERA_HEIGHT_M
            expected_roof_angle_rad = math.atan(height_above_camera / vi['dist'])
            expected_roof_angle_deg = math.degrees(expected_roof_angle_rad)

            # Expected roof position in image given this angle
            angle_offset_from_pitch = expected_roof_angle_deg - vi['pitch']
            expected_roof_pct = 0.5 + (angle_offset_from_pitch / vi['fov'])
            expected_roof_pct = max(0.1, min(0.95, expected_roof_pct))

            # A reasonable roof should appear in top half of image
            total_validated += 1
            if 0.5 < expected_roof_pct < 0.95:
                consistent_count += 1
            else:
                notes.append(f"Image at {vi['dist']:.0f}m: roof would be at {expected_roof_pct:.0%} (suspicious)")

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 5: Calculate confidence
        # ═══════════════════════════════════════════════════════════════════════
        confidence = 0.70  # Base confidence

        # Bonus for good reference distance
        if 25 < reference['dist'] < 50:
            confidence += 0.10

        # Bonus for floor count agreement
        if floor_based_height:
            height_diff_pct = abs(reference_height - floor_based_height) / floor_based_height
            if height_diff_pct < 0.15:  # Within 15%
                confidence += 0.10
                notes.append("✓ Geometric agrees with floor count")
            elif height_diff_pct > 0.50:  # More than 50% off
                confidence -= 0.15
                notes.append("⚠ Large disagreement with floor count")

        # Bonus for multi-image consistency
        if total_validated > 0:
            consistency_rate = consistent_count / total_validated
            confidence += 0.10 * consistency_rate
            if consistency_rate > 0.5:
                notes.append(f"✓ {consistent_count}/{total_validated} images consistent")

        # Penalty for extreme values
        if reference_height > 50:
            confidence -= 0.15
            notes.append("⚠ Height >50m unusual for residential")
        elif reference_height < 6:
            confidence -= 0.10
            notes.append("⚠ Height <6m unusual")

        confidence = max(0.30, min(0.95, confidence))

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 6: Final estimate (blend if floor count available)
        # ═══════════════════════════════════════════════════════════════════════
        final_height = reference_height
        if floor_based_height and abs(reference_height - floor_based_height) > 5:
            # Significant disagreement - blend towards floor-based
            blend_factor = 0.7 if confidence < 0.6 else 0.5
            final_height = reference_height * (1 - blend_factor) + floor_based_height * blend_factor
            notes.append(f"Blended estimate: {final_height:.1f}m")

        floor_count = int(round(final_height / 2.9))
        floor_count = max(1, min(floor_count, 30))

        return GeometricHeightEstimate(
            height_m=round(final_height, 1),
            confidence=round(confidence, 2),
            method="multi_geometric",
            camera_distance_m=round(reference['dist'], 1),
            roof_angle_deg=round(ref_estimate.roof_angle_deg, 1),
            floor_count=floor_count,
            has_attic=False,
            has_basement=False,
            notes=notes,
        )
