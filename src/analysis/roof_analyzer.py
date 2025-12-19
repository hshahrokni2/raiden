"""
Roof Analyzer - Analyze roof from satellite/aerial imagery.

Detects:
- Roof segments and their orientations
- Obstructions (HVAC units, vents, skylights, chimneys)
- Existing solar panels
- Usable area for new PV installation
- Roof type (flat, pitched, gabled)

Data sources (in priority order):
1. Google Solar API - Best: provides complete roof analysis
2. Swedish Lantmäteriet - Swedish orthophotos
3. Bing Maps Aerial - Good resolution, free tier
4. OpenStreetMap tags - building:roof:* metadata
5. Manual estimation from footprint + era
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import requests
from rich.console import Console

from ..core.config import settings

console = Console()


class RoofType(Enum):
    """Roof type classification."""
    FLAT = "flat"
    PITCHED = "pitched"  # Single slope
    GABLED = "gabled"   # Two slopes meeting at ridge
    HIPPED = "hipped"   # Four slopes
    MANSARD = "mansard"  # Double slope (Swedish: brutet tak)
    UNKNOWN = "unknown"


class ObstructionType(Enum):
    """Roof obstruction types."""
    HVAC_UNIT = "hvac_unit"
    VENTILATION_DUCT = "ventilation_duct"
    SKYLIGHT = "skylight"
    CHIMNEY = "chimney"
    ANTENNA = "antenna"
    EXISTING_SOLAR = "existing_solar"
    ACCESS_HATCH = "access_hatch"
    ELEVATOR_HOUSING = "elevator_housing"
    OTHER = "other"


@dataclass
class RoofObstruction:
    """Individual roof obstruction."""
    type: ObstructionType
    area_m2: float
    height_m: Optional[float] = None
    position: Optional[Tuple[float, float]] = None  # Relative position on roof
    casts_shadow: bool = True
    notes: str = ""


@dataclass
class RoofSegment:
    """A segment of the roof (e.g., one side of a gabled roof)."""
    segment_id: str
    area_m2: float
    azimuth_deg: float  # Orientation (0=N, 90=E, 180=S, 270=W)
    pitch_deg: float    # Slope angle (0 for flat)
    usable_area_m2: float  # After obstructions and setbacks

    # PV suitability
    pv_suitable: bool = True
    pv_efficiency_factor: float = 1.0  # 0-1, accounts for orientation
    annual_irradiance_kwh_m2: float = 0  # From solar analysis

    # Shading
    shading_factor: float = 1.0  # 0-1, where 1 = no shading
    shade_hours_per_year: float = 0


@dataclass
class ExistingSolarInstallation:
    """Existing solar installation on roof."""
    area_m2: float
    capacity_kwp: float
    annual_production_kwh: Optional[float] = None
    panel_type: str = "unknown"  # monocrystalline, polycrystalline, thin-film
    installation_year: Optional[int] = None
    detected_from: str = "unknown"  # satellite, osm, declaration


@dataclass
class RoofAnalysis:
    """Complete roof analysis results."""
    # Basic roof characteristics
    total_area_m2: float
    roof_type: RoofType
    primary_azimuth_deg: float  # Main roof orientation
    primary_pitch_deg: float    # Main roof slope

    # Segments (for complex roofs)
    segments: List[RoofSegment] = field(default_factory=list)

    # Obstructions detected
    obstructions: List[RoofObstruction] = field(default_factory=list)
    total_obstruction_area_m2: float = 0

    # Existing solar
    existing_solar: Optional[ExistingSolarInstallation] = None

    # Available space calculation
    gross_available_m2: float = 0      # Total roof minus obstructions
    setback_area_m2: float = 0         # Required setbacks from edges
    access_path_area_m2: float = 0     # Maintenance access paths
    net_available_m2: float = 0        # Final usable area for new PV

    # PV potential
    optimal_capacity_kwp: float = 0
    annual_yield_kwh_per_kwp: float = 0
    annual_generation_potential_kwh: float = 0

    # Shading
    average_shading_loss_pct: float = 0

    # Data source tracking
    data_source: str = "unknown"
    confidence: float = 0.0
    analysis_date: Optional[str] = None


class RoofAnalyzer:
    """
    Analyze roof from satellite/aerial imagery and map data.

    Usage:
        analyzer = RoofAnalyzer(google_api_key="...")
        analysis = analyzer.analyze(
            lat=59.302,
            lon=18.104,
            footprint_area_m2=320,
            building_height_m=21
        )
    """

    # Swedish solar parameters
    LATITUDE_IRRADIANCE = {
        # Latitude: (horizontal kWh/m²/year, optimal_tilt_deg)
        55.0: (1050, 38),  # Malmö
        57.7: (1000, 40),  # Gothenburg
        59.3: (950, 42),   # Stockholm
        63.8: (900, 45),   # Sundsvall
        67.8: (850, 50),   # Kiruna
    }

    # Roof utilization factors
    FLAT_ROOF_SETBACK_M = 1.0      # Required setback from edges
    PITCHED_ROOF_SETBACK_M = 0.5
    ACCESS_PATH_WIDTH_M = 0.8      # Between panel rows
    PANEL_SPACING_FACTOR = 0.15    # 15% loss for spacing on flat roofs

    # Panel specs
    PANEL_EFFICIENCY = 0.21        # Modern panels ~21%
    PANEL_POWER_DENSITY_WP_M2 = 210  # ~210 Wp per m²
    SYSTEM_LOSSES = 0.14           # Inverter, wiring, soiling

    def __init__(
        self,
        google_api_key: Optional[str] = None,
        lantmateriet_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize analyzer with API keys.

        Args:
            google_api_key: Google Solar API key (recommended)
            lantmateriet_key: Swedish Lantmäteriet API key
            cache_dir: Directory for caching API responses
        """
        self.google_api_key = google_api_key or settings.google_api_key
        self.lantmateriet_key = lantmateriet_key
        self.cache_dir = cache_dir or settings.cache_dir / "roof_analysis"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def analyze(
        self,
        lat: float,
        lon: float,
        footprint_area_m2: float,
        building_height_m: float = None,
        construction_year: int = None,
        known_roof_type: RoofType = None,
        existing_solar_kwp: float = 0,
        osm_roof_tags: Dict[str, str] = None,
    ) -> RoofAnalysis:
        """
        Analyze roof and calculate PV potential.

        Priority:
        1. Google Solar API (if key available)
        2. Swedish Lantmäteriet + OSM tags
        3. Estimation from footprint + era

        Args:
            lat, lon: Building coordinates
            footprint_area_m2: Building footprint area
            building_height_m: Building height (for shading)
            construction_year: For era-based estimation
            known_roof_type: Override detected roof type
            existing_solar_kwp: Known existing solar capacity
            osm_roof_tags: OSM roof:* tags if available

        Returns:
            RoofAnalysis with complete assessment
        """
        console.print(f"\n[cyan]Analyzing roof at {lat:.6f}, {lon:.6f}[/cyan]")

        # Try Google Solar API first
        if self.google_api_key:
            try:
                analysis = self._analyze_google_solar(lat, lon, footprint_area_m2)
                if analysis and analysis.confidence > 0.5:
                    console.print(f"  [green]Google Solar API: confidence {analysis.confidence:.0%}[/green]")
                    return analysis
            except Exception as e:
                console.print(f"  [yellow]Google Solar API failed: {e}[/yellow]")

        # Try OSM roof tags
        if osm_roof_tags:
            analysis = self._analyze_from_osm(
                osm_roof_tags, footprint_area_m2, lat
            )
            if analysis and analysis.confidence > 0.4:
                console.print(f"  [green]OSM tags: confidence {analysis.confidence:.0%}[/green]")
                return analysis

        # Fall back to estimation
        console.print("  [dim]Using estimation from footprint + era[/dim]")
        return self._estimate_from_geometry(
            lat=lat,
            footprint_area_m2=footprint_area_m2,
            construction_year=construction_year,
            known_roof_type=known_roof_type,
            existing_solar_kwp=existing_solar_kwp,
        )

    def _analyze_google_solar(
        self,
        lat: float,
        lon: float,
        footprint_area_m2: float,
    ) -> Optional[RoofAnalysis]:
        """
        Use Google Solar API for roof analysis.

        API provides:
        - Roof segments with pitch and azimuth
        - Annual solar flux per segment
        - Optimal panel placement
        - Shading analysis
        """
        if not self.google_api_key:
            return None

        # Google Solar API endpoints
        # https://developers.google.com/maps/documentation/solar

        # 1. Building Insights request
        url = "https://solar.googleapis.com/v1/buildingInsights:findClosest"
        params = {
            "location.latitude": lat,
            "location.longitude": lon,
            "requiredQuality": "HIGH",
            "key": self.google_api_key,
        }

        try:
            response = requests.get(url, params=params, timeout=30)

            if response.status_code == 404:
                console.print("  [dim]No Google Solar data for this location[/dim]")
                return None

            response.raise_for_status()
            data = response.json()

            return self._parse_google_solar_response(data, footprint_area_m2)

        except requests.RequestException as e:
            console.print(f"  [yellow]Google Solar API error: {e}[/yellow]")
            return None

    def _parse_google_solar_response(
        self,
        data: Dict[str, Any],
        footprint_area_m2: float,
    ) -> RoofAnalysis:
        """Parse Google Solar API response into RoofAnalysis."""

        solar_potential = data.get("solarPotential", {})
        roof_segments = solar_potential.get("roofSegmentStats", [])

        # Parse segments
        segments = []
        total_area = 0
        total_usable = 0
        weighted_azimuth = 0
        weighted_pitch = 0

        for i, seg in enumerate(roof_segments):
            stats = seg.get("stats", {})
            area = stats.get("areaMeters2", 0)
            pitch = seg.get("pitchDegrees", 0)
            azimuth = seg.get("azimuthDegrees", 180)
            sunshine_hours = stats.get("sunshineQuantiles", [0]*12)

            # Calculate usable area (Google provides this)
            usable = area * 0.85  # Approximate

            # PV efficiency factor based on orientation
            efficiency = self._orientation_efficiency(pitch, azimuth, 59.3)

            segment = RoofSegment(
                segment_id=f"seg_{i}",
                area_m2=area,
                azimuth_deg=azimuth,
                pitch_deg=pitch,
                usable_area_m2=usable,
                pv_suitable=area > 5,  # Min 5m² segment
                pv_efficiency_factor=efficiency,
                annual_irradiance_kwh_m2=sum(sunshine_hours) / len(sunshine_hours) * 1000,
            )
            segments.append(segment)

            total_area += area
            total_usable += usable
            weighted_azimuth += azimuth * area
            weighted_pitch += pitch * area

        # Determine roof type from segments
        if len(segments) == 1 and segments[0].pitch_deg < 5:
            roof_type = RoofType.FLAT
        elif len(segments) == 2:
            roof_type = RoofType.GABLED
        elif len(segments) >= 4:
            roof_type = RoofType.HIPPED
        else:
            roof_type = RoofType.PITCHED

        # Get max panel capacity from Google
        max_panels = solar_potential.get("maxArrayPanelsCount", 0)
        panel_capacity = solar_potential.get("panelCapacityWatts", 400)
        max_capacity_kwp = max_panels * panel_capacity / 1000

        # Annual generation
        solar_configs = solar_potential.get("solarPanelConfigs", [])
        if solar_configs:
            best_config = max(solar_configs, key=lambda x: x.get("panelsCount", 0))
            annual_kwh = best_config.get("yearlyEnergyDcKwh", 0)
        else:
            annual_kwh = max_capacity_kwp * 900  # Estimate

        # Primary orientation
        primary_azimuth = weighted_azimuth / total_area if total_area > 0 else 180
        primary_pitch = weighted_pitch / total_area if total_area > 0 else 0

        return RoofAnalysis(
            total_area_m2=total_area or footprint_area_m2,
            roof_type=roof_type,
            primary_azimuth_deg=primary_azimuth,
            primary_pitch_deg=primary_pitch,
            segments=segments,
            gross_available_m2=total_usable,
            net_available_m2=total_usable * 0.9,  # After setbacks
            optimal_capacity_kwp=max_capacity_kwp,
            annual_yield_kwh_per_kwp=annual_kwh / max_capacity_kwp if max_capacity_kwp > 0 else 900,
            annual_generation_potential_kwh=annual_kwh,
            data_source="google_solar_api",
            confidence=0.9,
        )

    def _analyze_from_osm(
        self,
        osm_tags: Dict[str, str],
        footprint_area_m2: float,
        latitude: float,
    ) -> Optional[RoofAnalysis]:
        """
        Analyze roof from OSM building tags.

        Relevant tags:
        - roof:shape (flat, gabled, hipped, etc.)
        - roof:orientation (along, across)
        - roof:levels
        - roof:material
        - building:levels (to estimate height)
        """
        roof_shape = osm_tags.get("roof:shape", "").lower()

        # Map OSM shapes to our types
        shape_map = {
            "flat": RoofType.FLAT,
            "gabled": RoofType.GABLED,
            "hipped": RoofType.HIPPED,
            "pitched": RoofType.PITCHED,
            "mansard": RoofType.MANSARD,
            "pyramidal": RoofType.HIPPED,
            "half-hipped": RoofType.HIPPED,
            "skillion": RoofType.PITCHED,
            "gambrel": RoofType.MANSARD,
        }

        roof_type = shape_map.get(roof_shape, RoofType.UNKNOWN)

        if roof_type == RoofType.UNKNOWN:
            return None

        # Estimate pitch based on type
        pitch_estimates = {
            RoofType.FLAT: 0,
            RoofType.PITCHED: 25,
            RoofType.GABLED: 30,
            RoofType.HIPPED: 25,
            RoofType.MANSARD: 45,
        }
        pitch = pitch_estimates.get(roof_type, 0)

        # Calculate usable area
        if roof_type == RoofType.FLAT:
            usable = footprint_area_m2 * 0.70  # Setbacks + spacing
        else:
            # Pitched roofs: only south-facing half is optimal
            usable = footprint_area_m2 * 0.40

        # Estimate capacity
        capacity_kwp = usable * self.PANEL_POWER_DENSITY_WP_M2 / 1000

        # Annual yield based on latitude
        yield_kwh_kwp = self._get_yield_for_latitude(latitude)

        return RoofAnalysis(
            total_area_m2=footprint_area_m2,
            roof_type=roof_type,
            primary_azimuth_deg=180,  # Assume south
            primary_pitch_deg=pitch,
            gross_available_m2=usable,
            net_available_m2=usable * 0.95,
            optimal_capacity_kwp=capacity_kwp,
            annual_yield_kwh_per_kwp=yield_kwh_kwp,
            annual_generation_potential_kwh=capacity_kwp * yield_kwh_kwp,
            data_source="osm_tags",
            confidence=0.6,
        )

    def _estimate_from_geometry(
        self,
        lat: float,
        footprint_area_m2: float,
        construction_year: int = None,
        known_roof_type: RoofType = None,
        existing_solar_kwp: float = 0,
    ) -> RoofAnalysis:
        """
        Estimate roof characteristics from geometry and era.

        Swedish buildings by era:
        - Pre-1960: Often gabled/hipped roofs
        - 1960-1980: Mix of flat and low-pitch
        - 1980+: More flat roofs on multi-family
        """
        # Determine roof type from era
        if known_roof_type:
            roof_type = known_roof_type
        elif construction_year:
            if construction_year < 1960:
                roof_type = RoofType.GABLED
            elif construction_year < 1990:
                roof_type = RoofType.FLAT  # Million program + post
            else:
                roof_type = RoofType.FLAT  # Modern multi-family
        else:
            roof_type = RoofType.FLAT  # Default assumption

        # Set pitch based on type
        pitch_map = {
            RoofType.FLAT: 0,
            RoofType.PITCHED: 25,
            RoofType.GABLED: 30,
            RoofType.HIPPED: 25,
            RoofType.MANSARD: 45,
        }
        pitch = pitch_map.get(roof_type, 0)

        # Calculate areas
        if roof_type == RoofType.FLAT:
            # Flat roof calculations
            total_area = footprint_area_m2

            # Typical obstructions on Swedish flat roofs
            hvac_area = footprint_area_m2 * 0.03  # ~3% HVAC
            access_area = footprint_area_m2 * 0.05  # ~5% access paths
            skylight_area = footprint_area_m2 * 0.02  # ~2% skylights
            total_obstruction = hvac_area + access_area + skylight_area

            # Setback from edges (fire code)
            perimeter = 4 * math.sqrt(footprint_area_m2)  # Approximate
            setback_area = perimeter * self.FLAT_ROOF_SETBACK_M

            # Panel spacing for flat roofs (avoid self-shading)
            gross_available = footprint_area_m2 - total_obstruction - setback_area
            net_available = gross_available * (1 - self.PANEL_SPACING_FACTOR)

            obstructions = [
                RoofObstruction(ObstructionType.HVAC_UNIT, hvac_area),
                RoofObstruction(ObstructionType.ACCESS_HATCH, access_area),
                RoofObstruction(ObstructionType.SKYLIGHT, skylight_area),
            ]

        else:
            # Pitched roof: area increases with slope
            slope_factor = 1 / math.cos(math.radians(pitch))
            total_area = footprint_area_m2 * slope_factor

            # Only south-facing portion is optimal
            south_facing = total_area * 0.5

            # Fewer obstructions on pitched roofs
            chimney_area = 1.0 if construction_year and construction_year < 1990 else 0
            skylight_area = footprint_area_m2 * 0.01
            total_obstruction = chimney_area + skylight_area

            setback_area = 0  # Less strict on pitched
            gross_available = south_facing - total_obstruction
            net_available = gross_available * 0.95

            obstructions = []
            if chimney_area > 0:
                obstructions.append(RoofObstruction(ObstructionType.CHIMNEY, chimney_area))
            if skylight_area > 0:
                obstructions.append(RoofObstruction(ObstructionType.SKYLIGHT, skylight_area))

        # Subtract existing solar
        existing_solar = None
        if existing_solar_kwp > 0:
            existing_area = existing_solar_kwp * 1000 / self.PANEL_POWER_DENSITY_WP_M2
            net_available = max(0, net_available - existing_area)
            existing_solar = ExistingSolarInstallation(
                area_m2=existing_area,
                capacity_kwp=existing_solar_kwp,
                detected_from="declaration",
            )

        # Calculate new capacity
        optimal_capacity = net_available * self.PANEL_POWER_DENSITY_WP_M2 / 1000

        # Annual yield
        yield_kwh_kwp = self._get_yield_for_latitude(lat)

        # Adjust for orientation (assume 180° = south for estimation)
        orientation_factor = self._orientation_efficiency(pitch, 180, lat)
        effective_yield = yield_kwh_kwp * orientation_factor * (1 - self.SYSTEM_LOSSES)

        annual_generation = optimal_capacity * effective_yield

        return RoofAnalysis(
            total_area_m2=total_area,
            roof_type=roof_type,
            primary_azimuth_deg=180,
            primary_pitch_deg=pitch,
            obstructions=obstructions,
            total_obstruction_area_m2=total_obstruction,
            existing_solar=existing_solar,
            gross_available_m2=gross_available,
            setback_area_m2=setback_area,
            access_path_area_m2=access_area if roof_type == RoofType.FLAT else 0,
            net_available_m2=net_available,
            optimal_capacity_kwp=optimal_capacity,
            annual_yield_kwh_per_kwp=effective_yield,
            annual_generation_potential_kwh=annual_generation,
            average_shading_loss_pct=0,  # Not calculated in estimation
            data_source="geometry_estimation",
            confidence=0.5,
        )

    def _get_yield_for_latitude(self, lat: float) -> float:
        """Interpolate annual yield for latitude."""
        # Find closest reference points
        lats = sorted(self.LATITUDE_IRRADIANCE.keys())

        if lat <= lats[0]:
            return self.LATITUDE_IRRADIANCE[lats[0]][0] * 0.85  # 85% of irradiance
        if lat >= lats[-1]:
            return self.LATITUDE_IRRADIANCE[lats[-1]][0] * 0.85

        # Linear interpolation
        for i in range(len(lats) - 1):
            if lats[i] <= lat <= lats[i + 1]:
                ratio = (lat - lats[i]) / (lats[i + 1] - lats[i])
                irr1 = self.LATITUDE_IRRADIANCE[lats[i]][0]
                irr2 = self.LATITUDE_IRRADIANCE[lats[i + 1]][0]
                return (irr1 + ratio * (irr2 - irr1)) * 0.85

        return 900 * 0.85  # Default Stockholm-ish

    def _orientation_efficiency(
        self,
        pitch: float,
        azimuth: float,
        latitude: float,
    ) -> float:
        """
        Calculate efficiency factor for panel orientation.

        Optimal in Sweden: ~40-45° tilt, 180° azimuth (south)

        Returns factor 0-1 where 1 = optimal.
        """
        # Optimal tilt is roughly latitude - 10-15°
        optimal_tilt = latitude - 15

        # Tilt penalty
        tilt_diff = abs(pitch - optimal_tilt)
        if tilt_diff <= 10:
            tilt_factor = 1.0
        elif tilt_diff <= 30:
            tilt_factor = 1.0 - (tilt_diff - 10) * 0.01  # 1% per degree
        else:
            tilt_factor = 0.8

        # Azimuth penalty (south = 180° is optimal)
        azimuth_diff = abs(azimuth - 180)
        if azimuth_diff <= 30:
            azimuth_factor = 1.0
        elif azimuth_diff <= 90:
            azimuth_factor = 1.0 - (azimuth_diff - 30) * 0.005
        else:
            azimuth_factor = 0.7

        return tilt_factor * azimuth_factor


def analyze_roof(
    lat: float,
    lon: float,
    footprint_area_m2: float,
    **kwargs,
) -> RoofAnalysis:
    """
    Convenience function to analyze roof.

    Args:
        lat, lon: Building coordinates
        footprint_area_m2: Building footprint
        **kwargs: Additional arguments for RoofAnalyzer.analyze()

    Returns:
        RoofAnalysis
    """
    analyzer = RoofAnalyzer()
    return analyzer.analyze(lat, lon, footprint_area_m2, **kwargs)
