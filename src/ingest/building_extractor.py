"""
Unified Building Data Extractor

Orchestrates data extraction from multiple sources:
- OSM/Overture: footprint, height, floors, materials
- Mapillary: street-level images for facade analysis
- AI Analysis: WWR, material classification
- Geometry Calculator: wall areas per orientation
- Archetype Matcher: U-values, HVAC properties

Input: Address, coordinates, or BRF declaration
Output: Complete BuildingProfile ready for EnergyPlus baseline generation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..core.config import settings
from ..geometry.building_geometry import BuildingGeometryCalculator, BuildingGeometry
from ..baseline.archetypes import ArchetypeMatcher, SwedishArchetype, BuildingType
from .osm_fetcher import OSMFetcher
from .overture_fetcher import OvertureFetcher

console = Console()


@dataclass
class DataSource:
    """Track data source and confidence."""
    source: str  # 'osm', 'overture', 'mapillary', 'ai', 'archetype', 'manual'
    confidence: float  # 0.0 - 1.0
    notes: List[str] = field(default_factory=list)


@dataclass
class EnvelopeData:
    """Building envelope thermal properties."""
    # U-values (W/m2K)
    wall_u: float
    roof_u: float
    floor_u: float
    window_u: float

    # Window-to-wall ratios per orientation
    wwr_north: float
    wwr_south: float
    wwr_east: float
    wwr_west: float
    wwr_average: float

    # Window properties
    window_shgc: float

    # Airtightness
    infiltration_ach: float  # ACH at 50Pa / 20

    # Facade material
    facade_material: str  # 'brick', 'concrete', 'render', 'wood', 'glass'
    facade_material_confidence: float

    # Source tracking
    u_value_source: DataSource
    wwr_source: DataSource
    material_source: DataSource


@dataclass
class HVACData:
    """HVAC system properties."""
    heating_system: str  # 'district', 'heat_pump_ground', etc.
    ventilation_type: str  # 'ftx', 'exhaust', 'natural'
    heat_recovery_efficiency: float  # 0.0 - 0.90
    ventilation_rate_l_s_m2: float  # BBR default 0.35
    sfp_kw_per_m3s: float  # Specific fan power

    source: DataSource


@dataclass
class SolarPotentialData:
    """Solar PV potential on roof."""
    # Roof characteristics
    roof_type: str  # 'flat', 'pitched', 'gabled'
    roof_area_total_m2: float
    roof_pitch_deg: float
    roof_azimuth_deg: float  # Primary orientation

    # Available space
    gross_available_m2: float  # After obstructions
    net_available_m2: float    # After setbacks and spacing

    # Obstructions detected
    obstruction_area_m2: float
    obstruction_types: List[str]  # ['hvac', 'skylight', etc.]

    # Existing solar
    existing_pv_area_m2: float
    existing_pv_kwp: float
    existing_pv_kwh_year: float

    # New PV potential
    new_capacity_kwp: float
    annual_yield_kwh_per_kwp: float
    annual_generation_kwh: float

    # Shading
    shading_loss_pct: float
    neighbor_count: int

    # Financial estimate
    install_cost_sek: float
    payback_years: float

    # Source tracking
    source: DataSource


@dataclass
class GeometryData:
    """Building geometry data."""
    # Footprint
    footprint_coords_wgs84: List[Tuple[float, float]]
    footprint_area_m2: float
    perimeter_m: float

    # Dimensions
    height_m: float
    floors: int
    floor_height_m: float
    gross_floor_area_m2: float

    # Wall areas per orientation (m2)
    wall_area_north: float
    wall_area_south: float
    wall_area_east: float
    wall_area_west: float
    total_wall_area: float

    # Window areas per orientation (m2)
    window_area_north: float
    window_area_south: float
    window_area_east: float
    window_area_west: float
    total_window_area: float

    # Roof
    roof_area_m2: float
    roof_type: str  # 'flat', 'pitched', 'gabled'
    roof_pv_potential_m2: float

    # Volume
    volume_m3: float

    # Source tracking
    footprint_source: DataSource
    height_source: DataSource


@dataclass
class EnergyDeclarationData:
    """Data from Swedish energy declaration (Energideklaration)."""
    energy_class: str  # A-G
    declared_energy_kwh_m2: float  # Primary energy
    specific_energy_kwh_m2: Optional[float]  # Actual heating
    construction_year: int

    # Energy breakdown if available
    heating_kwh: Optional[float] = None
    dhw_kwh: Optional[float] = None
    electricity_kwh: Optional[float] = None

    source: DataSource = field(default_factory=lambda: DataSource("declaration", 0.95))


@dataclass
class BuildingProfile:
    """
    Complete building profile ready for EnergyPlus baseline generation.

    This is the main output of BuildingDataExtractor.
    """
    # Identification
    building_id: str
    address: str

    # Location
    latitude: float
    longitude: float
    climate_zone: int  # Swedish climate zone 1-4

    # Basic info
    construction_year: int
    building_type: str  # 'multi_family', 'single_family', etc.

    # Geometry (calculated from footprint)
    geometry: GeometryData

    # Envelope (from archetype + AI detection)
    envelope: EnvelopeData

    # HVAC (from archetype + declaration)
    hvac: HVACData

    # Energy declaration (if available)
    energy_declaration: Optional[EnergyDeclarationData] = None

    # Solar potential (from roof analysis)
    solar_potential: Optional[SolarPotentialData] = None

    # Matched archetype
    archetype_name: str = ""
    archetype: Optional[SwedishArchetype] = None

    # Raw data for debugging
    raw_osm_data: Optional[Dict[str, Any]] = None
    raw_overture_data: Optional[Dict[str, Any]] = None
    mapillary_images: List[str] = field(default_factory=list)

    # Overall extraction confidence
    overall_confidence: float = 0.0
    extraction_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "building_id": self.building_id,
            "address": self.address,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "construction_year": self.construction_year,
            "building_type": self.building_type,
            "geometry": {
                "footprint_area_m2": self.geometry.footprint_area_m2,
                "gross_floor_area_m2": self.geometry.gross_floor_area_m2,
                "height_m": self.geometry.height_m,
                "floors": self.geometry.floors,
                "perimeter_m": self.geometry.perimeter_m,
                "wall_areas": {
                    "north": self.geometry.wall_area_north,
                    "south": self.geometry.wall_area_south,
                    "east": self.geometry.wall_area_east,
                    "west": self.geometry.wall_area_west,
                    "total": self.geometry.total_wall_area,
                },
                "window_areas": {
                    "north": self.geometry.window_area_north,
                    "south": self.geometry.window_area_south,
                    "east": self.geometry.window_area_east,
                    "west": self.geometry.window_area_west,
                    "total": self.geometry.total_window_area,
                },
                "roof_area_m2": self.geometry.roof_area_m2,
                "volume_m3": self.geometry.volume_m3,
            },
            "envelope": {
                "u_values": {
                    "wall": self.envelope.wall_u,
                    "roof": self.envelope.roof_u,
                    "floor": self.envelope.floor_u,
                    "window": self.envelope.window_u,
                },
                "wwr": {
                    "north": self.envelope.wwr_north,
                    "south": self.envelope.wwr_south,
                    "east": self.envelope.wwr_east,
                    "west": self.envelope.wwr_west,
                    "average": self.envelope.wwr_average,
                },
                "window_shgc": self.envelope.window_shgc,
                "infiltration_ach": self.envelope.infiltration_ach,
                "facade_material": self.envelope.facade_material,
            },
            "hvac": {
                "heating_system": self.hvac.heating_system,
                "ventilation_type": self.hvac.ventilation_type,
                "heat_recovery_efficiency": self.hvac.heat_recovery_efficiency,
                "ventilation_rate_l_s_m2": self.hvac.ventilation_rate_l_s_m2,
            },
            "solar_potential": {
                "roof_type": self.solar_potential.roof_type,
                "roof_area_m2": self.solar_potential.roof_area_total_m2,
                "available_area_m2": self.solar_potential.net_available_m2,
                "obstruction_area_m2": self.solar_potential.obstruction_area_m2,
                "existing_pv_kwp": self.solar_potential.existing_pv_kwp,
                "new_capacity_kwp": self.solar_potential.new_capacity_kwp,
                "annual_generation_kwh": self.solar_potential.annual_generation_kwh,
                "shading_loss_pct": self.solar_potential.shading_loss_pct,
                "install_cost_sek": self.solar_potential.install_cost_sek,
                "payback_years": self.solar_potential.payback_years,
            } if self.solar_potential else None,
            "archetype_name": self.archetype_name,
            "overall_confidence": self.overall_confidence,
            "extraction_notes": self.extraction_notes,
        }


class BuildingDataExtractor:
    """
    Extract complete building data from multiple sources.

    Pipeline:
    1. Geocode address (if needed)
    2. Fetch footprint from OSM/Overture
    3. Get height/floors from OSM/Overture
    4. Get facade material from OSM/Overture or AI
    5. Fetch Mapillary images
    6. Run AI facade analysis (WWR, material)
    7. Calculate geometry (wall areas per orientation)
    8. Analyze roof for solar PV potential
    9. Match to Swedish archetype
    10. Compile BuildingProfile

    Usage:
        extractor = BuildingDataExtractor()
        profile = extractor.extract_from_coordinates(
            lat=59.302, lon=18.104,
            construction_year=2003
        )
    """

    def __init__(
        self,
        use_ai: bool = True,
        ai_backend: str = "opencv",  # 'opencv', 'sam', 'lang_sam'
        cache_dir: Path = None,
    ):
        """
        Initialize extractor.

        Args:
            use_ai: Whether to use AI for WWR/material detection
            ai_backend: AI backend for facade analysis
            cache_dir: Directory for caching API responses
        """
        self.use_ai = use_ai
        self.ai_backend = ai_backend
        self.cache_dir = cache_dir or settings.cache_dir

        # Initialize fetchers
        self.osm_fetcher = OSMFetcher(cache_dir=self.cache_dir / "osm")
        self.overture_fetcher = OvertureFetcher(cache_dir=self.cache_dir / "overture")
        self.geometry_calculator = BuildingGeometryCalculator()
        self.archetype_matcher = ArchetypeMatcher()

        # Lazy-load AI modules
        self._facade_analyzer = None
        self._image_fetcher = None
        self._roof_analyzer = None

    @property
    def facade_analyzer(self):
        """Lazy-load facade analyzer."""
        if self._facade_analyzer is None:
            from ..ai.facade_analyzer import FacadeAnalyzer
            self._facade_analyzer = FacadeAnalyzer(
                device="cpu",
                wwr_backend=self.ai_backend,
            )
        return self._facade_analyzer

    @property
    def image_fetcher(self):
        """Lazy-load image fetcher."""
        if self._image_fetcher is None:
            from .image_fetcher import FacadeImageFetcher
            self._image_fetcher = FacadeImageFetcher(
                cache_dir=self.cache_dir / "images"
            )
        return self._image_fetcher

    @property
    def roof_analyzer(self):
        """Lazy-load roof analyzer."""
        if self._roof_analyzer is None:
            from ..analysis.roof_analyzer import RoofAnalyzer
            self._roof_analyzer = RoofAnalyzer(
                cache_dir=self.cache_dir / "roof_analysis"
            )
        return self._roof_analyzer

    def extract_from_coordinates(
        self,
        lat: float,
        lon: float,
        construction_year: int,
        address: str = "",
        building_id: str = "",
        height_m: float = None,
        floors: int = None,
        energy_declaration: EnergyDeclarationData = None,
        fetch_images: bool = True,
        run_ai_analysis: bool = True,
    ) -> BuildingProfile:
        """
        Extract building data from coordinates.

        Args:
            lat, lon: Building center coordinates (WGS84)
            construction_year: Year building was constructed
            address: Address string (optional)
            building_id: Unique ID (optional, generated if not provided)
            height_m: Known height (overrides detected)
            floors: Known floor count (overrides detected)
            energy_declaration: Energy declaration data (if available)
            fetch_images: Whether to fetch Mapillary images
            run_ai_analysis: Whether to run AI facade analysis

        Returns:
            Complete BuildingProfile
        """
        notes = []
        building_id = building_id or f"bldg_{lat:.6f}_{lon:.6f}"

        console.print(f"\n[bold cyan]Extracting building data[/bold cyan]")
        console.print(f"  Location: {lat:.6f}, {lon:.6f}")
        console.print(f"  Construction year: {construction_year}")

        # Step 1: Fetch footprint from OSM/Overture
        console.print("\n[cyan]1. Fetching building footprint...[/cyan]")
        footprint_data = self._fetch_footprint(lat, lon)

        if footprint_data is None:
            raise ValueError(f"Could not find building footprint at {lat}, {lon}")

        footprint_coords = footprint_data["coordinates"]
        notes.append(f"Footprint from {footprint_data['source']}")

        # Step 2: Get height and floors
        console.print("[cyan]2. Getting building dimensions...[/cyan]")
        detected_height = footprint_data.get("height")
        detected_floors = footprint_data.get("levels")

        if height_m is None:
            height_m = detected_height or self._estimate_height(floors or detected_floors, construction_year)
            notes.append(f"Height: {'detected' if detected_height else 'estimated'} = {height_m}m")

        if floors is None:
            floors = detected_floors or self._estimate_floors(height_m, construction_year)
            notes.append(f"Floors: {'detected' if detected_floors else 'estimated'} = {floors}")

        # Step 3: Get facade material
        console.print("[cyan]3. Detecting facade material...[/cyan]")
        facade_material = footprint_data.get("facade_material") or footprint_data.get("material")
        material_source = "osm" if facade_material else None
        material_confidence = 0.8 if facade_material else 0.0

        # Step 4: Match archetype (initial, may be refined with AI material detection)
        console.print("[cyan]4. Matching to Swedish archetype...[/cyan]")
        archetype = self.archetype_matcher.match(
            construction_year=construction_year,
            building_type=BuildingType.MULTI_FAMILY,
            facade_material=facade_material,
        )
        console.print(f"  Matched: [green]{archetype.name}[/green]")
        notes.append(f"Archetype: {archetype.name}")

        # Step 5: Get WWR (from archetype initially)
        wwr_by_orientation = {
            'N': archetype.typical_wwr * 0.8,  # North typically less glass
            'S': archetype.typical_wwr * 1.2,  # South typically more glass
            'E': archetype.typical_wwr,
            'W': archetype.typical_wwr,
        }
        wwr_source = DataSource("archetype", 0.5, ["Based on construction era"])

        # Step 6: Fetch Mapillary images (optional)
        mapillary_images = []
        if fetch_images and self.use_ai:
            console.print("[cyan]5. Fetching street-level images...[/cyan]")
            try:
                mapillary_images = self._fetch_mapillary_images(lat, lon, footprint_coords)
                if mapillary_images:
                    notes.append(f"Fetched {len(mapillary_images)} Mapillary images")
            except Exception as e:
                console.print(f"  [yellow]Image fetch failed: {e}[/yellow]")
                notes.append(f"Mapillary fetch failed: {e}")

        # Step 7: Run AI analysis (optional)
        if run_ai_analysis and self.use_ai and mapillary_images:
            console.print("[cyan]6. Running AI facade analysis...[/cyan]")
            try:
                ai_result = self.facade_analyzer.analyze_building(
                    facade_images={img["direction"]: img["path"] for img in mapillary_images},
                    construction_year=construction_year,
                    use_ai=True,
                )

                if ai_result.wwr:
                    wwr_by_orientation = {
                        'N': ai_result.wwr.north,
                        'S': ai_result.wwr.south,
                        'E': ai_result.wwr.east,
                        'W': ai_result.wwr.west,
                    }
                    wwr_source = DataSource("ai", ai_result.wwr.confidence, [f"Backend: {self.ai_backend}"])
                    notes.append(f"AI WWR detection (confidence: {ai_result.wwr.confidence:.2f})")

                if ai_result.facade_material.value != "unknown":
                    facade_material = ai_result.facade_material.value
                    material_confidence = ai_result.material_confidence
                    material_source = "ai"
                    notes.append(f"AI material: {facade_material} (confidence: {material_confidence:.2f})")

                    # Re-match archetype with detected material
                    archetype = self.archetype_matcher.match(
                        construction_year=construction_year,
                        building_type=BuildingType.MULTI_FAMILY,
                        facade_material=facade_material,
                    )

            except Exception as e:
                console.print(f"  [yellow]AI analysis failed: {e}[/yellow]")
                notes.append(f"AI analysis failed: {e}")

        # Step 8: Calculate geometry
        console.print("[cyan]7. Calculating building geometry...[/cyan]")
        geometry = self.geometry_calculator.calculate(
            footprint_coords=footprint_coords,
            height_m=height_m,
            floors=floors,
            wwr_by_orientation=wwr_by_orientation,
        )

        # Step 9: Analyze roof for solar potential
        console.print("[cyan]8. Analyzing roof for solar potential...[/cyan]")
        solar_potential = None
        try:
            # Get existing solar info from energy declaration if available
            existing_solar_kwp = 0
            if energy_declaration and hasattr(energy_declaration, 'declared_energy_kwh_m2'):
                # Check for existing solar in declaration (simplified)
                pass

            roof_analysis = self.roof_analyzer.analyze(
                lat=lat,
                lon=lon,
                footprint_area_m2=geometry.footprint_area_m2,
                building_height_m=height_m,
                construction_year=construction_year,
                existing_solar_kwp=existing_solar_kwp,
                osm_roof_tags=footprint_data.get("raw_osm", {}).get("tags", {}),
            )

            # Convert to SolarPotentialData
            solar_potential = SolarPotentialData(
                roof_type=roof_analysis.roof_type.value,
                roof_area_total_m2=roof_analysis.total_area_m2,
                roof_pitch_deg=roof_analysis.primary_pitch_deg,
                roof_azimuth_deg=roof_analysis.primary_azimuth_deg,
                gross_available_m2=roof_analysis.gross_available_m2,
                net_available_m2=roof_analysis.net_available_m2,
                obstruction_area_m2=roof_analysis.total_obstruction_area_m2,
                obstruction_types=[o.type.value for o in roof_analysis.obstructions],
                existing_pv_area_m2=roof_analysis.existing_solar.area_m2 if roof_analysis.existing_solar else 0,
                existing_pv_kwp=roof_analysis.existing_solar.capacity_kwp if roof_analysis.existing_solar else 0,
                existing_pv_kwh_year=roof_analysis.existing_solar.annual_production_kwh or 0 if roof_analysis.existing_solar else 0,
                new_capacity_kwp=roof_analysis.optimal_capacity_kwp,
                annual_yield_kwh_per_kwp=roof_analysis.annual_yield_kwh_per_kwp,
                annual_generation_kwh=roof_analysis.annual_generation_potential_kwh,
                shading_loss_pct=roof_analysis.average_shading_loss_pct,
                neighbor_count=0,  # Would need OSM analysis
                install_cost_sek=roof_analysis.optimal_capacity_kwp * 12000,  # ~12k SEK/kWp
                payback_years=roof_analysis.optimal_capacity_kwp * 12000 / (roof_analysis.annual_generation_potential_kwh * 1.5) if roof_analysis.annual_generation_potential_kwh > 0 else 0,
                source=DataSource(roof_analysis.data_source, roof_analysis.confidence),
            )
            notes.append(f"Roof analysis: {roof_analysis.data_source} (confidence: {roof_analysis.confidence:.0%})")
            console.print(f"  PV potential: {roof_analysis.optimal_capacity_kwp:.1f} kWp, {roof_analysis.annual_generation_potential_kwh:,.0f} kWh/year")

        except Exception as e:
            console.print(f"  [yellow]Roof analysis failed: {e}[/yellow]")
            notes.append(f"Roof analysis failed: {e}")

        # Step 10: Build profile
        console.print("[cyan]9. Compiling building profile...[/cyan]")

        # Calculate average WWR
        wwr_avg = sum(wwr_by_orientation.values()) / 4

        # Use archetype for defaults
        envelope = EnvelopeData(
            wall_u=archetype.envelope.wall_u_value,
            roof_u=archetype.envelope.roof_u_value,
            floor_u=archetype.envelope.floor_u_value,
            window_u=archetype.envelope.window_u_value,
            wwr_north=wwr_by_orientation['N'],
            wwr_south=wwr_by_orientation['S'],
            wwr_east=wwr_by_orientation['E'],
            wwr_west=wwr_by_orientation['W'],
            wwr_average=wwr_avg,
            window_shgc=archetype.envelope.window_shgc,
            infiltration_ach=archetype.envelope.infiltration_ach,
            facade_material=facade_material or "unknown",
            facade_material_confidence=material_confidence,
            u_value_source=DataSource("archetype", 0.7, [f"Era: {archetype.name}"]),
            wwr_source=wwr_source,
            material_source=DataSource(material_source or "unknown", material_confidence),
        )

        hvac = HVACData(
            heating_system=archetype.hvac.heating_system.value,
            ventilation_type=archetype.hvac.ventilation_type.value,
            heat_recovery_efficiency=archetype.hvac.heat_recovery_efficiency,
            ventilation_rate_l_s_m2=archetype.hvac.ventilation_rate_l_s_m2,
            sfp_kw_per_m3s=archetype.hvac.sfp_kw_per_m3s,
            source=DataSource("archetype", 0.6, [f"Era: {archetype.name}"]),
        )

        geometry_data = GeometryData(
            footprint_coords_wgs84=footprint_coords,
            footprint_area_m2=geometry.footprint_area_m2,
            perimeter_m=geometry.perimeter_m,
            height_m=height_m,
            floors=floors,
            floor_height_m=geometry.floor_height_m,
            gross_floor_area_m2=geometry.gross_floor_area_m2,
            wall_area_north=geometry.facades['N'].wall_area_m2,
            wall_area_south=geometry.facades['S'].wall_area_m2,
            wall_area_east=geometry.facades['E'].wall_area_m2,
            wall_area_west=geometry.facades['W'].wall_area_m2,
            total_wall_area=geometry.total_wall_area_m2,
            window_area_north=geometry.facades['N'].window_area_m2,
            window_area_south=geometry.facades['S'].window_area_m2,
            window_area_east=geometry.facades['E'].window_area_m2,
            window_area_west=geometry.facades['W'].window_area_m2,
            total_window_area=geometry.total_window_area_m2,
            roof_area_m2=geometry.roof.total_area_m2,
            roof_type="flat" if geometry.roof.primary_slope_deg < 5 else "pitched",
            roof_pv_potential_m2=geometry.roof.available_pv_area_m2,
            volume_m3=geometry.volume_m3,
            footprint_source=DataSource(footprint_data['source'], 0.9),
            height_source=DataSource("detected" if detected_height else "estimated", 0.7 if detected_height else 0.5),
        )

        # Calculate overall confidence
        confidences = [
            geometry_data.footprint_source.confidence,
            geometry_data.height_source.confidence,
            envelope.wwr_source.confidence,
            envelope.material_source.confidence,
        ]
        overall_confidence = sum(confidences) / len(confidences)

        profile = BuildingProfile(
            building_id=building_id,
            address=address,
            latitude=lat,
            longitude=lon,
            climate_zone=self._get_climate_zone(lat),
            construction_year=construction_year,
            building_type="multi_family",
            geometry=geometry_data,
            envelope=envelope,
            hvac=hvac,
            energy_declaration=energy_declaration,
            solar_potential=solar_potential,
            archetype_name=archetype.name,
            archetype=archetype,
            raw_osm_data=footprint_data.get("raw_osm"),
            raw_overture_data=footprint_data.get("raw_overture"),
            mapillary_images=[img["path"] for img in mapillary_images],
            overall_confidence=overall_confidence,
            extraction_notes=notes,
        )

        console.print(f"\n[bold green]Extraction complete![/bold green]")
        console.print(f"  Confidence: {overall_confidence:.1%}")
        console.print(f"  GFA: {geometry.gross_floor_area_m2:,.0f} m2")
        console.print(f"  Wall area: {geometry.total_wall_area_m2:,.0f} m2")
        console.print(f"  WWR: {wwr_avg:.1%}")

        return profile

    def extract_from_footprint(
        self,
        footprint_coords: List[Tuple[float, float]],
        construction_year: int,
        height_m: float = None,
        floors: int = None,
        **kwargs,
    ) -> BuildingProfile:
        """
        Extract building data from known footprint coordinates.

        Args:
            footprint_coords: List of (lon, lat) coordinates
            construction_year: Year built
            height_m: Building height (optional)
            floors: Number of floors (optional)
            **kwargs: Additional args passed to extract_from_coordinates

        Returns:
            BuildingProfile
        """
        # Calculate centroid
        lons = [c[0] for c in footprint_coords]
        lats = [c[1] for c in footprint_coords]
        center_lon = sum(lons) / len(lons)
        center_lat = sum(lats) / len(lats)

        # Override footprint fetching
        return self.extract_from_coordinates(
            lat=center_lat,
            lon=center_lon,
            construction_year=construction_year,
            height_m=height_m,
            floors=floors,
            **kwargs,
        )

    def _fetch_footprint(
        self,
        lat: float,
        lon: float,
        search_radius_m: float = 50,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch building footprint from OSM and Overture.

        Tries Overture first (better coverage), falls back to OSM.
        """
        # Calculate bounding box
        # At lat 59: 1 degree lon ~ 56km, 1 degree lat ~ 111km
        lat_offset = search_radius_m / 111000
        lon_offset = search_radius_m / (111000 * 0.51)  # cos(59)

        bbox = (
            lon - lon_offset,
            lat - lat_offset,
            lon + lon_offset,
            lat + lat_offset,
        )

        # Try OSM first (more detailed tags)
        osm_buildings = self.osm_fetcher.get_nearby_buildings(lon, lat, search_radius_m)

        if osm_buildings:
            # Find closest
            closest = min(
                osm_buildings,
                key=lambda b: self._distance_to_centroid(b["coordinates"], lat, lon)
            )

            return {
                "coordinates": closest["coordinates"],
                "source": "osm",
                "height": closest.get("height"),
                "levels": closest.get("levels"),
                "material": closest.get("material"),
                "facade_material": closest.get("facade_material"),
                "roof_material": closest.get("roof_material"),
                "address": closest.get("address"),
                "raw_osm": closest,
            }

        # Try Overture
        try:
            overture_buildings = self.overture_fetcher.get_buildings_in_bbox(*bbox)

            if overture_buildings:
                # Find closest
                closest = None
                min_dist = float("inf")

                for feature in overture_buildings:
                    parsed = self.overture_fetcher.parse_building(feature)
                    if parsed.get("coordinates"):
                        coords = parsed["coordinates"]
                        if isinstance(coords[0][0], list):  # MultiPolygon
                            coords = coords[0][0]
                        else:
                            coords = coords[0]

                        dist = self._distance_to_centroid(coords, lat, lon)
                        if dist < min_dist:
                            min_dist = dist
                            closest = {
                                "coordinates": coords,
                                "source": "overture",
                                "height": parsed.get("height"),
                                "levels": parsed.get("num_floors"),
                                "facade_material": parsed.get("facade_material"),
                                "roof_material": parsed.get("roof_material"),
                                "raw_overture": feature,
                            }

                if closest:
                    return closest

        except Exception as e:
            console.print(f"[yellow]Overture fetch failed: {e}[/yellow]")

        return None

    def _fetch_mapillary_images(
        self,
        lat: float,
        lon: float,
        footprint_coords: List[Tuple[float, float]],
    ) -> List[Dict[str, Any]]:
        """Fetch Mapillary images around building."""
        images = []

        try:
            fetched = self.image_fetcher.fetch_for_building(
                building_coords=footprint_coords,
                center_lat=lat,
                center_lon=lon,
            )

            for direction, img_data in fetched.items():
                if img_data and img_data.get("path"):
                    images.append({
                        "direction": direction,
                        "path": img_data["path"],
                        "source": img_data.get("source", "mapillary"),
                    })

        except Exception as e:
            console.print(f"[yellow]Image fetcher error: {e}[/yellow]")

        return images

    def _distance_to_centroid(
        self,
        coords: List[Tuple[float, float]],
        target_lat: float,
        target_lon: float,
    ) -> float:
        """Calculate distance from polygon centroid to target point."""
        if not coords:
            return float("inf")

        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        centroid_lon = sum(lons) / len(lons)
        centroid_lat = sum(lats) / len(lats)

        # Simple Euclidean distance (good enough for small areas)
        return ((centroid_lon - target_lon)**2 + (centroid_lat - target_lat)**2) ** 0.5

    def _estimate_height(self, floors: int, construction_year: int) -> float:
        """Estimate building height from floors and era."""
        if floors:
            # Floor height varies by era
            if construction_year < 1960:
                floor_height = 3.0
            elif construction_year < 1975:
                floor_height = 2.6
            elif construction_year < 2000:
                floor_height = 2.5
            else:
                floor_height = 2.7
            return floors * floor_height

        # Default to 7-floor building
        return 21.0

    def _estimate_floors(self, height_m: float, construction_year: int) -> int:
        """Estimate floor count from height and era."""
        if height_m:
            if construction_year < 1960:
                floor_height = 3.0
            elif construction_year < 1975:
                floor_height = 2.6
            elif construction_year < 2000:
                floor_height = 2.5
            else:
                floor_height = 2.7
            return max(1, round(height_m / floor_height))

        return 7  # Default

    def _get_climate_zone(self, lat: float) -> int:
        """Get Swedish climate zone from latitude."""
        # Swedish climate zones (simplified)
        if lat >= 66:
            return 1  # Norrland inland
        elif lat >= 63:
            return 2  # Norrland coast
        elif lat >= 60:
            return 3  # Svealand
        else:
            return 4  # Götaland


def extract_building(
    lat: float,
    lon: float,
    construction_year: int,
    **kwargs,
) -> BuildingProfile:
    """
    Convenience function to extract building data.

    Args:
        lat, lon: Building coordinates
        construction_year: Year built
        **kwargs: Additional arguments for BuildingDataExtractor

    Returns:
        BuildingProfile
    """
    extractor = BuildingDataExtractor()
    return extractor.extract_from_coordinates(lat, lon, construction_year, **kwargs)
