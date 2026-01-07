"""
Address-to-Report Pipeline.

The unified entry point for Raiden's vision:
"Given just an address, automatically analyze the building and generate recommendations."

Workflow:
1. Geocode address to coordinates
2. Fetch building footprint from OSM/Overture
3. Fetch street-level images from Mapillary (facade detection)
4. Detect building form (lamellhus, skivhus, etc.)
5. Fetch/generate building characteristics
6. Run baseline simulation with calibration
7. Filter applicable ECMs
8. Run ECM simulations
9. Generate maintenance plan with cash flow cascade
10. Analyze effektvakt potential
11. Generate comprehensive HTML report
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import tempfile
import os
from datetime import datetime
import requests

from ..utils.logging_config import get_logger
from ..utils.validation import (
    validate_address,
    validate_coordinates,
    ValidationError,
    AddressComponents,
)
from ..utils.retry import retry_with_backoff, RetryConfig, RetryableRequest
from .coordinates import CoordinateTransformer

logger = get_logger(__name__)

# Singleton coordinate transformer for SWEREF99 <-> WGS84 conversion
_coord_transformer = None

def _get_coord_transformer() -> CoordinateTransformer:
    """Get or create coordinate transformer (singleton for efficiency)."""
    global _coord_transformer
    if _coord_transformer is None:
        _coord_transformer = CoordinateTransformer()
    return _coord_transformer


def _convert_sweref_footprint_to_wgs84(footprint_coords: list) -> list:
    """
    Convert SWEREF99 3D footprint coordinates to WGS84 2D.

    Sweden Buildings GeoJSON uses SWEREF99 TM (EPSG:3006) with 3D coords:
    [[[x, y, z], [x, y, z], ...], ...]  (nested rings with 3D points)

    We need WGS84 2D for geometry calculations:
    [(lon, lat), (lon, lat), ...]

    Returns:
        List of (lon, lat) tuples for the outer ring only
    """
    if not footprint_coords:
        return []

    transformer = _get_coord_transformer()

    # Extract outer ring (first element)
    try:
        outer_ring = footprint_coords[0] if footprint_coords else []

        # Check if it's already 2D WGS84 format [(lon, lat), ...]
        if outer_ring and isinstance(outer_ring[0], (tuple, list)):
            first_coord = outer_ring[0]
            # SWEREF99 has large values (6-7 digits), WGS84 is small (2-3 digits)
            if len(first_coord) >= 2:
                x_val = abs(first_coord[0])
                # If x coordinate is > 1000, it's SWEREF99 (meters, not degrees)
                if x_val > 1000:
                    # SWEREF99 3D format: [[x, y, z], ...]
                    wgs84_coords = transformer.coords_3d_to_2d_wgs84(outer_ring)
                    return wgs84_coords
                else:
                    # Already WGS84: return as tuples
                    return [(c[0], c[1]) for c in outer_ring]

        return []
    except Exception as e:
        logger.warning(f"Failed to convert footprint coords: {e}")
        return []


@dataclass
class GeocodingResult:
    """Result of geocoding an address."""
    latitude: float
    longitude: float
    display_name: str
    address_components: Dict[str, str] = field(default_factory=dict)
    success: bool = True
    error: str = ""


@dataclass
class BuildingData:
    """Building data assembled from various sources."""
    # Location
    address: str  # Primary/input address
    latitude: float
    longitude: float

    # Property-level data (fastighetsbeteckning)
    # A property can contain multiple buildings/addresses
    property_designation: str = ""  # e.g., "Böljan 2"
    property_owner: str = ""  # e.g., "BRF Sjöstadsparterren"
    all_addresses: List[str] = field(default_factory=list)  # All addresses in property
    num_buildings: int = 1  # Number of buildings in property
    num_trapphus: int = 1  # Total entrances/staircases
    # Individual building details for multi-roof analysis
    # Each dict has: address, lat, lon, footprint_area_m2, height_m, num_floors, atemp_m2
    building_details: List[Dict[str, Any]] = field(default_factory=list)

    # Building characteristics (aggregated for whole property)
    construction_year: int = 0  # 0 = unknown
    building_type: str = "multi_family"
    facade_material: str = "concrete"
    atemp_m2: float = 0  # TOTAL for property
    num_floors: int = 4  # Max floors across buildings
    num_apartments: int = 0  # TOTAL for property

    # Building form and geometry
    building_form: str = "generic"  # lamellhus, skivhus, punkthus, etc.
    building_width_m: float = 12.0
    building_length_m: float = 30.0
    has_gallery_access: bool = False  # loftgångshus

    # Footprint geometry (from GeoJSON or OSM)
    footprint_area_m2: float = 0.0
    height_m: float = 0.0
    footprint_coords: List[tuple] = field(default_factory=list)  # [(x, y), ...]

    # Window-to-Wall Ratio (from AI detection or estimation)
    wwr: float = 0.20  # Default 20%, range 0.10-0.40 for Swedish buildings
    wwr_by_direction: Dict[str, float] = field(default_factory=dict)  # {N: 0.15, S: 0.25, ...}

    # Energy performance (from energy declaration if available)
    declared_energy_kwh_m2: float = 0
    energy_class: str = "Unknown"
    heating_system: str = "district"
    has_ftx: bool = False
    has_heat_pump: bool = False
    has_solar: bool = False

    # Detailed energy source kWh (from Sweden Buildings / Gripen)
    exhaust_air_hp_kwh: float = 0.0
    ground_source_hp_kwh: float = 0.0
    air_source_hp_kwh: float = 0.0
    district_heating_kwh: float = 0.0

    # Financials (for BRF)
    current_fund_sek: float = 0
    annual_fund_contribution_sek: float = 0
    annual_energy_cost_sek: float = 0

    # Peak demand
    peak_el_kw: float = 0
    peak_fv_kw: float = 0

    # Facade images (from Mapillary)
    facade_images: Dict[str, List[str]] = field(default_factory=dict)  # {N: [urls], S: [urls], ...}

    # Source confidence
    data_sources: List[str] = field(default_factory=list)
    confidence_score: float = 0.5


@dataclass
class PipelineResult:
    """Complete result from the address pipeline."""
    success: bool
    address: str
    building_data: BuildingData = None
    analysis_results: Any = None  # BuildingAnalysisResult
    maintenance_plan: Any = None  # MaintenancePlan
    effektvakt_result: Any = None  # PeakShavingResult
    report_path: Path = None
    error: str = ""
    processing_time_seconds: float = 0


class AddressGeocoder:
    """Geocode Swedish addresses using Nominatim (free, no account)."""

    # Retry config for geocoding API
    RETRY_CONFIG = RetryConfig(
        max_retries=3,
        base_delay=1.0,
        max_delay=10.0,
        retryable_exceptions=(
            ConnectionError,
            TimeoutError,
            OSError,
            requests.exceptions.RequestException,
        ),
    )

    def __init__(self, user_agent: str = "Raiden-BuildingAnalysis/1.0"):
        self.user_agent = user_agent
        self.base_url = "https://nominatim.openstreetmap.org/search"

    def geocode(self, address: str) -> GeocodingResult:
        """
        Geocode a Swedish address to coordinates.

        Args:
            address: Swedish street address (e.g., "Aktergatan 11, Stockholm")

        Returns:
            GeocodingResult with coordinates and metadata
        """
        # Validate address input
        try:
            parsed = validate_address(address)
            logger.debug(
                f"Geocoding address: {parsed.street_name} {parsed.street_number or ''}"
                f"{', ' + parsed.city if parsed.city else ''}"
            )
        except ValidationError as e:
            logger.warning(f"Address validation warning: {e}")
            # Continue with original address - validation is advisory

        @retry_with_backoff(config=self.RETRY_CONFIG)
        def _fetch_geocode():
            params = {
                "q": address,
                "format": "json",
                "countrycodes": "se",
                "addressdetails": 1,
                "limit": 1,
            }
            headers = {"User-Agent": self.user_agent}
            response = requests.get(
                self.base_url,
                params=params,
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            return response.json()

        try:
            results = _fetch_geocode()

            if not results:
                logger.warning(f"No geocoding results for: {address}")
                return GeocodingResult(
                    latitude=0,
                    longitude=0,
                    display_name="",
                    success=False,
                    error=f"No results found for address: {address}"
                )

            result = results[0]
            lat = float(result["lat"])
            lon = float(result["lon"])

            # Validate coordinates are in Sweden
            try:
                validate_coordinates(lat, lon, require_sweden=True)
            except ValidationError as e:
                logger.warning(f"Geocoded location may be incorrect: {e}")

            logger.info(
                f"Geocoded '{address}' → ({lat:.5f}, {lon:.5f})",
                extra={"address": address}
            )

            return GeocodingResult(
                latitude=lat,
                longitude=lon,
                display_name=result.get("display_name", address),
                address_components=result.get("address", {}),
                success=True,
            )

        except requests.RequestException as e:
            logger.error(
                f"Geocoding failed after retries: {e}",
                extra={"address": address, "error_type": type(e).__name__}
            )
            return GeocodingResult(
                latitude=0,
                longitude=0,
                display_name="",
                success=False,
                error=f"Geocoding service unavailable: {e}"
            )


class BuildingDataFetcher:
    """Fetch building data from various sources.

    Data source priority:
    1. Sweden Buildings GeoJSON (37,489 Stockholm buildings with 167 properties)
    2. OSM/Overture (footprint, height, floors)
    3. Mapillary + AI Analysis (WWR, facade material from street view)
    4. Nominatim (geocoding fallback)
    5. Era-based inference (LAST RESORT only)
    """

    def __init__(self, mapillary_token: Optional[str] = None):
        self.geocoder = AddressGeocoder()
        self.mapillary_token = mapillary_token or os.environ.get("MAPILLARY_TOKEN")
        self._facade_fetcher = None
        self._sweden_buildings_loader = None
        self._wwr_detector = None
        self._material_classifier = None
        self._brfdashboard_fetcher = None

    @property
    def wwr_detector(self):
        """Lazy load WWR detector."""
        if self._wwr_detector is None:
            try:
                from ..ai.wwr_detector import WWRDetector
                self._wwr_detector = WWRDetector(backend="opencv")
                logger.info("WWR detector loaded (opencv backend)")
            except Exception as e:
                logger.warning(f"WWR detector not available: {e}")
        return self._wwr_detector

    @property
    def material_classifier(self):
        """Lazy load material classifier."""
        if self._material_classifier is None:
            try:
                from ..ai.material_classifier import MaterialClassifier
                self._material_classifier = MaterialClassifier()
                logger.info("Material classifier loaded")
            except Exception as e:
                logger.warning(f"Material classifier not available: {e}")
        return self._material_classifier

    @property
    def sweden_buildings(self):
        """Lazy load Sweden Buildings GeoJSON."""
        if self._sweden_buildings_loader is None:
            try:
                from ..ingest.sweden_buildings import SwedenBuildingsLoader
                self._sweden_buildings_loader = SwedenBuildingsLoader()
            except Exception as e:
                logger.warning(f"Sweden Buildings loader not available: {e}")
        return self._sweden_buildings_loader

    @property
    def facade_fetcher(self):
        """Lazy load FacadeImageFetcher."""
        if self._facade_fetcher is None:
            try:
                from ..ingest.image_fetcher import FacadeImageFetcher
                self._facade_fetcher = FacadeImageFetcher(
                    mapillary_token=self.mapillary_token
                )
            except ImportError:
                logger.warning("FacadeImageFetcher not available")
        return self._facade_fetcher

    @property
    def brfdashboard(self):
        """Lazy load BRF Dashboard database fetcher."""
        if self._brfdashboard_fetcher is None:
            try:
                from ..db.brfdashboard import BRFDashboardFetcher
                self._brfdashboard_fetcher = BRFDashboardFetcher()
                if self._brfdashboard_fetcher.available:
                    logger.info("BRF Dashboard database connected")
                else:
                    logger.warning("BRF Dashboard database not available")
            except Exception as e:
                logger.warning(f"BRF Dashboard fetcher not available: {e}")
        return self._brfdashboard_fetcher

    def _check_brfdashboard_for_property(
        self,
        address: str,
        property_designation: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Check BRF Dashboard database for property data and existing Raiden analysis.

        The BRF Dashboard database contains:
        - v_building_complete: 54,932 buildings with all curated data
        - recommended_measures: Energy expert recommendations with savings estimates
        - implemented_measures: Already-done ECMs (for smart filtering)
        - energy_costs: Actual SEK costs from BRF annual reports
        - Existing Raiden analysis results (facade_material, building_form, wwr)

        Args:
            address: Street address
            property_designation: Fastighetsbeteckning if known

        Returns:
            Dict with database data if found, None otherwise
        """
        if not self.brfdashboard or not self.brfdashboard.available:
            return None

        try:
            # First try v_building_complete (richest data source with 54,932 buildings)
            building = self.brfdashboard.get_building_complete(address=address)

            if building:
                logger.info(f"✓ Found in BRF Dashboard v_building_complete: {building.address}")

                result = {
                    "source": "brfdashboard_v_building_complete",
                    "building_id": building.building_id,
                    "formular_id": building.formular_id,
                    "property_designation": building.property_designation,
                    "address": building.address,
                    "built_year": building.construction_year,
                    "renovation_year": building.renovation_year,
                    "total_area_sqm": building.atemp_m2,
                    "total_apartments": building.num_apartments,
                    "floors": building.num_floors,
                    "height_m": building.height_m,
                    "footprint_m2": building.footprint_area_m2,
                    "heating_type": building.heating_type_simple,
                    "heating_system_detailed": building.heating_system_detailed,
                    "latitude": building.latitude,
                    "longitude": building.longitude,
                    "energy_class": building.energy_class,
                    "energy_kwh_m2": building.declared_energy_kwh_m2,
                    "ventilation_type": building.ventilation_type,
                    "has_ftx": building.has_ftx,
                    "has_district_heating": building.has_district_heating,
                    "has_heat_pump": building.has_heat_pump,
                    "heat_pump_type": building.heat_pump_type,
                    "has_solar_pv": building.has_solar_pv,
                    "has_solar_thermal": building.has_solar_thermal,
                }

                # Add recommended_measures (VALUABLE - expert recommendations!)
                if building.recommended_measures:
                    result["recommended_measures"] = [
                        {
                            "category": m.category,
                            "description": m.description,
                            "savings_kwh_year": m.savings_kwh_year,
                            "cost_per_saved_kwh": m.cost_per_saved_kwh,
                        }
                        for m in building.recommended_measures
                    ]
                    result["total_recommended_savings_kwh"] = building.total_recommended_savings_kwh()
                    logger.info(f"  → Found {len(building.recommended_measures)} expert recommendations")

                # Add implemented_measures (for smart ECM filtering)
                if building.implemented_measures:
                    result["implemented_measures"] = [
                        {
                            "measure_type": m.measure_type,
                            "description": m.description,
                            "implementation_year": m.implementation_year,
                        }
                        for m in building.implemented_measures
                    ]

                # Add energy_costs (VALUABLE - real costs from annual reports!)
                if building.energy_costs:
                    result["energy_costs"] = {
                        "heating_cost_kr": building.energy_costs.heating_cost_kr,
                        "electricity_cost_kr": building.energy_costs.electricity_cost_kr,
                        "water_cost_kr": building.energy_costs.water_cost_kr,
                        "total_energy_cost_kr": building.energy_costs.total_energy_cost_kr(),
                        "fiscal_year": building.energy_costs.fiscal_year,
                    }
                    if building.atemp_m2 and building.atemp_m2 > 0:
                        result["energy_costs"]["cost_per_sqm"] = building.energy_costs.cost_per_sqm(building.atemp_m2)
                    logger.info(f"  → Found energy costs: {building.energy_costs.heating_cost_kr:,.0f} SEK heating")

                # Existing Raiden analysis results (skip re-analysis if available)
                if building.raiden_facade_material:
                    result["raiden_facade_material"] = building.raiden_facade_material
                if building.raiden_building_form:
                    result["raiden_building_form"] = building.raiden_building_form
                if building.raiden_wwr:
                    result["raiden_wwr"] = building.raiden_wwr
                if building.raiden_confidence:
                    result["raiden_confidence"] = building.raiden_confidence

                # Store the full BuildingComplete object for ECM prioritization
                result["_building_complete"] = building

                return result

            # Fallback: try brf_property (aggregated BRF data)
            brf_data = self.brfdashboard.get_brf_property(
                address=address,
                property_designation=property_designation
            )

            if brf_data:
                logger.info(f"✓ Found in BRF Dashboard brf_property: {brf_data.property_designation}")

                result = {
                    "source": "brfdashboard",
                    "property_designation": brf_data.property_designation,
                    "address": brf_data.address,
                    "built_year": brf_data.built_year,
                    "total_area_sqm": brf_data.total_area_sqm,
                    "total_apartments": brf_data.total_apartments,
                    "floors": brf_data.floors,
                    "buildings_count": brf_data.buildings_count,
                    "heating_type": brf_data.heating_type,
                    "latitude": brf_data.latitude,
                    "longitude": brf_data.longitude,
                }

                # Gripen energy data
                if brf_data.gripen_energy_class:
                    result["energy_class"] = brf_data.gripen_energy_class
                    result["energy_kwh_m2"] = brf_data.gripen_specific_energy_use
                elif brf_data.energy_class:
                    result["energy_class"] = brf_data.energy_class
                    result["energy_kwh_m2"] = brf_data.energy_kwh_per_sqm

                # Existing Raiden analysis results (skip re-analysis if available)
                if brf_data.raiden_facade_material:
                    result["raiden_facade_material"] = brf_data.raiden_facade_material
                if brf_data.raiden_building_form:
                    result["raiden_building_form"] = brf_data.raiden_building_form
                if brf_data.raiden_wwr:
                    result["raiden_wwr"] = brf_data.raiden_wwr
                if brf_data.raiden_confidence:
                    result["raiden_confidence"] = brf_data.raiden_confidence

                return result

            # Last fallback: try sweden_buildings directly
            sweden_building = self.brfdashboard.get_sweden_building_by_address(address)
            if sweden_building:
                logger.info(f"✓ Found in BRF Dashboard sweden_buildings: {sweden_building.address}")
                return {
                    "source": "brfdashboard_sweden",
                    "property_designation": sweden_building.property_designation,
                    "address": sweden_building.address,
                    "built_year": sweden_building.construction_year,
                    "total_area_sqm": sweden_building.atemp_m2,
                    "num_apartments": sweden_building.num_apartments,
                    "floors": sweden_building.num_floors,
                    "energy_class": sweden_building.energy_class,
                    "energy_kwh_m2": sweden_building.energy_performance_kwh_m2,
                    "ventilation_type": sweden_building.ventilation_type,
                    "has_ftx": sweden_building.has_ftx,
                }

        except Exception as e:
            logger.warning(f"BRF Dashboard lookup failed: {e}")

        return None

    def _find_all_buildings_in_property(
        self,
        property_designation: str,
    ) -> List[Any]:
        """
        Find all buildings that belong to the same property (fastighetsbeteckning).

        In Sweden, a BRF owns a property (fastighet) which can contain multiple
        buildings/addresses. Energy analysis should be done at property level.

        Args:
            property_designation: The fastighetsbeteckning (e.g., "Böljan 2")

        Returns:
            List of SwedishBuilding objects in this property
        """
        if not property_designation or not self.sweden_buildings:
            return []

        # Search through all buildings for matching property
        matching = []
        try:
            # Access the internal buildings list
            all_buildings = self.sweden_buildings._buildings

            for building in all_buildings:
                props = building.raw_properties
                fastbet = props.get('IdFastBet', '')
                if fastbet == property_designation:
                    matching.append(building)

            logger.info(f"Found {len(matching)} buildings in property '{property_designation}'")

        except Exception as e:
            logger.warning(f"Error searching for property buildings: {e}")

        return matching

    def _aggregate_property_data(
        self,
        buildings: List[Any],
        primary_building: Any,
    ) -> Dict[str, Any]:
        """
        Aggregate data from multiple buildings in the same property.

        Args:
            buildings: List of SwedishBuilding objects in the property
            primary_building: The building that matches the input address

        Returns:
            Dict with aggregated property-level data
        """
        if not buildings:
            return {}

        # Get property info from primary building
        props = primary_building.raw_properties
        property_designation = props.get('IdFastBet', '')
        property_owner = props.get('42P_ByggnadsAgare', '')

        # De-duplicate buildings by address (database may have duplicate records)
        # Use address as key since same address = same physical entrance
        unique_buildings = {}
        for b in buildings:
            addr = b.address
            if addr not in unique_buildings:
                unique_buildings[addr] = b
            # If duplicate, keep the one with more data (higher atemp)
            elif (b.atemp_m2 or 0) > (unique_buildings[addr].atemp_m2 or 0):
                unique_buildings[addr] = b

        buildings = list(unique_buildings.values())
        logger.info(f"De-duplicated to {len(buildings)} unique addresses")

        # Aggregate across all unique buildings
        all_addresses = sorted(unique_buildings.keys())
        total_apartments = sum(b.num_apartments or 0 for b in buildings)
        total_atemp = sum(b.atemp_m2 or 0 for b in buildings)
        total_trapphus = sum(b.raw_properties.get('EgenAntalTrapphus', 0) or 0 for b in buildings)
        max_floors = max((b.num_floors or 0 for b in buildings), default=0)
        total_footprint = sum(b.footprint_area_m2 or 0 for b in buildings)

        # Aggregate energy data
        total_district_heating = sum(b.district_heating_kwh or 0 for b in buildings)
        total_exhaust_hp = sum(b.exhaust_air_hp_kwh or 0 for b in buildings)
        total_ground_hp = sum(b.ground_source_hp_kwh or 0 for b in buildings)

        # Use most common construction year (should be same for property)
        years = [b.construction_year for b in buildings if b.construction_year]
        construction_year = max(set(years), key=years.count) if years else 2000

        # Use most common energy class
        classes = [b.energy_class for b in buildings if b.energy_class]
        energy_class = max(set(classes), key=classes.count) if classes else "Unknown"

        logger.info(f"Property '{property_designation}' aggregated: {len(all_addresses)} addresses, "
                    f"{total_apartments:.0f} apartments, {total_atemp:.0f} m²")

        # Preserve individual building data for multi-roof solar analysis
        # Each building has its own roof that should be analyzed
        building_details = []
        for b in buildings:
            # Get centroid coordinates in WGS84 (lat, lon)
            # SwedishBuilding stores SWEREF99 TM coordinates, need to convert
            lat, lon = None, None
            try:
                centroid = b.get_centroid_wgs84()
                if centroid and centroid[0] != 0 and centroid[1] != 0:
                    lat, lon = centroid[0], centroid[1]
            except Exception:
                pass

            building_details.append({
                "address": b.address,
                "lat": lat,
                "lon": lon,
                "footprint_area_m2": b.footprint_area_m2 or 0,
                "height_m": b.height_m,
                "num_floors": b.num_floors,
                "atemp_m2": b.atemp_m2 or 0,
            })

        return {
            "property_designation": property_designation,
            "property_owner": property_owner,
            "all_addresses": all_addresses,
            "num_buildings": len(buildings),
            "num_trapphus": int(total_trapphus),
            "num_apartments": int(total_apartments),
            "atemp_m2": total_atemp,
            "num_floors": int(max_floors),
            "footprint_area_m2": total_footprint,
            "construction_year": construction_year,
            "energy_class": energy_class,
            "district_heating_kwh": total_district_heating,
            "exhaust_air_hp_kwh": total_exhaust_hp,
            "ground_source_hp_kwh": total_ground_hp,
            # NEW: Individual building details for multi-roof analysis
            "building_details": building_details,
        }

    def _analyze_facade_images(
        self,
        image_urls: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """
        Analyze facade images using AI to detect WWR and material.

        Args:
            image_urls: Dict mapping direction (N/S/E/W) to list of image URLs

        Returns:
            Dict with detected 'wwr', 'facade_material', 'confidence'
        """
        import requests
        from PIL import Image
        from io import BytesIO

        results = {
            "wwr_by_direction": {},
            "wwr_average": 0.0,
            "facade_material": None,
            "material_confidence": 0.0,
            "analyzed_images": 0,
        }

        all_wwr = []
        material_votes = {}

        for direction, urls in image_urls.items():
            if not urls:
                continue

            for url in urls[:2]:  # Analyze max 2 images per direction
                try:
                    # Download image
                    response = requests.get(url, timeout=10)
                    if not response.ok:
                        continue

                    image = Image.open(BytesIO(response.content))
                    results["analyzed_images"] += 1

                    # Detect WWR
                    if self.wwr_detector:
                        try:
                            wwr_result = self.wwr_detector.calculate_wwr(image)
                            if wwr_result and wwr_result.average > 0:
                                all_wwr.append(wwr_result.average)
                                if direction not in results["wwr_by_direction"]:
                                    results["wwr_by_direction"][direction] = []
                                results["wwr_by_direction"][direction].append(wwr_result.average)
                                logger.info(f"  WWR from {direction}: {wwr_result.average:.1%}")
                        except Exception as e:
                            logger.debug(f"WWR detection failed: {e}")

                    # Detect material
                    if self.material_classifier:
                        try:
                            mat_result = self.material_classifier.classify(image)
                            if mat_result and mat_result.material:
                                mat_name = mat_result.material.value if hasattr(mat_result.material, 'value') else str(mat_result.material)
                                material_votes[mat_name] = material_votes.get(mat_name, 0) + mat_result.confidence
                                logger.info(f"  Material from {direction}: {mat_name} ({mat_result.confidence:.0%})")
                        except Exception as e:
                            logger.debug(f"Material detection failed: {e}")

                except Exception as e:
                    logger.debug(f"Failed to analyze image {url}: {e}")

        # Aggregate results
        if all_wwr:
            results["wwr_average"] = sum(all_wwr) / len(all_wwr)

        if material_votes:
            # Pick material with highest weighted votes
            best_material = max(material_votes, key=material_votes.get)
            results["facade_material"] = best_material
            results["material_confidence"] = material_votes[best_material] / sum(material_votes.values())

        return results

    def _find_in_sweden_buildings(self, address: str) -> Optional[Any]:
        """
        Search for building in Sweden Buildings GeoJSON.

        Tries multiple matching strategies:
        1. Exact address match
        2. Street name + number match
        3. Fuzzy matching on street name

        Args:
            address: Swedish address string

        Returns:
            SwedishBuilding if found, None otherwise
        """
        if not self.sweden_buildings:
            return None

        try:
            # Strategy 1: Direct address search
            matches = self.sweden_buildings.find_by_address(address)
            if matches:
                # Return the best match (first result)
                logger.info(f"  Found {len(matches)} matches for '{address}'")
                return matches[0]

            # Strategy 2: Try extracting street name and number
            import re
            # Pattern: "Street Name 123" or "Street Name 123A"
            match = re.match(r'^([^\d]+)\s*(\d+\w*)', address)
            if match:
                street_name = match.group(1).strip()
                street_number = match.group(2).strip()

                # Search by street name
                matches = self.sweden_buildings.find_by_address(street_name)
                if matches:
                    # Try to find exact number match
                    for building in matches:
                        if street_number in building.address:
                            return building
                    # Return first match on street if no exact number
                    if matches:
                        logger.info(f"  Found {len(matches)} on '{street_name}', returning first")
                        return matches[0]

            # Strategy 3: Try city/district if address has comma
            if "," in address:
                parts = address.split(",")
                street_part = parts[0].strip()
                matches = self.sweden_buildings.find_by_address(street_part)
                if matches:
                    return matches[0]

            return None

        except Exception as e:
            logger.warning(f"Sweden Buildings search failed: {e}")
            return None

    def fetch(
        self,
        address: str,
        known_data: Optional[Dict] = None,
        fetch_images: bool = True,
    ) -> BuildingData:
        """
        Fetch building data from address.

        Data source priority:
        1. Sweden Buildings GeoJSON (37,489 buildings with energy data) - TRY FIRST!
        2. OSM/Overture (footprint, geometry)
        3. Mapillary (facade images)
        4. Nominatim (geocoding)

        Args:
            address: Swedish street address
            known_data: Optional dict with any known building data
            fetch_images: If True, attempt to fetch facade images from Mapillary

        Returns:
            BuildingData with assembled information

        Raises:
            ValidationError: If address is fundamentally invalid
        """
        # Validate address
        try:
            address_parts = validate_address(address)
            logger.info(
                f"Fetching building data for: {address}",
                extra={"address": address}
            )
        except ValidationError as e:
            logger.error(f"Invalid address: {e}", extra={"address": address})
            raise

        sources = []

        # ============================================================
        # STEP 1: Try Sweden Buildings GeoJSON FIRST (richest source!)
        # ============================================================
        sweden_building = self._find_in_sweden_buildings(address)

        if sweden_building:
            logger.info(f"✓ Found in Sweden Buildings GeoJSON: {sweden_building.address}")
            sources.append("sweden_buildings_geojson")

            # Get WGS84 coordinates from SWEREF99
            lat, lon = sweden_building.get_centroid_wgs84()

            # Determine heating system
            heating = sweden_building.get_primary_heating()
            # Check if ANY heat pump is present (buildings can have district heating AND heat pump)
            has_hp = (
                heating in ["ground_source_hp", "exhaust_air_hp", "air_water_hp", "air_air_hp"] or
                sweden_building.exhaust_air_hp_kwh > 0 or
                sweden_building.ground_source_hp_kwh > 0 or
                sweden_building.air_water_hp_kwh > 0 or
                sweden_building.air_air_hp_kwh > 0
            )

            # Create BuildingData from rich GeoJSON data
            # Calculate height from floors if not available
            floors = sweden_building.num_floors or 4
            height = sweden_building.height_m or (floors * 3.0)  # Assume 3m per floor
            footprint = sweden_building.footprint_area_m2 or 0.0

            # Calculate width/length from footprint (assume square-ish)
            import math
            if footprint > 0:
                # Estimate width/length assuming 1:2 aspect ratio (typical lamellhus)
                width = math.sqrt(footprint / 2)
                length = width * 2
            else:
                width = 12.0
                length = 30.0

            data = BuildingData(
                address=sweden_building.address,
                latitude=lat,
                longitude=lon,
                construction_year=sweden_building.construction_year or 2000,
                building_type="multi_family" if "Flerbostadshus" in sweden_building.building_category else "single_family",
                atemp_m2=sweden_building.atemp_m2,
                num_floors=int(floors),
                num_apartments=int(sweden_building.num_apartments or 0),
                # Geometry from GeoJSON
                footprint_area_m2=footprint,
                height_m=height,
                footprint_coords=_convert_sweref_footprint_to_wgs84(sweden_building.footprint_coords) if sweden_building.footprint_coords else [],
                building_width_m=width,
                building_length_m=length,
                # Energy data
                declared_energy_kwh_m2=sweden_building.energy_performance_kwh_m2 or 0,
                energy_class=sweden_building.energy_class or "Unknown",
                heating_system="district" if sweden_building.district_heating_kwh > 0 else ("heat_pump" if has_hp else "electric"),
                has_ftx=sweden_building.ventilation_type == "FTX",
                has_heat_pump=has_hp,
                has_solar=sweden_building.has_solar_pv or sweden_building.has_solar_thermal,
                # Detailed energy source kWh (crucial for existing measures detection!)
                exhaust_air_hp_kwh=sweden_building.exhaust_air_hp_kwh or 0.0,
                ground_source_hp_kwh=sweden_building.ground_source_hp_kwh or 0.0,
                # Air source = air-to-water + air-to-air heat pumps
                air_source_hp_kwh=(sweden_building.air_water_hp_kwh or 0.0) + (sweden_building.air_air_hp_kwh or 0.0),
                district_heating_kwh=sweden_building.district_heating_kwh or 0.0,
                data_sources=sources,
            )

            # ============================================================
            # PROPERTY AGGREGATION: Find ALL buildings in same property
            # A property (fastighetsbeteckning) can contain multiple addresses
            # ============================================================
            property_designation = sweden_building.raw_properties.get('IdFastBet', '')
            if property_designation:
                property_buildings = self._find_all_buildings_in_property(property_designation)

                if len(property_buildings) > 1:
                    # Multiple buildings in property - aggregate data
                    aggregated = self._aggregate_property_data(property_buildings, sweden_building)

                    # Update BuildingData with property-level aggregated values
                    data.property_designation = aggregated.get("property_designation", "")
                    data.property_owner = aggregated.get("property_owner", "")
                    data.all_addresses = aggregated.get("all_addresses", [data.address])
                    data.num_buildings = aggregated.get("num_buildings", 1)
                    data.num_trapphus = aggregated.get("num_trapphus", 1)

                    # Use aggregated totals for the WHOLE property
                    data.num_apartments = aggregated.get("num_apartments", data.num_apartments)
                    data.atemp_m2 = aggregated.get("atemp_m2", data.atemp_m2)
                    data.num_floors = aggregated.get("num_floors", data.num_floors)
                    data.footprint_area_m2 = aggregated.get("footprint_area_m2", data.footprint_area_m2)
                    data.construction_year = aggregated.get("construction_year", data.construction_year)
                    data.energy_class = aggregated.get("energy_class", data.energy_class)

                    # Aggregate energy source kWh
                    data.district_heating_kwh = aggregated.get("district_heating_kwh", data.district_heating_kwh)
                    data.exhaust_air_hp_kwh = aggregated.get("exhaust_air_hp_kwh", data.exhaust_air_hp_kwh)
                    data.ground_source_hp_kwh = aggregated.get("ground_source_hp_kwh", data.ground_source_hp_kwh)

                    # Individual building details for multi-roof solar/facade analysis
                    data.building_details = aggregated.get("building_details", [])

                    sources.append("property_aggregation")
                    logger.info(f"✓ Property aggregation: {data.num_buildings} buildings, "
                                f"{data.num_apartments} apartments, {data.atemp_m2:.0f} m²")
                else:
                    # Single building in property
                    data.property_designation = property_designation
                    data.property_owner = sweden_building.raw_properties.get('42P_ByggnadsAgare', '')
                    data.all_addresses = [data.address]
                    data.num_trapphus = int(sweden_building.raw_properties.get('EgenAntalTrapphus', 1) or 1)

            # Try to determine facade material from building type/era
            if sweden_building.construction_year:
                if sweden_building.construction_year < 1945:
                    data.facade_material = "brick"
                elif sweden_building.construction_year < 1975:
                    data.facade_material = "concrete"
                else:
                    data.facade_material = "plaster"

            # ============================================================
            # CHECK BRF DASHBOARD DATABASE for existing Raiden analysis
            # ============================================================
            brfdashboard_data = self._check_brfdashboard_for_property(
                address=address,
                property_designation=data.property_designation
            )

            has_existing_raiden_analysis = False
            if brfdashboard_data:
                sources.append("brfdashboard")

                # Use existing Raiden analysis if available (skip re-analysis)
                if brfdashboard_data.get("raiden_facade_material"):
                    data.facade_material = brfdashboard_data["raiden_facade_material"]
                    has_existing_raiden_analysis = True
                    logger.info(f"  ✓ Using cached facade material: {data.facade_material}")
                if brfdashboard_data.get("raiden_building_form"):
                    data.building_form = brfdashboard_data["raiden_building_form"]
                if brfdashboard_data.get("raiden_wwr"):
                    data.wwr = brfdashboard_data["raiden_wwr"]
                    has_existing_raiden_analysis = True
                    logger.info(f"  ✓ Using cached WWR: {data.wwr:.2f}")

                # Use Gripen energy class if available and better confidence
                if brfdashboard_data.get("energy_class") and brfdashboard_data.get("energy_kwh_m2"):
                    if brfdashboard_data.get("energy_kwh_m2", 0) > 0:
                        data.energy_class = brfdashboard_data["energy_class"]
                        data.declared_energy_kwh_m2 = brfdashboard_data["energy_kwh_m2"]
                        logger.info(f"  ✓ Using Gripen energy: class {data.energy_class}, {data.declared_energy_kwh_m2:.0f} kWh/m²")

            # Override with known data if provided
            if known_data:
                for key, value in known_data.items():
                    if hasattr(data, key) and value is not None:
                        setattr(data, key, value)
                sources.append("user_provided")

            # Optionally fetch facade images AND run AI analysis for material detection
            # Skip if we have existing Raiden analysis with high confidence
            if fetch_images and self.facade_fetcher and not has_existing_raiden_analysis:
                try:
                    images = self._fetch_facade_images(lat, lon, None)
                    if images:
                        data.facade_images = images
                        sources.append("mapillary")

                        # Run AI analysis on facade images to get accurate material detection
                        # This is important because era-based guessing (brick/concrete/plaster) is often wrong
                        ai_results = self._analyze_facade_images(images)
                        if ai_results.get("facade_material"):
                            data.facade_material = ai_results["facade_material"]
                            logger.info(f"  ✓ AI detected material: {data.facade_material} ({ai_results.get('material_confidence', 0):.0%})")
                            sources.append("ai_material_detection")
                        if ai_results.get("wwr_average") and ai_results["wwr_average"] > 0:
                            data.wwr = ai_results["wwr_average"]
                            sources.append("ai_wwr_detection")
                        if ai_results.get("wwr_by_direction"):
                            for direction, wwr_list in ai_results["wwr_by_direction"].items():
                                if wwr_list:
                                    data.wwr_by_direction[direction] = sum(wwr_list) / len(wwr_list)
                except Exception as e:
                    logger.warning(f"Mapillary/AI analysis failed: {e}")

            # Estimate any missing values (like peak_el_kw, peak_fv_kw)
            data = self._estimate_missing_values(data)

            data.data_sources = sources
            data.confidence_score = min(len(sources) / 3 + 0.5, 1.0)  # Higher base confidence
            return data

        # ============================================================
        # STEP 2: Building not in GeoJSON - Fall back to OSM/Nominatim
        # ============================================================
        logger.info(f"Building not in Sweden Buildings GeoJSON, falling back to OSM...")

        # Start with geocoding
        geo = self.geocoder.geocode(address)

        if not geo.success:
            logger.warning(f"Geocoding failed: {geo.error}")
            # Use placeholder coordinates for Stockholm
            geo.latitude = 59.3293
            geo.longitude = 18.0686
        else:
            sources.append("nominatim")

        # Initialize with defaults
        data = BuildingData(
            address=address,
            latitude=geo.latitude,
            longitude=geo.longitude,
            construction_year=2000,  # Default to modern
            data_sources=sources,
        )

        # Override with any known data
        if known_data:
            for key, value in known_data.items():
                if hasattr(data, key) and value is not None:
                    setattr(data, key, value)
            sources.append("user_provided")

        # Try to fetch from OSM (building footprint and geometry)
        osm_geometry = None
        try:
            osm_data, osm_geometry = self._fetch_osm_building_with_geometry(
                geo.latitude, geo.longitude
            )
            if osm_data:
                if "building:levels" in osm_data:
                    data.num_floors = int(osm_data["building:levels"])
                sources.append("osm")
        except Exception as e:
            logger.warning(f"OSM fetch failed: {e}")

        # Detect building form from geometry
        if osm_geometry:
            form_result = self._detect_building_form(
                osm_geometry,
                data.num_floors,
                data.construction_year
            )
            data.building_form = form_result["form"]
            data.building_width_m = form_result["width_m"]
            data.building_length_m = form_result["length_m"]
            data.has_gallery_access = form_result["has_gallery"]
            sources.append("form_detection")

        # ============================================================
        # STEP 3: Fetch Mapillary images + AI analysis for WWR & material
        # ============================================================
        if fetch_images and self.facade_fetcher:
            try:
                images = self._fetch_facade_images(
                    geo.latitude, geo.longitude, osm_geometry
                )
                if images:
                    data.facade_images = images
                    sources.append("mapillary")

                    # Run AI analysis on facade images
                    logger.info("Running AI analysis on Mapillary images...")
                    ai_results = self._analyze_facade_images(images)

                    if ai_results["analyzed_images"] > 0:
                        sources.append("ai_vision")

                        # Use detected WWR (average)
                        if ai_results["wwr_average"] > 0:
                            data.wwr = ai_results["wwr_average"]
                            logger.info(f"  ✓ Detected WWR: {data.wwr:.1%}")

                        # Store per-direction WWR
                        if ai_results["wwr_by_direction"]:
                            for direction, wwr_list in ai_results["wwr_by_direction"].items():
                                data.wwr_by_direction[direction] = sum(wwr_list) / len(wwr_list)

                        # Use detected facade material
                        if ai_results["facade_material"]:
                            data.facade_material = ai_results["facade_material"]
                            logger.info(f"  ✓ Detected material: {data.facade_material} ({ai_results['material_confidence']:.0%})")
                            sources.append("ai_material_detection")
            except Exception as e:
                logger.warning(f"Mapillary fetch failed: {e}")

        # Estimate missing values based on construction year
        data = self._estimate_missing_values(data)

        data.data_sources = sources
        data.confidence_score = min(len(sources) / 5, 1.0)

        return data

    # Retry config for OSM/Overpass API
    OSM_RETRY_CONFIG = RetryConfig(
        max_retries=2,
        base_delay=2.0,
        max_delay=15.0,
        retryable_exceptions=(
            ConnectionError,
            TimeoutError,
            OSError,
            requests.exceptions.RequestException,
        ),
        retryable_status_codes=(429, 500, 502, 503, 504),
    )

    def _fetch_osm_building_with_geometry(
        self, lat: float, lon: float
    ) -> Tuple[Optional[Dict], Optional[List]]:
        """Fetch building data and geometry from OpenStreetMap with retry."""

        @retry_with_backoff(config=self.OSM_RETRY_CONFIG)
        def _query_overpass():
            query = f"""
            [out:json][timeout:10];
            (
              way["building"](around:50,{lat},{lon});
              relation["building"](around:50,{lat},{lon});
            );
            out body geom;
            """
            response = requests.post(
                "https://overpass-api.de/api/interpreter",
                data=query,
                timeout=15
            )
            response.raise_for_status()
            return response.json()

        try:
            data = _query_overpass()

            if data.get("elements"):
                element = data["elements"][0]
                tags = element.get("tags", {})

                # Extract geometry
                geometry = None
                if "geometry" in element:
                    geometry = [
                        (node["lon"], node["lat"])
                        for node in element["geometry"]
                    ]
                elif "bounds" in element:
                    bounds = element["bounds"]
                    # Create rectangle from bounds
                    geometry = [
                        (bounds["minlon"], bounds["minlat"]),
                        (bounds["maxlon"], bounds["minlat"]),
                        (bounds["maxlon"], bounds["maxlat"]),
                        (bounds["minlon"], bounds["maxlat"]),
                    ]

                logger.debug(f"OSM building found at ({lat}, {lon})")
                return tags, geometry

            logger.debug(f"No OSM building found at ({lat}, {lon})")
            return None, None

        except requests.RequestException as e:
            # Graceful degradation - continue without OSM data
            logger.warning(
                f"OSM service unavailable, continuing without footprint data: {e}",
                extra={"error_type": type(e).__name__}
            )
            return None, None
        except Exception as e:
            logger.debug(f"OSM query failed: {e}")
            return None, None

    def _detect_building_form(
        self,
        geometry: List[Tuple[float, float]],
        num_floors: int,
        construction_year: int
    ) -> Dict:
        """
        Detect building form from footprint geometry.

        Args:
            geometry: List of (lon, lat) coordinates
            num_floors: Number of floors
            construction_year: Year built

        Returns:
            Dict with form, width_m, length_m, has_gallery
        """
        import math

        # Calculate bounding box and dimensions
        lons = [p[0] for p in geometry]
        lats = [p[1] for p in geometry]

        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)

        # Convert to meters (approximate)
        center_lat = (min_lat + max_lat) / 2
        lon_to_m = 111000 * math.cos(math.radians(center_lat))
        lat_to_m = 111000

        width_m = (max_lon - min_lon) * lon_to_m
        length_m = (max_lat - min_lat) * lat_to_m

        # Ensure width < length
        if width_m > length_m:
            width_m, length_m = length_m, width_m

        # Use building_forms detection
        try:
            from ..baseline.building_forms import detect_building_form, BuildingForm
            form = detect_building_form(
                stories=num_floors,
                width_m=width_m,
                length_m=length_m,
                construction_year=construction_year,
                has_gallery=False  # Would need image analysis to detect
            )
            form_str = form.value
        except ImportError:
            # Fallback detection
            aspect_ratio = length_m / width_m if width_m > 0 else 1

            if num_floors >= 8 and aspect_ratio > 3:
                form_str = "skivhus"
            elif num_floors >= 8 and aspect_ratio < 1.3:
                form_str = "punkthus"
            elif 3 <= num_floors <= 5 and aspect_ratio > 3:
                form_str = "lamellhus"
            else:
                form_str = "generic"

        return {
            "form": form_str,
            "width_m": width_m,
            "length_m": length_m,
            "has_gallery": False,  # Would need image analysis
        }

    def _fetch_facade_images(
        self,
        lat: float,
        lon: float,
        geometry: Optional[List] = None
    ) -> Dict[str, List[str]]:
        """
        Fetch facade images from Mapillary and other sources.

        Args:
            lat: Building latitude
            lon: Building longitude
            geometry: Optional building footprint

        Returns:
            Dict mapping orientation (N/S/E/W) to list of image URLs
        """
        if not self.facade_fetcher:
            return {}

        # Create building coords from geometry or use point
        if geometry:
            building_coords = geometry
        else:
            # Create small box around point
            offset = 0.0001  # ~10m
            building_coords = [
                (lon - offset, lat - offset),
                (lon + offset, lat - offset),
                (lon + offset, lat + offset),
                (lon - offset, lat + offset),
            ]

        # Fetch images
        images_by_direction = self.facade_fetcher.fetch_for_building(
            building_coords=building_coords,
            building_id=f"{lat:.5f}_{lon:.5f}",
            search_radius_m=100,
            orientations=["N", "S", "E", "W"],
        )

        # Extract URLs
        result = {}
        for direction, images in images_by_direction.items():
            if direction == "unclassified":
                continue
            urls = [img.url for img in images if img.url][:3]  # Max 3 per direction
            if urls:
                result[direction] = urls

        return result

    def _fetch_osm_building(self, lat: float, lon: float) -> Optional[Dict]:
        """Fetch building data from OpenStreetMap (legacy method)."""
        try:
            # Overpass API query for building at location
            query = f"""
            [out:json][timeout:10];
            (
              way["building"](around:50,{lat},{lon});
              relation["building"](around:50,{lat},{lon});
            );
            out body;
            """

            response = requests.post(
                "https://overpass-api.de/api/interpreter",
                data=query,
                timeout=15
            )

            if response.ok:
                data = response.json()
                if data.get("elements"):
                    return data["elements"][0].get("tags", {})
        except Exception as e:
            logger.debug(f"OSM query failed: {e}")

        return None

    def _estimate_missing_values(self, data: BuildingData) -> BuildingData:
        """
        Estimate missing values as LAST RESORT (when AI detection didn't work).

        This is only used when:
        - Building not in Sweden Buildings GeoJSON
        - AI detection from Mapillary images failed or wasn't available
        """
        year = data.construction_year

        # Estimate Atemp if not provided
        if data.atemp_m2 == 0:
            if data.num_apartments > 0:
                # Typical Swedish apartment: ~70-80 m²
                data.atemp_m2 = data.num_apartments * 75
            elif data.num_floors > 0:
                # Assume typical floor plate
                data.atemp_m2 = data.num_floors * 400
            else:
                data.atemp_m2 = 2000  # Default

        # ONLY estimate facade material if AI detection didn't work
        # (i.e., material is still the default "concrete" AND no AI source)
        ai_detected_material = "ai_material_detection" in data.data_sources
        if data.facade_material == "concrete" and not ai_detected_material:
            logger.info("  Using era-based material inference (AI detection unavailable)")
            if year < 1945:
                data.facade_material = "brick"
            elif year < 1975:
                data.facade_material = "concrete"
            else:
                data.facade_material = "plaster"

        # Estimate energy performance from era
        if data.declared_energy_kwh_m2 == 0:
            if year < 1960:
                data.declared_energy_kwh_m2 = 150
            elif year < 1975:
                data.declared_energy_kwh_m2 = 130
            elif year < 1990:
                data.declared_energy_kwh_m2 = 100
            elif year < 2010:
                data.declared_energy_kwh_m2 = 80
            else:
                data.declared_energy_kwh_m2 = 50

        # Estimate peak demand
        if data.peak_el_kw == 0:
            # Typical: 5-10 W/m²
            data.peak_el_kw = data.atemp_m2 * 0.008

        if data.peak_fv_kw == 0:
            # District heating peak: 30-50 W/m² in winter
            data.peak_fv_kw = data.atemp_m2 * 0.040

        # Estimate financials for BRF
        if data.num_apartments > 0 and data.annual_energy_cost_sek == 0:
            # Typical energy cost: 50-80 SEK/m²/year
            data.annual_energy_cost_sek = data.atemp_m2 * 60

        return data


class AddressPipeline:
    """
    The unified address-to-report pipeline.

    Usage:
        pipeline = AddressPipeline()
        result = pipeline.analyze("Aktergatan 11, Stockholm")
        print(f"Report saved to: {result.report_path}")
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("./output")
        self.data_fetcher = BuildingDataFetcher()

    def analyze(
        self,
        address: str,
        known_data: Optional[Dict] = None,
        skip_simulation: bool = False,
        generate_report: bool = True,
    ) -> PipelineResult:
        """
        Run complete analysis pipeline from address.

        Args:
            address: Swedish street address
            known_data: Optional dict with known building data
            skip_simulation: If True, skip EnergyPlus simulations
            generate_report: If True, generate HTML report

        Returns:
            PipelineResult with all analysis results
        """
        import time
        start_time = time.time()

        logger.info(
            f"Starting analysis pipeline",
            extra={"address": address}
        )

        # Validate address early
        try:
            validate_address(address)
        except ValidationError as e:
            logger.error(
                f"Invalid address provided: {str(e)}",
                extra={"address": address, "error_type": "validation"}
            )
            suggestions = getattr(e, 'suggestions', [])
            return PipelineResult(
                success=False,
                address=address,
                error=f"Invalid address: {str(e)}. {' '.join(suggestions) if suggestions else ''}",
                processing_time_seconds=time.time() - start_time,
            )

        try:
            # Step 1: Fetch building data
            logger.info("Step 1/5: Fetching building data...")
            building_data = self.data_fetcher.fetch(address, known_data)
            logger.info(
                f"  Found: {building_data.atemp_m2:.0f} m², year {building_data.construction_year}",
                extra={
                    "building_id": address,
                    "atemp_m2": building_data.atemp_m2,
                    "construction_year": building_data.construction_year,
                }
            )

            # Step 2: Run energy analysis (optional)
            analysis_results = None
            if not skip_simulation:
                logger.info("Step 2/5: Running energy analysis...")
                try:
                    analysis_results = self._run_energy_analysis(building_data)
                except Exception as e:
                    # Graceful degradation - continue without simulation
                    logger.warning(
                        f"Energy analysis failed, continuing without: {e}",
                        extra={"error_type": type(e).__name__}
                    )
            else:
                logger.info("Step 2/5: Skipping energy analysis (skip_simulation=True)")

            # Step 3: Generate maintenance plan
            logger.info("Step 3/5: Generating maintenance plan...")
            try:
                maintenance_plan = self._generate_maintenance_plan(building_data, analysis_results)
            except Exception as e:
                # Graceful degradation - continue without maintenance plan
                logger.warning(
                    f"Maintenance plan generation failed: {e}",
                    extra={"error_type": type(e).__name__}
                )
                maintenance_plan = None

            # Step 4: Analyze effektvakt potential
            logger.info("Step 4/5: Analyzing effektvakt potential...")
            try:
                effektvakt_result = self._analyze_effektvakt(building_data)
            except Exception as e:
                # Graceful degradation - continue without effektvakt
                logger.warning(
                    f"Effektvakt analysis failed: {e}",
                    extra={"error_type": type(e).__name__}
                )
                effektvakt_result = None

            # Step 5: Generate report
            report_path = None
            if generate_report:
                logger.info("Step 5/5: Generating HTML report...")
                try:
                    report_path = self._generate_report(
                        building_data,
                        analysis_results,
                        maintenance_plan,
                        effektvakt_result,
                    )
                except Exception as e:
                    logger.error(
                        f"Report generation failed: {e}",
                        extra={"error_type": type(e).__name__}
                    )
            else:
                logger.info("Step 5/5: Skipping report generation")

            processing_time = time.time() - start_time
            logger.info(
                f"Analysis complete in {processing_time:.1f}s",
                extra={"address": address, "processing_time_seconds": processing_time}
            )

            return PipelineResult(
                success=True,
                address=address,
                building_data=building_data,
                analysis_results=analysis_results,
                maintenance_plan=maintenance_plan,
                effektvakt_result=effektvakt_result,
                report_path=report_path,
                processing_time_seconds=processing_time,
            )

        except ValidationError as e:
            logger.error(
                f"Validation error: {e}",
                extra={"address": address, "error_type": "validation"}
            )
            suggestions = getattr(e, 'suggestions', [])
            return PipelineResult(
                success=False,
                address=address,
                error=f"Validation error: {str(e)}. {' '.join(suggestions) if suggestions else ''}",
                processing_time_seconds=time.time() - start_time,
            )
        except ConnectionError as e:
            logger.error(
                f"Network error - external services unavailable: {e}",
                extra={"address": address, "error_type": "network"}
            )
            return PipelineResult(
                success=False,
                address=address,
                error=f"Network error: Unable to connect to external services. Please check your internet connection and try again.",
                processing_time_seconds=time.time() - start_time,
            )
        except Exception as e:
            logger.error(
                f"Pipeline error: {e}",
                extra={"address": address, "error_type": type(e).__name__},
                exc_info=True
            )
            return PipelineResult(
                success=False,
                address=address,
                error=f"Analysis failed: {str(e)}. Please check the logs for details.",
                processing_time_seconds=time.time() - start_time,
            )

    def _run_energy_analysis(self, building_data: BuildingData) -> Optional[Any]:
        """
        Run comprehensive energy analysis using FullPipelineAnalyzer.

        This connects to the full analysis pipeline which includes:
        - Bayesian calibration with GP surrogates
        - EnergyPlus simulation
        - 51 ECM analysis with cost calculation
        - Ground floor commercial detection
        - Mixed-use zone modeling
        """
        import asyncio

        try:
            from ..analysis.full_pipeline import FullPipelineAnalyzer

            # Get API keys from environment
            google_api_key = os.environ.get("GOOGLE_API_KEY")
            mapillary_token = os.environ.get("MAPILLARY_TOKEN")

            # Initialize the full pipeline analyzer
            analyzer = FullPipelineAnalyzer(
                google_api_key=google_api_key,
                mapillary_token=mapillary_token,
                output_dir=self.output_dir,
                use_bayesian_calibration=True,
            )

            # Prepare building data dict for the analyzer
            building_dict = {
                "address": building_data.address,
                "construction_year": building_data.construction_year,
                "atemp_m2": building_data.atemp_m2,
                "num_floors": building_data.num_floors,
                "num_apartments": building_data.num_apartments,
                "facade_material": building_data.facade_material,
                "heating_system": building_data.heating_system,
                "has_ftx": building_data.has_ftx,
                "has_heat_pump": building_data.has_heat_pump,
                "has_solar": building_data.has_solar,
                # Detailed energy source kWh (crucial for existing measures detection!)
                "exhaust_air_hp_kwh": building_data.exhaust_air_hp_kwh,
                "ground_source_hp_kwh": building_data.ground_source_hp_kwh,
                "air_source_hp_kwh": building_data.air_source_hp_kwh,
                "district_heating_kwh": building_data.district_heating_kwh,
                "declared_energy_kwh_m2": building_data.declared_energy_kwh_m2,
                "energy_class": building_data.energy_class,
                # Geometry data (critical for EnergyPlus simulation)
                "footprint_area_m2": building_data.footprint_area_m2,
                "height_m": building_data.height_m,
                "footprint_coords": building_data.footprint_coords,
                "building_width_m": building_data.building_width_m,
                "building_length_m": building_data.building_length_m,
                # Multi-building property data (for multi-roof solar analysis)
                "building_details": building_data.building_details,
                "property_designation": building_data.property_designation,
                "all_addresses": building_data.all_addresses,
            }

            # Run async analyzer (convert to sync)
            logger.info("Running full energy analysis pipeline...")

            # Use asyncio.run for sync context, or get existing loop
            try:
                loop = asyncio.get_running_loop()
                # Already in async context
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        analyzer.analyze(
                            address=building_data.address,
                            lat=building_data.latitude,
                            lon=building_data.longitude,
                            building_data=building_dict,
                            run_simulations=True,
                        )
                    )
                    result = future.result(timeout=600)  # 10 min timeout
            except RuntimeError:
                # No event loop running, safe to use asyncio.run
                result = asyncio.run(
                    analyzer.analyze(
                        address=building_data.address,
                        lat=building_data.latitude,
                        lon=building_data.longitude,
                        building_data=building_dict,
                        run_simulations=True,
                    )
                )

            logger.info(f"Energy analysis complete: {len(result.get('ecm_results', []))} ECMs analyzed")

            # Update building_data with AI-detected material from full_pipeline
            # This is more accurate than era-based guessing
            detected_material = result.get("facade_material") or result.get("building_context", {}).get("facade_material")
            if detected_material and detected_material != "unknown":
                # Normalize "render" to "plaster" for consistency in reports
                if detected_material == "render":
                    detected_material = "plaster"
                building_data.facade_material = detected_material
                logger.info(f"Updated facade material from AI detection: {detected_material}")

            return result

        except ImportError as e:
            logger.warning(f"FullPipelineAnalyzer not available: {e}")
            return None
        except Exception as e:
            logger.warning(f"Energy analysis failed: {e}", exc_info=True)
            return None

    def _generate_maintenance_plan(
        self,
        building_data: BuildingData,
        analysis_results: Any = None,
    ) -> Any:
        """Generate maintenance plan with cash flow cascade.

        Uses ACTUAL ECM simulation results if available, otherwise falls back
        to hardcoded estimates.
        """
        try:
            from ..planning import (
                MaintenancePlan,
                BRFFinancials,
                ECMSequencer,
                ECMCandidate,
                CashFlowSimulator,
            )

            # Create BRF financials
            financials = BRFFinancials(
                current_fund_sek=building_data.current_fund_sek or 500_000,
                annual_fund_contribution_sek=building_data.annual_fund_contribution_sek or 200_000,
                current_avgift_sek_month=4500,
                num_apartments=building_data.num_apartments or 50,
                annual_energy_cost_sek=building_data.annual_energy_cost_sek or 500_000,
                peak_el_kw=building_data.peak_el_kw,
                peak_fv_kw=building_data.peak_fv_kw,
            )

            # TRY to use ACTUAL ECM results from simulation (not hardcoded!)
            candidates = self._extract_ecm_candidates_from_results(
                analysis_results, building_data
            )

            # Fall back to hardcoded candidates only if no simulation results
            if not candidates:
                logger.info("No simulation results - using estimated ECM candidates")
                candidates = self._create_ecm_candidates(building_data)

            # Create optimal sequence
            sequencer = ECMSequencer()
            plan = sequencer.create_optimal_plan(
                candidates=candidates,
                financials=financials,
                renovations=[],
                start_year=datetime.now().year,
                plan_horizon_years=15,
            )

            # Simulate cash flow
            simulator = CashFlowSimulator()
            plan = simulator.simulate(plan, start_year=datetime.now().year)

            plan.brf_name = building_data.address
            plan.atemp_m2 = building_data.atemp_m2
            plan.num_apartments = building_data.num_apartments
            plan.construction_year = building_data.construction_year

            return plan

        except Exception as e:
            logger.warning(f"Maintenance plan generation failed: {e}")
            return None

    def _create_ecm_candidates(self, building_data: BuildingData) -> List:
        """Create ECM candidates based on building characteristics."""
        from ..planning import ECMCandidate

        atemp = building_data.atemp_m2

        candidates = [
            # Steg 0: Zero-cost
            ECMCandidate(
                ecm_id="duc_calibration",
                name="DUC-optimering",
                investment_sek=5000,
                annual_savings_sek=max(10000, atemp * 2),
                payback_years=0.2,
                is_zero_cost=True,
                steg=0,
            ),
            ECMCandidate(
                ecm_id="heating_curve_adjustment",
                name="Värmekurvejustering",
                investment_sek=2000,
                annual_savings_sek=max(10000, atemp * 2),
                payback_years=0.1,
                is_zero_cost=True,
                steg=0,
            ),
            ECMCandidate(
                ecm_id="night_setback",
                name="Nattsänkning",
                investment_sek=1000,
                annual_savings_sek=max(8000, atemp * 1.5),
                payback_years=0.1,
                is_zero_cost=True,
                steg=0,
            ),
            ECMCandidate(
                ecm_id="effektvakt_optimization",
                name="Effektvaktsoptimering",
                investment_sek=3000,
                annual_savings_sek=max(15000, building_data.peak_el_kw * 59 * 12 * 0.3),
                payback_years=0.1,
                is_zero_cost=True,
                steg=0,
            ),

            # Steg 1: Quick wins (< 500k SEK)
            ECMCandidate(
                ecm_id="air_sealing",
                name="Tätning",
                investment_sek=min(300000, atemp * 30),
                annual_savings_sek=max(50000, atemp * 8),
                payback_years=4,
                steg=1,
            ),
            ECMCandidate(
                ecm_id="smart_thermostats",
                name="Smarta termostater",
                investment_sek=building_data.num_apartments * 2000 if building_data.num_apartments else 100000,
                annual_savings_sek=max(20000, atemp * 2),
                payback_years=5,
                steg=1,
            ),

            # Steg 2: Standard (500k - 2M SEK)
            ECMCandidate(
                ecm_id="demand_controlled_ventilation",
                name="Behovsstyrd ventilation",
                investment_sek=min(1500000, atemp * 100),
                annual_savings_sek=max(100000, atemp * 20),
                payback_years=8,
                steg=2,
            ),
        ]

        # Add roof insulation if old building
        if building_data.construction_year < 1990:
            candidates.append(ECMCandidate(
                ecm_id="roof_insulation",
                name="Tilläggsisolering tak",
                investment_sek=min(800000, atemp * 50),
                annual_savings_sek=max(20000, atemp * 2),
                payback_years=20,
                steg=2,
            ))

        # Steg 3: Premium (> 2M SEK)
        if building_data.construction_year < 1985:
            candidates.append(ECMCandidate(
                ecm_id="wall_internal_insulation",
                name="Tilläggsisolering väggar",
                investment_sek=min(4000000, atemp * 200),
                annual_savings_sek=max(80000, atemp * 8),
                payback_years=25,
                steg=3,
            ))

        return candidates

    def _extract_ecm_candidates_from_results(
        self,
        analysis_results: Any,
        building_data: BuildingData,
    ) -> List:
        """Extract ECM candidates from Snowball Packages (preferred) or raw ECM results.

        CRITICAL FIX (2025-01-06): This function now uses the PACKAGES structure from
        full_pipeline.py instead of re-processing ecm_results. This ensures the Cash Flow
        table matches the Steg cards (Package cards) in the report.

        Priority:
        1. If packages exist → use them directly (preserves Steg 1-3 grouping)
        2. If no packages → fall back to raw ecm_results processing

        Args:
            analysis_results: Full pipeline results with ecm_results and packages
            building_data: Building characteristics for category assignment

        Returns:
            List of ECMCandidate with actual simulation-based costs/savings
        """
        from ..planning import ECMCandidate

        if not analysis_results:
            return []

        # Get ECM results and packages from simulation
        ecm_results = analysis_results.get("ecm_results", [])
        # CRITICAL FIX (2025-01-06): full_pipeline uses "snowball_packages" key, not "packages"
        packages = analysis_results.get("snowball_packages", []) or analysis_results.get("packages", [])

        candidates = []

        # ═══════════════════════════════════════════════════════════════════════
        # PRIORITY 1: Use packages if available (ensures Cash Flow matches Steg cards)
        # ═══════════════════════════════════════════════════════════════════════
        if packages:
            logger.info(f"Using {len(packages)} Snowball packages for ECM candidates")

            # Build ECM lookup from ecm_results for cost/savings data
            ecm_lookup = {ecm.get("ecm_id", ""): ecm for ecm in ecm_results}

            for pkg in packages:
                # Handle both SnowballPackage dataclasses and dicts
                if hasattr(pkg, 'package_name'):
                    # SnowballPackage dataclass from full_pipeline
                    pkg_name = pkg.package_name
                    ecm_ids = pkg.ecm_ids
                    pkg_cost = pkg.total_investment_sek
                    pkg_payback = pkg.simple_payback_years
                    pkg_number = pkg.package_number  # Direct steg number!
                else:
                    # Dict from cached/serialized data
                    pkg_name = pkg.get("name", "")
                    ecm_ids = pkg.get("ecm_ids", [])
                    pkg_cost = pkg.get("estimated_cost_sek", pkg.get("total_investment_sek", 0))
                    pkg_payback = pkg.get("simple_payback_years", 99)
                    pkg_number = pkg.get("package_number", 0)

                # Extract steg number from package_number or package name
                if pkg_number and pkg_number > 0:
                    steg = pkg_number
                else:
                    # Fallback: Extract from package name (e.g., "Steg 1: Snabba Vinster" → 1)
                    steg = 1  # Default
                    if "Steg 1" in pkg_name or "steg 1" in pkg_name.lower():
                        steg = 1
                    elif "Steg 2" in pkg_name or "steg 2" in pkg_name.lower():
                        steg = 2
                    elif "Steg 3" in pkg_name or "steg 3" in pkg_name.lower():
                        steg = 3
                    elif "Steg 0" in pkg_name or "steg 0" in pkg_name.lower():
                        steg = 0

                # Create ECMCandidate for each ECM in the package
                for ecm_id in ecm_ids:
                    # Look up ECM details from ecm_results
                    ecm_data = ecm_lookup.get(ecm_id, {})

                    investment = ecm_data.get("investment_sek", 0)
                    annual_savings = ecm_data.get("annual_savings_sek", 0)
                    payback = ecm_data.get("simple_payback_years", pkg_payback)
                    ecm_name = ecm_data.get("ecm_name", ecm_id)

                    # For package-level costs, distribute if ECM has no individual cost
                    if investment == 0 and len(ecm_ids) > 0:
                        investment = pkg_cost / len(ecm_ids)

                    is_zero_cost = steg == 0 or investment < 50000

                    candidates.append(ECMCandidate(
                        ecm_id=ecm_id,
                        name=ecm_name,
                        investment_sek=investment,
                        annual_savings_sek=annual_savings,
                        payback_years=payback,
                        is_zero_cost=is_zero_cost,
                        steg=steg,
                    ))

            logger.info(
                f"Extracted {len(candidates)} ECM candidates from packages "
                f"(Steg 0: {sum(1 for c in candidates if c.steg == 0)}, "
                f"Steg 1: {sum(1 for c in candidates if c.steg == 1)}, "
                f"Steg 2: {sum(1 for c in candidates if c.steg == 2)}, "
                f"Steg 3: {sum(1 for c in candidates if c.steg == 3)})"
            )
            return candidates

        # ═══════════════════════════════════════════════════════════════════════
        # FALLBACK: Use raw ecm_results if no packages exist
        # ═══════════════════════════════════════════════════════════════════════
        if not ecm_results:
            return []

        logger.info("No packages found, falling back to raw ecm_results processing")

        # ECM category mapping for steg assignment (fallback only)
        steg_0_ecms = {
            "duc_calibration", "heating_curve_adjustment", "night_setback",
            "effektvakt_optimization", "bms_optimization", "pump_optimization",
            "ventilation_schedule_optimization", "hot_water_temperature",
            "dhw_circulation_optimization", "radiator_balancing",
            "district_heating_optimization",
        }
        steg_1_ecms = {
            "air_sealing", "smart_thermostats", "led_lighting", "led_common_areas",
            "led_outdoor", "occupancy_sensors", "daylight_sensors", "low_flow_fixtures",
            "dhw_tank_insulation", "pipe_insulation", "radiator_fans",
            "entrance_door_replacement",
        }
        steg_2_ecms = {
            "demand_controlled_ventilation", "roof_insulation", "ftx_overhaul",
            "heat_recovery_dhw", "solar_thermal", "individual_metering",
            "building_automation_system", "thermal_bridge_remediation",
            "basement_insulation",
        }

        for ecm in ecm_results:
            ecm_id = ecm.get("ecm_id", "")
            savings_percent = ecm.get("savings_percent", 0)

            # Skip ECMs with no savings or negative savings
            if savings_percent <= 0:
                continue

            investment = ecm.get("investment_sek", 0)
            annual_savings = ecm.get("annual_savings_sek", 0)
            payback = ecm.get("simple_payback_years", 99)

            # Determine steg based on ECM type AND investment amount
            if ecm_id in steg_0_ecms or investment < 50000:
                steg = 0
                is_zero_cost = True
            elif ecm_id in steg_1_ecms or investment < 500000:
                steg = 1
                is_zero_cost = False
            elif ecm_id in steg_2_ecms or investment < 2000000:
                steg = 2
                is_zero_cost = False
            else:
                steg = 3
                is_zero_cost = False

            ecm_name = ecm.get("ecm_name", ecm_id)

            candidates.append(ECMCandidate(
                ecm_id=ecm_id,
                name=ecm_name,
                investment_sek=investment,
                annual_savings_sek=annual_savings,
                payback_years=payback,
                is_zero_cost=is_zero_cost,
                steg=steg,
            ))

        logger.info(
            f"Extracted {len(candidates)} ECM candidates from ecm_results (fallback) "
            f"(Steg 0: {sum(1 for c in candidates if c.steg == 0)}, "
            f"Steg 1: {sum(1 for c in candidates if c.steg == 1)}, "
            f"Steg 2: {sum(1 for c in candidates if c.steg == 2)}, "
            f"Steg 3: {sum(1 for c in candidates if c.steg == 3)})"
        )

        return candidates

    def _analyze_effektvakt(self, building_data: BuildingData) -> Any:
        """Analyze effektvakt (peak shaving) potential."""
        try:
            from ..planning import analyze_effektvakt_potential

            # Determine heating type
            if building_data.has_heat_pump:
                heating_type = "heat_pump"
            else:
                heating_type = "district"

            result = analyze_effektvakt_potential(
                atemp_m2=building_data.atemp_m2,
                construction_year=building_data.construction_year,
                heating_type=heating_type,
                current_el_peak_kw=building_data.peak_el_kw,
                current_fv_peak_kw=building_data.peak_fv_kw,
            )

            return result

        except Exception as e:
            logger.warning(f"Effektvakt analysis failed: {e}")
            return None

    def _create_building_json(self, building_data: BuildingData) -> Dict:
        """Create building JSON structure for analyzer."""
        return {
            "brf_name": building_data.address,
            "original_summary": {
                "location": "Stockholm",
                "construction_year": building_data.construction_year,
                "total_heated_area_sqm": building_data.atemp_m2,
                "floors": building_data.num_floors,
                "energy_performance_kwh_per_sqm": building_data.declared_energy_kwh_m2,
                "energy_class": building_data.energy_class,
                "building_type": building_data.building_type,
            },
            "buildings": [{
                "address": building_data.address,
                "envelope": {
                    "facade_material": building_data.facade_material,
                },
            }],
        }

    def _generate_report(
        self,
        building_data: BuildingData,
        analysis_results: Any,
        maintenance_plan: Any,
        effektvakt_result: Any,
    ) -> Path:
        """Generate HTML report."""
        from ..reporting.html_report import (
            HTMLReportGenerator,
            ReportData,
            MaintenancePlanData,
            EffektvaktData,
            CalibrationAnomaliesData,
        )
        # Import calibration anomaly converter if available
        try:
            from ..agents import calibration_analysis_to_report_data
            AGENTIC_AVAILABLE = True
        except ImportError:
            AGENTIC_AVAILABLE = False

        # Build maintenance plan data for report
        mp_data = None
        if maintenance_plan:
            projections = []
            for proj in maintenance_plan.projections[:15]:
                projections.append({
                    "year": proj.year,
                    "fund_start_sek": proj.fund_start_sek,
                    "fund_contribution_sek": proj.fund_contribution_sek,  # Annual fund contribution from avgift
                    "investment_sek": proj.renovation_spend_sek + proj.ecm_investment_sek,
                    "energy_savings_sek": proj.energy_savings_sek,
                    "fund_end_sek": proj.fund_end_sek,
                    "loan_balance_sek": proj.loan_balance_sek,
                    "fund_warning": proj.fund_warning,
                    "ecm_investments": proj.ecm_investments,
                })

            # Calculate zero-cost savings
            zero_cost_savings = sum(
                inv.annual_savings_sek
                for inv in maintenance_plan.ecm_investments
                if inv.investment_sek < 20000  # Zero-cost threshold
            )

            mp_data = MaintenancePlanData(
                total_investment_sek=maintenance_plan.total_investment_sek,
                total_savings_30yr_sek=maintenance_plan.total_savings_30yr_sek,
                net_present_value_sek=maintenance_plan.net_present_value_sek,
                break_even_year=maintenance_plan.break_even_year,
                final_fund_balance_sek=maintenance_plan.final_fund_balance_sek,
                max_loan_used_sek=maintenance_plan.max_loan_used_sek,
                projections=projections,
                zero_cost_annual_savings=zero_cost_savings,
            )

        # Build effektvakt data for report
        eff_data = None
        if effektvakt_result:
            eff_data = EffektvaktData(
                current_el_peak_kw=effektvakt_result.current_el_peak_kw,
                current_fv_peak_kw=effektvakt_result.current_fv_peak_kw,
                optimized_el_peak_kw=effektvakt_result.optimized_el_peak_kw,
                optimized_fv_peak_kw=effektvakt_result.optimized_fv_peak_kw,
                el_peak_reduction_kw=effektvakt_result.el_peak_reduction_kw,
                fv_peak_reduction_kw=effektvakt_result.fv_peak_reduction_kw,
                annual_el_savings_sek=effektvakt_result.annual_el_savings_sek,
                annual_fv_savings_sek=effektvakt_result.annual_fv_savings_sek,
                total_annual_savings_sek=effektvakt_result.total_annual_savings_sek,
                pre_heat_hours=effektvakt_result.pre_heat_hours,
                pre_heat_temp_c=effektvakt_result.pre_heat_temp_increase_c,
                coast_duration_hours=effektvakt_result.coast_duration_hours,
                requires_bms=effektvakt_result.requires_bms,
                manual_possible=effektvakt_result.manual_possible,
                notes=effektvakt_result.notes or [],
            )

        # Extract ECM results from analysis
        from ..reporting.html_report import ECMResult

        ecm_results_list = []
        baseline_kwh_m2 = building_data.declared_energy_kwh_m2

        if analysis_results:
            # Get baseline from calibration if available
            baseline_kwh_m2 = analysis_results.get("baseline_kwh_m2", baseline_kwh_m2)

            # Convert ECM result dicts to ECMResult objects
            raw_ecm_results = analysis_results.get("ecm_results", [])
            for ecm in raw_ecm_results:
                try:
                    # Get result kWh - full_pipeline uses "heating_kwh_m2" key
                    result_kwh = ecm.get("heating_kwh_m2") or ecm.get("result_kwh_m2") or baseline_kwh_m2
                    savings_kwh = baseline_kwh_m2 - result_kwh

                    ecm_results_list.append(ECMResult(
                        id=ecm.get("ecm_id", "unknown"),
                        name=ecm.get("ecm_name", ecm.get("ecm_id", "Unknown")),
                        name_sv=ecm.get("name_sv", ecm.get("ecm_name", ecm.get("ecm_id", "Unknown"))),
                        category=ecm.get("category", "Other"),
                        baseline_kwh_m2=baseline_kwh_m2,
                        result_kwh_m2=result_kwh,
                        savings_kwh_m2=savings_kwh,
                        savings_percent=ecm.get("savings_percent", 0),
                        estimated_cost_sek=ecm.get("investment_sek", 0),
                        simple_payback_years=ecm.get("simple_payback_years", 99),
                        # Multi-end-use energy tracking
                        total_kwh_m2=ecm.get("total_kwh_m2", 0),
                        total_savings_percent=ecm.get("total_savings_percent", 0),
                        heating_kwh_m2=ecm.get("heating_kwh_m2", result_kwh),
                        dhw_kwh_m2=ecm.get("dhw_kwh_m2", 0),
                        property_el_kwh_m2=ecm.get("property_el_kwh_m2", 0),
                        savings_by_end_use=ecm.get("savings_by_end_use"),
                    ))
                except Exception as e:
                    logger.warning(f"Failed to parse ECM result: {e}")

        # FALLBACK: Create ECMResult objects from maintenance plan when analysis fails
        # This ensures the report has ECM data even without EnergyPlus simulation
        if not ecm_results_list and maintenance_plan and hasattr(maintenance_plan, 'ecm_investments'):
            logger.info("Creating ECM results from maintenance plan (fallback)")
            for inv in maintenance_plan.ecm_investments:
                if hasattr(inv, 'ecm_id') and hasattr(inv, 'investment_sek'):
                    # Calculate approximate savings percent from payback
                    annual_savings = getattr(inv, 'annual_savings_sek', 0) or 0
                    # Estimate kWh savings from SEK savings (assuming ~1 SEK/kWh for district heating)
                    savings_kwh = annual_savings / (building_data.atemp_m2 or 1)
                    savings_pct = (savings_kwh / baseline_kwh_m2 * 100) if baseline_kwh_m2 > 0 else 0

                    # Get Swedish name from ECM catalog if available
                    from ..ecm import get_ecm
                    ecm_catalog = get_ecm(inv.ecm_id)
                    fallback_name = getattr(inv, 'name', inv.ecm_id.replace('_', ' ').title())
                    name_sv = ecm_catalog.name_sv if ecm_catalog else fallback_name

                    ecm_results_list.append(ECMResult(
                        id=inv.ecm_id,
                        name=fallback_name,
                        name_sv=name_sv,
                        category="Estimated",  # Mark as estimated since no simulation
                        baseline_kwh_m2=baseline_kwh_m2,
                        result_kwh_m2=max(0, baseline_kwh_m2 - savings_kwh),
                        savings_kwh_m2=savings_kwh,
                        savings_percent=savings_pct,
                        estimated_cost_sek=getattr(inv, 'investment_sek', 0),
                        simple_payback_years=getattr(inv, 'payback_years', 99),
                    ))
            if ecm_results_list:
                logger.info(f"Created {len(ecm_results_list)} ECM results from maintenance plan")

        # Extract existing measures from context (convert enum values to strings)
        existing_measures = []
        if analysis_results and "context" in analysis_results:
            context = analysis_results["context"]
            if hasattr(context, "existing_measures") and context.existing_measures:
                # Convert ExistingMeasure enums to their string values
                existing_measures = [m.value if hasattr(m, 'value') else str(m) for m in context.existing_measures]

        # FALLBACK: Extract existing measures from building_data if analysis failed
        # This ensures we show existing measures even when EnergyPlus analysis fails
        if not existing_measures:
            logger.info("Extracting existing measures from building_data (fallback)")
            # Check for FTX/heat recovery ventilation
            if building_data.has_ftx:
                existing_measures.append("ftx_system")
                logger.info("  - Detected FTX system from building_data")

            # Check for heat pumps (specific types)
            if building_data.ground_source_hp_kwh and building_data.ground_source_hp_kwh > 0:
                existing_measures.append("heat_pump_ground")
                logger.info(f"  - Detected ground source HP ({building_data.ground_source_hp_kwh:.0f} kWh)")
            if building_data.exhaust_air_hp_kwh and building_data.exhaust_air_hp_kwh > 0:
                existing_measures.append("heat_pump_exhaust")
                logger.info(f"  - Detected exhaust air HP ({building_data.exhaust_air_hp_kwh:.0f} kWh)")
            if building_data.air_source_hp_kwh and building_data.air_source_hp_kwh > 0:
                existing_measures.append("heat_pump_air")
                logger.info(f"  - Detected air source HP ({building_data.air_source_hp_kwh:.0f} kWh)")

            # Generic heat pump check (if no specific type detected)
            if building_data.has_heat_pump and not any("heat_pump" in m for m in existing_measures):
                existing_measures.append("heat_pump_generic")
                logger.info("  - Detected generic heat pump from building_data")

            # Check for solar
            if building_data.has_solar:
                existing_measures.append("solar_pv")
                logger.info("  - Detected solar PV from building_data")

            if existing_measures:
                logger.info(f"Fallback detected {len(existing_measures)} existing measures: {existing_measures}")

        # Update building_data with correct data sources from fusion
        if analysis_results and "data_fusion" in analysis_results:
            fusion = analysis_results["data_fusion"]
            if hasattr(fusion, "data_sources") and fusion.data_sources:
                # Use fusion sources which correctly includes google_streetview, google_solar, etc.
                building_data.data_sources = fusion.data_sources

        # Get applicable ECM IDs
        applicable_ecm_ids = [e.id for e in ecm_results_list]

        # FALLBACK: If analysis failed but maintenance plan exists, extract ECM IDs from it
        # This ensures consistency between "Tillämpliga åtgärder" and "Underhållsplan" sections
        if not applicable_ecm_ids and maintenance_plan and hasattr(maintenance_plan, 'ecm_investments'):
            applicable_ecm_ids = [inv.ecm_id for inv in maintenance_plan.ecm_investments if hasattr(inv, 'ecm_id')]
            if applicable_ecm_ids:
                logger.info(f"Extracted {len(applicable_ecm_ids)} applicable ECM IDs from maintenance plan (fallback)")

        # Convert snowball packages to report format
        from ..analysis.package_generator import ECMPackage, ECMPackageItem
        report_packages = []
        prev_class = ''  # Track previous package's after_energy_class for progressive display
        if analysis_results and "snowball_packages" in analysis_results:
            snowball_pkgs = analysis_results["snowball_packages"]
            for pkg in snowball_pkgs:
                # Create ECMPackageItem for each ECM in the package
                ecm_items = []
                for ecm_id in pkg.ecm_ids:
                    # Find the ECM result
                    matching_ecm = next((e for e in ecm_results_list if e.id == ecm_id), None)
                    if matching_ecm:
                        # Use total_savings_percent (all end-uses) if available, else heating-only
                        ecm_savings_pct = matching_ecm.total_savings_percent if matching_ecm.total_savings_percent else matching_ecm.savings_percent
                        # Use Swedish name if available, otherwise English
                        ecm_name_sv = getattr(matching_ecm, 'name_sv', None) or matching_ecm.name
                        ecm_items.append(ECMPackageItem(
                            id=ecm_id,
                            name=matching_ecm.name,
                            name_sv=ecm_name_sv,
                            individual_savings_percent=ecm_savings_pct,
                            estimated_cost_sek=matching_ecm.estimated_cost_sek,
                        ))

                # Create ECMPackage with primary energy info
                # Build description including energy class improvement if available
                # Use PREVIOUS package's after_class for progressive display
                display_before_class = prev_class if prev_class else pkg.before_energy_class
                after_class = pkg.after_energy_class

                # Calculate classes improved from display_before to after
                class_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
                try:
                    before_idx = class_order.index(display_before_class) if display_before_class else 6
                    after_idx = class_order.index(after_class) if after_class else 6
                    classes_improved_display = before_idx - after_idx
                except ValueError:
                    classes_improved_display = 0

                # Build description - avoid duplicating "Steg N:" if already in package_name
                pkg_name = pkg.package_name
                if pkg_name.startswith(f"Steg {pkg.package_number}"):
                    desc = pkg_name  # Already has Steg prefix
                else:
                    desc = f"Steg {pkg.package_number}: {pkg_name}"

                if display_before_class and after_class:
                    desc += f" | Energiklass: {display_before_class} → {after_class}"
                    if classes_improved_display > 0:
                        desc += f" (+{classes_improved_display})"

                # Update prev_class for next iteration
                if after_class:
                    prev_class = after_class

                report_packages.append(ECMPackage(
                    id=f"pkg_{pkg.package_number}",
                    name=pkg.package_name,
                    description=desc,
                    ecms=ecm_items,
                    combined_savings_percent=pkg.savings_percent,
                    combined_savings_kwh_m2=pkg.combined_kwh_m2,
                    total_cost_sek=pkg.total_investment_sek,
                    simple_payback_years=pkg.simple_payback_years,
                    annual_cost_savings_sek=pkg.annual_savings_sek,
                    co2_reduction_kg_m2=0,  # Not calculated in snowball packages
                    # Primary energy & energy class (Swedish BBR)
                    before_primary_kwh_m2=getattr(pkg, 'before_primary_kwh_m2', 0),
                    after_primary_kwh_m2=getattr(pkg, 'after_primary_kwh_m2', 0),
                    primary_savings_percent=getattr(pkg, 'primary_savings_percent', 0),
                    before_energy_class=getattr(pkg, 'before_energy_class', ''),
                    after_energy_class=getattr(pkg, 'after_energy_class', ''),
                    classes_improved=getattr(pkg, 'classes_improved', 0),
                    # Energy progression (total energy)
                    before_total_kwh_m2=getattr(pkg, 'before_total_kwh_m2', 0),
                    after_total_kwh_m2=getattr(pkg, 'after_total_kwh_m2', 0),
                    cumulative_savings_percent=getattr(pkg, 'cumulative_savings_percent', 0),
                    # Fund-based timing
                    fund_recommended_year=getattr(pkg, 'fund_recommended_year', 0),
                    fund_available_sek=getattr(pkg, 'fund_available_sek', 0),
                    years_to_afford=getattr(pkg, 'years_to_afford', 0),
                ))

        # Get baseline energy breakdown from analysis results
        baseline_dhw_kwh_m2 = 0
        baseline_property_el_kwh_m2 = 0
        baseline_cooling_kwh_m2 = 0
        baseline_total_kwh_m2 = 0
        if analysis_results and "baseline_energy" in analysis_results:
            be = analysis_results["baseline_energy"]
            baseline_dhw_kwh_m2 = be.get("dhw_kwh_m2", 0) if isinstance(be, dict) else getattr(be, "dhw_kwh_m2", 0)
            baseline_property_el_kwh_m2 = be.get("property_el_kwh_m2", 0) if isinstance(be, dict) else getattr(be, "property_el_kwh_m2", 0)
            baseline_cooling_kwh_m2 = be.get("cooling_kwh_m2", 0) if isinstance(be, dict) else getattr(be, "cooling_kwh_m2", 0)
            baseline_total_kwh_m2 = be.get("total_kwh_m2", 0) if isinstance(be, dict) else getattr(be, "total_kwh_m2", 0)

        # Extract calibration anomalies from Agentic Raiden analysis
        calibration_anomalies_data = None
        if analysis_results and AGENTIC_AVAILABLE:
            calibration_analysis = analysis_results.get("calibration_analysis")
            if calibration_analysis and hasattr(calibration_analysis, 'has_anomalies'):
                try:
                    calibration_anomalies_data = calibration_analysis_to_report_data(calibration_analysis)
                    logger.info(f"Calibration anomalies: {len(calibration_anomalies_data.anomalies)} anomalies detected")
                except Exception as e:
                    logger.warning(f"Failed to convert calibration anomalies: {e}")

        # Build report data
        report_data = ReportData(
            building_name=building_data.address,
            address=building_data.address,
            construction_year=building_data.construction_year,
            building_type=building_data.building_type,
            facade_material=building_data.facade_material,
            atemp_m2=building_data.atemp_m2,
            floors=building_data.num_floors,
            energy_class=building_data.energy_class,
            declared_heating_kwh_m2=building_data.declared_energy_kwh_m2,
            baseline_heating_kwh_m2=baseline_kwh_m2,
            existing_measures=existing_measures,
            applicable_ecms=applicable_ecm_ids,
            excluded_ecms=[],  # Could populate from filter result if available
            ecm_results=ecm_results_list,
            # Multi-end-use energy breakdown
            baseline_dhw_kwh_m2=baseline_dhw_kwh_m2,
            baseline_property_el_kwh_m2=baseline_property_el_kwh_m2,
            baseline_cooling_kwh_m2=baseline_cooling_kwh_m2,
            baseline_total_kwh_m2=baseline_total_kwh_m2,
            packages=report_packages,
            maintenance_plan=mp_data,
            effektvakt=eff_data,
            num_apartments=building_data.num_apartments,
            current_fund_sek=building_data.current_fund_sek,
            annual_energy_cost_sek=building_data.annual_energy_cost_sek,
            analysis_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            # Agentic Raiden calibration anomalies
            calibration_anomalies=calibration_anomalies_data,
        )

        # Generate and save report
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create safe filename from address
        safe_name = "".join(c if c.isalnum() else "_" for c in building_data.address)[:50]
        report_path = self.output_dir / f"report_{safe_name}.html"

        generator = HTMLReportGenerator()
        generator.generate(report_data, report_path)

        logger.info(f"Report saved to: {report_path}")
        return report_path


def analyze_address(
    address: str,
    output_dir: Optional[Path] = None,
    **kwargs
) -> PipelineResult:
    """
    Convenience function to analyze a building by address.

    Args:
        address: Swedish street address
        output_dir: Output directory for reports
        **kwargs: Additional arguments passed to pipeline.analyze()

    Returns:
        PipelineResult with analysis results

    Example:
        >>> result = analyze_address("Aktergatan 11, Stockholm")
        >>> print(f"Report: {result.report_path}")
    """
    pipeline = AddressPipeline(output_dir=output_dir)
    return pipeline.analyze(address, **kwargs)
