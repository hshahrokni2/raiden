"""
Swedish Buildings GeoJSON loader.

Loads the comprehensive Swedish building dataset (Lantmäteriet + Boverket)
with 37,489 buildings and 167 properties per building.

Data includes:
- Building footprints (EPSG:3006 SWEREF99 TM)
- Building heights (3D coordinates)
- Energy declarations (energy class, kWh/m², heating systems)
- Ventilation type (F, FT, FTX, natural)
- Heat pumps (ground source, exhaust air, air-water)
- Solar (PV, thermal)
- Address and location data

Source: Generated_Buildings.geojson (260MB, 37,489 buildings)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console

from ..core.config import settings

console = Console()

# Default path to the Swedish buildings GeoJSON
DEFAULT_GEOJSON_PATH = Path(__file__).parent.parent.parent / "data" / "sweden_buildings.geojson"


def _parse_pct(value: Any) -> float:
    """Parse percentage value from GeoJSON (handles empty strings, None, floats)."""
    if value is None or value == "" or value == "":
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def _parse_airflow(value: Any) -> Optional[float]:
    """Parse airflow value from GeoJSON (handles Swedish decimal format)."""
    if value is None or value == "" or value == "":
        return None
    try:
        # Swedish format uses comma as decimal separator
        if isinstance(value, str):
            value = value.replace(",", ".")
        return float(value)
    except (ValueError, TypeError):
        return None


@dataclass
class SwedishBuilding:
    """Parsed Swedish building from GeoJSON."""

    # Identification
    building_id: int
    uuid: Optional[str] = None

    # Location
    address: str = ""
    postal_code: str = ""
    city: str = ""
    municipality: str = ""
    county: str = ""
    deso: str = ""  # DeSO area code

    # Geometry (SWEREF99 TM - EPSG:3006)
    footprint_coords: List[List[Tuple[float, float, float]]] = field(default_factory=list)
    footprint_area_m2: float = 0.0
    height_m: float = 0.0

    # Building characteristics
    building_type: str = ""  # Friliggande, Kedjehus, Radhus, etc.
    building_category: str = ""  # Småhus, Flerbostadshus, etc.
    construction_year: Optional[int] = None
    atemp_m2: float = 0.0
    num_apartments: Optional[int] = None
    num_floors: Optional[int] = None

    # Use-type breakdown (percentages of Atemp, 0-100)
    # Critical for multi-zone modeling - different zones have different ventilation!
    atemp_residential_pct: float = 100.0  # Bostad - residential
    atemp_retail_pct: float = 0.0  # Butik - retail/shops
    atemp_restaurant_pct: float = 0.0  # Restaurang - restaurant/kitchen
    atemp_office_pct: float = 0.0  # Kontor - office
    atemp_hotel_pct: float = 0.0  # Hotell - hotel
    atemp_school_pct: float = 0.0  # Skolor - schools
    atemp_healthcare_pct: float = 0.0  # Vård - healthcare
    atemp_grocery_pct: float = 0.0  # Livsmedel - grocery store
    atemp_theater_pct: float = 0.0  # Teater - theater/cinema
    atemp_pool_pct: float = 0.0  # Bad - swimming pool
    atemp_other_pct: float = 0.0  # Övrig - other
    atemp_healthcare_day_pct: float = 0.0  # VårdDag - day care

    # Ventilation details (from declaration)
    has_ftx: bool = False  # FTX system present
    has_f_only: bool = False  # F-only (exhaust) present
    has_ft: bool = False  # FT (supply+exhaust, no HR) present
    has_natural_vent: bool = False  # Självdrag - natural ventilation
    design_airflow_l_s_m2: Optional[float] = None  # Projekterat ventilationsflöde

    # Energy declaration
    energy_class: str = ""  # A, B, C, D, E, F, G
    energy_performance_kwh_m2: Optional[float] = None

    # Heating sources (kWh) - from UPPV fields
    district_heating_kwh: float = 0.0
    ground_source_hp_kwh: float = 0.0
    exhaust_air_hp_kwh: float = 0.0
    air_water_hp_kwh: float = 0.0
    air_air_hp_kwh: float = 0.0
    electric_direct_kwh: float = 0.0
    oil_kwh: float = 0.0
    gas_kwh: float = 0.0
    wood_kwh: float = 0.0
    pellets_kwh: float = 0.0
    biofuel_kwh: float = 0.0

    # Electricity (kWh)
    property_electricity_kwh: float = 0.0  # Fastighetsel
    hot_water_electricity_kwh: float = 0.0  # El för varmvatten

    # Cooling (kWh)
    district_cooling_kwh: float = 0.0

    # Ventilation
    ventilation_type: str = ""  # F, FT, FTX, Självdrag

    # Solar
    has_solar_pv: bool = False
    has_solar_thermal: bool = False
    solar_pv_kwh: float = 0.0
    solar_thermal_kwh: float = 0.0
    solar_production_kwh: float = 0.0  # Beräknad elproduktion

    # Energy totals
    total_energy_kwh: float = 0.0
    primary_energy_kwh: float = 0.0

    # Ownership
    owner_type: str = ""  # Privatägd, Bostadsrätt, etc.

    # Raw properties (for additional access)
    raw_properties: Dict[str, Any] = field(default_factory=dict)

    def get_centroid_sweref(self) -> Tuple[float, float]:
        """Get centroid in SWEREF99 TM coordinates."""
        if not self.footprint_coords:
            return (0.0, 0.0)

        # Get first ring of first polygon
        ring = self.footprint_coords[0] if self.footprint_coords else []
        if not ring:
            return (0.0, 0.0)

        x_sum = sum(p[0] for p in ring)
        y_sum = sum(p[1] for p in ring)
        n = len(ring)

        return (x_sum / n, y_sum / n)

    def get_centroid_wgs84(self) -> Tuple[float, float]:
        """Get centroid in WGS84 (lat, lon)."""
        x, y = self.get_centroid_sweref()
        return sweref99_to_wgs84(x, y)

    def get_primary_heating(self) -> str:
        """Determine primary heating source."""
        sources = {
            "district_heating": self.district_heating_kwh,
            "ground_source_hp": self.ground_source_hp_kwh,
            "exhaust_air_hp": self.exhaust_air_hp_kwh,
            "air_water_hp": self.air_water_hp_kwh,
            "air_air_hp": self.air_air_hp_kwh,
            "electric_direct": self.electric_direct_kwh,
            "oil": self.oil_kwh,
            "gas": self.gas_kwh,
            "wood": self.wood_kwh,
            "pellets": self.pellets_kwh,
        }

        if not any(sources.values()):
            return "unknown"

        return max(sources, key=sources.get)

    def is_mixed_use(self, threshold_pct: float = 5.0) -> bool:
        """Check if building has significant non-residential use."""
        non_residential = (
            self.atemp_retail_pct + self.atemp_restaurant_pct +
            self.atemp_office_pct + self.atemp_hotel_pct +
            self.atemp_school_pct + self.atemp_healthcare_pct +
            self.atemp_grocery_pct + self.atemp_theater_pct +
            self.atemp_pool_pct + self.atemp_other_pct +
            self.atemp_healthcare_day_pct
        )
        return non_residential >= threshold_pct

    def get_zone_breakdown(self) -> Dict[str, float]:
        """
        Get zone breakdown for multi-zone modeling.

        Returns dict of zone_type -> fraction (0.0-1.0).
        Only includes zones with >0% area.
        """
        zones = {}

        # Map percentage fields to zone types
        zone_map = {
            'residential': self.atemp_residential_pct,
            'retail': self.atemp_retail_pct,
            'restaurant': self.atemp_restaurant_pct,
            'office': self.atemp_office_pct,
            'hotel': self.atemp_hotel_pct,
            'school': self.atemp_school_pct,
            'healthcare': self.atemp_healthcare_pct,
            'grocery': self.atemp_grocery_pct,
            'theater': self.atemp_theater_pct,
            'pool': self.atemp_pool_pct,
            'other': self.atemp_other_pct,
            'daycare': self.atemp_healthcare_day_pct,
        }

        for zone_type, pct in zone_map.items():
            if pct > 0:
                zones[zone_type] = pct / 100.0

        # If no zones defined (all zeros), assume 100% residential
        if not zones:
            zones['residential'] = 1.0

        return zones

    def get_commercial_fraction(self) -> float:
        """Get fraction of building that is commercial (non-residential)."""
        return 1.0 - (self.atemp_residential_pct / 100.0)

    def has_high_ventilation_zones(self) -> bool:
        """Check if building has zones requiring high ventilation (restaurant, pool, grocery)."""
        return (
            self.atemp_restaurant_pct > 0 or
            self.atemp_pool_pct > 0 or
            self.atemp_grocery_pct > 0
        )

    def get_effective_ventilation_params(self) -> Dict[str, float]:
        """
        Calculate effective ventilation parameters for the whole building.

        Returns weighted-average airflow and heat recovery based on zone mix.
        Critical for calibration and energy modeling.
        """
        from .zone_configs import ZONE_CONFIGS

        zones = self.get_zone_breakdown()

        # Calculate weighted averages
        total_airflow_weighted = 0.0
        total_hr_weighted = 0.0
        total_heat_loss_factor = 0.0  # For proper HR weighting

        for zone_type, fraction in zones.items():
            config = ZONE_CONFIGS.get(zone_type, ZONE_CONFIGS['residential'])
            airflow = config['airflow_l_s_m2']
            hr = config['heat_recovery_eff']

            # Weight airflow by area fraction
            total_airflow_weighted += airflow * fraction

            # HR must be weighted by HEAT LOSS, not area
            # Heat loss ∝ airflow × (1 - HR)
            heat_loss = airflow * (1.0 - hr)
            total_heat_loss_factor += heat_loss * fraction
            total_hr_weighted += airflow * hr * fraction

        # Effective HR = 1 - (weighted_heat_loss / weighted_airflow)
        if total_airflow_weighted > 0:
            effective_hr = 1.0 - (total_heat_loss_factor / total_airflow_weighted)
        else:
            effective_hr = 0.0

        return {
            'effective_airflow_l_s_m2': total_airflow_weighted,
            'effective_heat_recovery': max(0.0, min(1.0, effective_hr)),
            'heat_loss_factor': total_heat_loss_factor,
            'zones': zones,
        }


def sweref99_to_wgs84(x: float, y: float) -> Tuple[float, float]:
    """
    Convert SWEREF99 TM (EPSG:3006) to WGS84 (lat, lon).

    Approximate conversion using the TM projection formulas.
    For high precision, use pyproj.
    """
    # SWEREF99 TM parameters
    lat0 = 0.0  # Origin latitude
    lon0 = 15.0  # Central meridian
    k0 = 0.9996  # Scale factor
    x0 = 500000  # False easting
    y0 = 0  # False northing

    # WGS84 ellipsoid
    a = 6378137.0  # Semi-major axis
    f = 1 / 298.257222101  # Flattening
    e2 = 2 * f - f * f  # Eccentricity squared

    # Remove false easting/northing
    x = x - x0
    y = y - y0

    # Footpoint latitude
    M = y / k0
    mu = M / (a * (1 - e2 / 4 - 3 * e2 * e2 / 64))

    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))

    phi1 = mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * math.sin(2 * mu)
    phi1 += (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * math.sin(4 * mu)
    phi1 += (151 * e1 ** 3 / 96) * math.sin(6 * mu)

    # Calculate latitude
    N1 = a / math.sqrt(1 - e2 * math.sin(phi1) ** 2)
    T1 = math.tan(phi1) ** 2
    C1 = e2 * math.cos(phi1) ** 2 / (1 - e2)
    R1 = a * (1 - e2) / (1 - e2 * math.sin(phi1) ** 2) ** 1.5
    D = x / (N1 * k0)

    lat = phi1 - (N1 * math.tan(phi1) / R1) * (
        D ** 2 / 2 - (5 + 3 * T1 + 10 * C1 - 4 * C1 ** 2 - 9 * e2 / (1 - e2)) * D ** 4 / 24
    )

    lon = math.radians(lon0) + (
        D - (1 + 2 * T1 + C1) * D ** 3 / 6
    ) / math.cos(phi1)

    return (math.degrees(lat), math.degrees(lon))


class SwedenBuildingsLoader:
    """
    Load and query Swedish buildings from GeoJSON.

    Provides spatial search and filtering capabilities.
    """

    def __init__(self, geojson_path: Path | None = None):
        self.geojson_path = geojson_path or DEFAULT_GEOJSON_PATH
        self._buildings: List[SwedishBuilding] = []
        self._spatial_index: Dict[str, List[int]] = {}  # Grid-based index
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Lazy load the GeoJSON data."""
        if self._loaded:
            return

        if not self.geojson_path.exists():
            console.print(f"[red]Swedish buildings file not found: {self.geojson_path}[/red]")
            return

        console.print(f"[cyan]Loading Swedish buildings from {self.geojson_path.name}...[/cyan]")

        with open(self.geojson_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        features = data.get("features", [])
        console.print(f"[dim]Parsing {len(features)} buildings...[/dim]")

        for i, feature in enumerate(features):
            building = self._parse_feature(feature)
            self._buildings.append(building)

            # Build spatial index (10km grid)
            centroid = building.get_centroid_sweref()
            grid_key = f"{int(centroid[0] // 10000)}_{int(centroid[1] // 10000)}"
            if grid_key not in self._spatial_index:
                self._spatial_index[grid_key] = []
            self._spatial_index[grid_key].append(i)

        self._loaded = True
        console.print(f"[green]Loaded {len(self._buildings)} buildings[/green]")

    def _parse_feature(self, feature: Dict[str, Any]) -> SwedishBuilding:
        """Parse a GeoJSON feature into SwedishBuilding."""
        props = feature.get("properties", {})
        geom = feature.get("geometry", {})

        # Extract coordinates
        coords = []
        if geom.get("type") == "GeometryCollection":
            for g in geom.get("geometries", []):
                if g.get("type") == "MultiPolygon":
                    for polygon in g.get("coordinates", []):
                        if polygon:
                            coords.extend(polygon)
        elif geom.get("type") == "MultiPolygon":
            for polygon in geom.get("coordinates", []):
                if polygon:
                    coords.extend(polygon)
        elif geom.get("type") == "Polygon":
            coords = geom.get("coordinates", [])

        # Determine ventilation type
        vent_type = ""
        if props.get("VentTypFTX") == "Ja":
            vent_type = "FTX"
        elif props.get("VentTypFT") == "Ja":
            vent_type = "FT"
        elif props.get("VentTypF") == "Ja" or props.get("VentTypFmed") == "Ja":
            vent_type = "F"
        elif props.get("VentTypSjalvdrag") == "Ja":
            vent_type = "S"

        return SwedishBuilding(
            building_id=props.get("byggnadsid", 0),
            uuid=props.get("50A_UUID"),

            # Location
            address=props.get("IdAdr", ""),
            postal_code=str(props.get("IdPostnr", "")),
            city=props.get("IdPostort", ""),
            municipality=props.get("IdKommun", ""),
            county=props.get("IdLan", ""),
            deso=props.get("Deso", ""),

            # Geometry
            footprint_coords=coords,
            footprint_area_m2=props.get("Footprint_area_Sweref", 0.0),
            height_m=props.get("height", 0.0),

            # Building characteristics
            building_type=props.get("EgenByggnadsTyp", ""),
            building_category=props.get("EgenByggnadsKat", "") or props.get("50A_ByggnadsKategori", ""),
            construction_year=props.get("EgenNybyggAr"),
            atemp_m2=props.get("EgenAtemp", 0.0),
            num_apartments=props.get("EgenAntalBolgh"),
            num_floors=props.get("EgenAntalPlan"),

            # Use-type breakdown (percentages)
            # Note: These are stored as floats (0-100) or empty strings in GeoJSON
            atemp_residential_pct=_parse_pct(props.get("EgenAtempBostad", 100.0)),
            atemp_retail_pct=_parse_pct(props.get("EgenAtempButik", 0)),
            atemp_restaurant_pct=_parse_pct(props.get("EgenAtempRestaurang", 0)),
            atemp_office_pct=_parse_pct(props.get("EgenAtempKontor", 0)),
            atemp_hotel_pct=_parse_pct(props.get("EgenAtempHotell", 0)),
            atemp_school_pct=_parse_pct(props.get("EgenAtempSkolor", 0)),
            atemp_healthcare_pct=_parse_pct(props.get("EgenAtempVard", 0)),
            atemp_grocery_pct=_parse_pct(props.get("EgenAtempLivsmedel", 0)),
            atemp_theater_pct=_parse_pct(props.get("EgenAtempTeater", 0)),
            atemp_pool_pct=_parse_pct(props.get("EgenAtempBad", 0)),
            atemp_other_pct=_parse_pct(props.get("EgenAtempOvrig", 0)),
            atemp_healthcare_day_pct=_parse_pct(props.get("EgenAtempVardDag", 0)),

            # Ventilation details
            has_ftx=props.get("VentTypFTX") == "Ja",
            has_f_only=props.get("VentTypF") == "Ja" or props.get("VentTypFmed") == "Ja",
            has_ft=props.get("VentTypFT") == "Ja",
            has_natural_vent=props.get("VentTypSjalvdrag") == "Ja",
            design_airflow_l_s_m2=_parse_airflow(props.get("EgenProjVentFlode")),

            # Energy
            energy_class=props.get("EgiEnergiklass", ""),
            energy_performance_kwh_m2=props.get("EgiEnergiPrestanda"),

            # Heating - use UPPV fields (uppvärmning = heating)
            # Falls back to non-UPPV if UPPV is 0
            district_heating_kwh=float(props.get("EgiFjarrvarmeUPPV") or props.get("EgiFjarrvarme") or 0),
            ground_source_hp_kwh=float(props.get("EgiPumpMarkUPPV") or props.get("EgiPumpMark") or 0),
            exhaust_air_hp_kwh=float(props.get("EgiPumpFranluftUPPV") or props.get("EgiPumpFranluft") or 0),
            air_water_hp_kwh=float(props.get("EgiPumpLuftVattenUPPV") or props.get("EgiPumpLuftVatten") or 0),
            air_air_hp_kwh=float(props.get("EgiPumpLuftLuftUPPV") or props.get("EgiPumpLuftLuft") or 0),
            electric_direct_kwh=float(props.get("EgiElDirektUPPV") or props.get("EgiElDirekt") or 0),
            oil_kwh=float(props.get("EgiOljaUPPV") or props.get("EgiOlja") or 0),
            gas_kwh=float(props.get("EgiGasUPPV") or props.get("EgiGas") or 0),
            wood_kwh=float(props.get("EgiVedUPPV") or props.get("EgiVed") or 0),
            pellets_kwh=float(props.get("EgiFlisUPPV") or props.get("EgiFlis") or 0),
            biofuel_kwh=float(props.get("EgiOvrBiobransleUPPV") or props.get("EgiOvrBiobransle") or 0),

            # Electricity
            property_electricity_kwh=float(props.get("EgiFastighet") or 0),
            hot_water_electricity_kwh=float(props.get("EgiElVV") or 0),

            # Cooling
            district_cooling_kwh=float(props.get("EgiFjarrkyla") or 0),

            # Ventilation
            ventilation_type=vent_type,

            # Solar
            has_solar_pv=props.get("EgiGruppSolcell") == "Ja",
            has_solar_thermal=props.get("EgiGruppSolvarme") == "Ja",
            solar_pv_kwh=float(props.get("EgiSolcell") or 0),
            solar_thermal_kwh=float(props.get("EgiSolvarme") or 0),
            solar_production_kwh=float(props.get("EgiBerElProduktion") or 0),

            # Energy totals
            total_energy_kwh=float(props.get("EgiEnergianvandning") or 0),
            primary_energy_kwh=float(props.get("EgiPrimarenergianvandning") or 0),

            # Ownership
            owner_type=props.get("42P_ByggnadsAgare", ""),

            raw_properties=props,
        )

    def get_all_buildings(self) -> List[SwedishBuilding]:
        """Get all buildings."""
        self._ensure_loaded()
        return self._buildings

    def find_by_address(self, address: str) -> List[SwedishBuilding]:
        """
        Find buildings matching an address with smart matching.

        Priority:
        1. Exact match (street + house number)
        2. Partial match on street name

        Handles cases like:
        - "Rusthållarvägen 2" should match "Rusthållarvägen 2" not "Rusthållarvägen 25"
        - "Aktergatan 11, Stockholm" should match "Aktergatan 11"
        """
        import re
        self._ensure_loaded()

        # Clean and parse the search address
        # Remove city suffix: "Aktergatan 11, Stockholm" -> "Aktergatan 11"
        address_clean = address.split(",")[0].strip()
        address_lower = address_clean.lower()

        # Extract street name and house number
        # Pattern: "Street Name 123" or "Street Name 123A"
        match = re.match(r'^(.+?)\s+(\d+\w*)$', address_clean, re.IGNORECASE)

        if match:
            search_street = match.group(1).lower().strip()
            search_number = match.group(2).lower().strip()

            # First try exact match: street + house number
            exact_matches = []
            partial_matches = []

            for building in self._buildings:
                b_match = re.match(r'^(.+?)\s+(\d+\w*)$', building.address, re.IGNORECASE)
                if b_match:
                    b_street = b_match.group(1).lower().strip()
                    b_number = b_match.group(2).lower().strip()

                    # Exact street + number match
                    if b_street == search_street and b_number == search_number:
                        exact_matches.append(building)
                    # Same street, different number (for partial fallback)
                    elif b_street == search_street:
                        partial_matches.append(building)
                    # Partial street match (e.g., "aktergatan" in "stora aktergatan")
                    elif search_street in b_street or b_street in search_street:
                        if b_number == search_number:
                            exact_matches.append(building)
                        else:
                            partial_matches.append(building)

            if exact_matches:
                return exact_matches
            if partial_matches:
                return partial_matches

        # Fallback: original partial matching logic
        matches = []
        for building in self._buildings:
            if address_lower in building.address.lower():
                matches.append(building)
            elif address_lower in f"{building.address} {building.city}".lower():
                matches.append(building)

        return matches

    def find_by_location(
        self,
        lat: float,
        lon: float,
        radius_m: float = 100,
    ) -> List[SwedishBuilding]:
        """Find buildings near a WGS84 location."""
        self._ensure_loaded()

        # Convert to SWEREF99
        # Simple approximation (reverse of sweref99_to_wgs84)
        x = 500000 + (lon - 15) * 111320 * math.cos(math.radians(lat))
        y = lat * 111320

        # Check nearby grid cells
        grid_x = int(x // 10000)
        grid_y = int(y // 10000)

        candidates = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                key = f"{grid_x + dx}_{grid_y + dy}"
                if key in self._spatial_index:
                    candidates.extend(self._spatial_index[key])

        # Filter by distance
        matches = []
        for idx in set(candidates):
            building = self._buildings[idx]
            bx, by = building.get_centroid_sweref()
            dist = math.sqrt((bx - x) ** 2 + (by - y) ** 2)
            if dist <= radius_m:
                matches.append((building, dist))

        # Sort by distance
        matches.sort(key=lambda x: x[1])
        return [m[0] for m in matches]

    def find_by_municipality(self, municipality: str) -> List[SwedishBuilding]:
        """Find all buildings in a municipality."""
        self._ensure_loaded()
        muni_lower = municipality.lower()
        return [b for b in self._buildings if muni_lower in b.municipality.lower()]

    def find_by_energy_class(self, energy_class: str) -> List[SwedishBuilding]:
        """Find buildings with a specific energy class."""
        self._ensure_loaded()
        return [b for b in self._buildings if b.energy_class == energy_class.upper()]

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the loaded buildings."""
        self._ensure_loaded()

        # Count by energy class
        by_class = {}
        for b in self._buildings:
            cls = b.energy_class or "Unknown"
            by_class[cls] = by_class.get(cls, 0) + 1

        # Count by building type
        by_type = {}
        for b in self._buildings:
            typ = b.building_category or "Unknown"
            by_type[typ] = by_type.get(typ, 0) + 1

        # Count by ventilation
        by_vent = {}
        for b in self._buildings:
            vent = b.ventilation_type or "Unknown"
            by_vent[vent] = by_vent.get(vent, 0) + 1

        return {
            "total_buildings": len(self._buildings),
            "by_energy_class": by_class,
            "by_building_type": by_type,
            "by_ventilation": by_vent,
            "with_solar_pv": sum(1 for b in self._buildings if b.has_solar_pv),
            "with_heat_pump": sum(1 for b in self._buildings if b.ground_source_hp_kwh > 0 or b.exhaust_air_hp_kwh > 0),
        }


# Convenience function
def load_sweden_buildings(geojson_path: Path | None = None) -> SwedenBuildingsLoader:
    """Load the Swedish buildings dataset."""
    return SwedenBuildingsLoader(geojson_path)


def find_building_by_address(address: str) -> List[SwedishBuilding]:
    """Find buildings by address."""
    loader = SwedenBuildingsLoader()
    return loader.find_by_address(address)
