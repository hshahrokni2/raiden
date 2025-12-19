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

    # Energy declaration
    energy_class: str = ""  # A, B, C, D, E, F, G
    energy_performance_kwh_m2: Optional[float] = None

    # Heating (kWh)
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

    # Ventilation
    ventilation_type: str = ""  # F, FT, FTX, Självdrag

    # Solar
    has_solar_pv: bool = False
    has_solar_thermal: bool = False
    solar_pv_kwh: float = 0.0
    solar_thermal_kwh: float = 0.0

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

            # Energy
            energy_class=props.get("EgiEnergiklass", ""),
            energy_performance_kwh_m2=props.get("EgiEnergiPrestanda"),

            # Heating
            district_heating_kwh=props.get("EgiFjarrvarme", 0.0) or 0.0,
            ground_source_hp_kwh=props.get("EgiPumpMark", 0.0) or 0.0,
            exhaust_air_hp_kwh=props.get("EgiPumpFranluft", 0.0) or 0.0,
            air_water_hp_kwh=props.get("EgiPumpLuftVatten", 0.0) or 0.0,
            air_air_hp_kwh=props.get("EgiPumpLuftLuft", 0.0) or 0.0,
            electric_direct_kwh=props.get("EgiElDirekt", 0.0) or 0.0,
            oil_kwh=props.get("EgiOlja", 0.0) or 0.0,
            gas_kwh=props.get("EgiGas", 0.0) or 0.0,
            wood_kwh=props.get("EgiVed", 0.0) or 0.0,
            pellets_kwh=props.get("EgiFlis", 0.0) or 0.0,

            # Ventilation
            ventilation_type=vent_type,

            # Solar
            has_solar_pv=props.get("EgiGruppSolcell") == "Ja",
            has_solar_thermal=props.get("EgiGruppSolvarme") == "Ja",
            solar_pv_kwh=float(props.get("EgiSolcell") or 0),
            solar_thermal_kwh=float(props.get("EgiSolvarme") or 0),

            # Ownership
            owner_type=props.get("42P_ByggnadsAgare", ""),

            raw_properties=props,
        )

    def get_all_buildings(self) -> List[SwedishBuilding]:
        """Get all buildings."""
        self._ensure_loaded()
        return self._buildings

    def find_by_address(self, address: str) -> List[SwedishBuilding]:
        """Find buildings matching an address (partial match)."""
        self._ensure_loaded()
        address_lower = address.lower()

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
