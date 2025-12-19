"""Data ingestion and fetching modules."""

from .brf_parser import BRFParser
from .osm_fetcher import OSMFetcher
from .overture_fetcher import OvertureFetcher
from .microsoft_buildings import (
    MicrosoftBuildingsFetcher,
    get_microsoft_buildings,
    lat_lon_to_quadkey,
)
from .energidek_parser import EnergyDeclarationParser, parse_energy_declaration
from .image_fetcher import (
    FacadeImage,
    FacadeImageFetcher,
    MapillaryFetcher,
    KartaViewFetcher,
    WikimediaCommonsFetcher,
    ManualImageLoader,
    fetch_facade_images,
)
from .building_extractor import (
    BuildingDataExtractor,
    BuildingProfile,
    GeometryData,
    EnvelopeData,
    HVACData,
    DataSource,
    extract_building,
)
from .sweden_buildings import (
    SwedenBuildingsLoader,
    SwedishBuilding,
    load_sweden_buildings,
    find_building_by_address,
    sweref99_to_wgs84,
)

__all__ = [
    # Parsers
    "BRFParser",
    "EnergyDeclarationParser",
    "parse_energy_declaration",
    # Fetchers
    "OSMFetcher",
    "OvertureFetcher",
    "MicrosoftBuildingsFetcher",
    "get_microsoft_buildings",
    "lat_lon_to_quadkey",
    "FacadeImageFetcher",
    "MapillaryFetcher",
    "KartaViewFetcher",
    "WikimediaCommonsFetcher",
    "ManualImageLoader",
    # Image types
    "FacadeImage",
    "fetch_facade_images",
    # Unified extraction
    "BuildingDataExtractor",
    "BuildingProfile",
    "GeometryData",
    "EnvelopeData",
    "HVACData",
    "DataSource",
    "extract_building",
    # Swedish buildings GeoJSON
    "SwedenBuildingsLoader",
    "SwedishBuilding",
    "load_sweden_buildings",
    "find_building_by_address",
    "sweref99_to_wgs84",
]
