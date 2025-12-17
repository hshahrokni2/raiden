"""Data ingestion and fetching modules."""

from .brf_parser import BRFParser
from .osm_fetcher import OSMFetcher
from .overture_fetcher import OvertureFetcher
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

__all__ = [
    "BRFParser",
    "OSMFetcher",
    "OvertureFetcher",
    "EnergyDeclarationParser",
    "parse_energy_declaration",
    "FacadeImage",
    "FacadeImageFetcher",
    "MapillaryFetcher",
    "KartaViewFetcher",
    "ManualImageLoader",
    "fetch_facade_images",
]
