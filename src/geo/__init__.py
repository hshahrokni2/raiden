"""
Raiden Geo Module

Building footprint resolution from existing data sources.

Key features:
- OSM point-in-polygon lookup
- Courtyard detection (returns all surrounding buildings)
- Address-based slicing for multi-building BRFs
- Smart boundary estimation for shared buildings
"""

from .footprint_resolver import (
    FootprintResolver,
    ResolvedFootprint,
    resolve_footprint,
    resolve_brf_footprints,
    resolve_shared_building,
)

__all__ = [
    "FootprintResolver",
    "ResolvedFootprint",
    "resolve_footprint",
    "resolve_brf_footprints",
    "resolve_shared_building",
]

