"""
Database module for Raiden ECM Analysis.

Provides Supabase integration for storing and retrieving analysis results.
"""

from .client import get_client, SupabaseClient
from .models import (
    BuildingRecord,
    BaselineRecord,
    ECMResultRecord,
    PackageRecord,
)
from .repository import (
    BuildingRepository,
    ECMResultRepository,
    PackageRepository,
)

__all__ = [
    "get_client",
    "SupabaseClient",
    "BuildingRecord",
    "BaselineRecord",
    "ECMResultRecord",
    "PackageRecord",
    "BuildingRepository",
    "ECMResultRepository",
    "PackageRepository",
]
