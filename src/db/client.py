"""
Supabase client for Raiden.
"""

import os
from typing import Optional
from functools import lru_cache

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    Client = None

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class SupabaseClient:
    """Wrapper for Supabase client with lazy initialization."""

    _instance: Optional["SupabaseClient"] = None
    _client: Optional[Client] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not SUPABASE_AVAILABLE:
            raise ImportError(
                "supabase-py not installed. Install with: pip install supabase"
            )

    @property
    def client(self) -> Client:
        """Get or create Supabase client."""
        if self._client is None:
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_KEY")

            if not url or not key:
                raise ValueError(
                    "SUPABASE_URL and SUPABASE_KEY must be set in environment"
                )

            self._client = create_client(url, key)
        return self._client

    # ============================================
    # Buildings
    # ============================================

    def insert_building(self, data: dict) -> dict:
        """Insert a new building record."""
        result = self.client.table("buildings").insert(data).execute()
        return result.data[0] if result.data else None

    def get_building(self, building_id: str) -> Optional[dict]:
        """Get building by ID."""
        result = (
            self.client.table("buildings")
            .select("*")
            .eq("id", building_id)
            .execute()
        )
        return result.data[0] if result.data else None

    def get_building_by_address(self, address: str) -> Optional[dict]:
        """Get building by address."""
        result = (
            self.client.table("buildings")
            .select("*")
            .ilike("address", f"%{address}%")
            .execute()
        )
        return result.data[0] if result.data else None

    def list_buildings(
        self,
        region: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list:
        """List buildings with optional filtering."""
        query = self.client.table("buildings").select("*")

        if region:
            query = query.eq("region", region)

        result = query.range(offset, offset + limit - 1).execute()
        return result.data

    def update_building(self, building_id: str, data: dict) -> dict:
        """Update building record."""
        result = (
            self.client.table("buildings")
            .update(data)
            .eq("id", building_id)
            .execute()
        )
        return result.data[0] if result.data else None

    def delete_building(self, building_id: str) -> bool:
        """Delete building and all related records."""
        self.client.table("buildings").delete().eq("id", building_id).execute()
        return True

    # ============================================
    # Baseline Simulations
    # ============================================

    def insert_baseline(self, data: dict) -> dict:
        """Insert baseline simulation result."""
        result = self.client.table("baseline_simulations").insert(data).execute()
        return result.data[0] if result.data else None

    def get_baseline(self, building_id: str) -> Optional[dict]:
        """Get latest baseline for building."""
        result = (
            self.client.table("baseline_simulations")
            .select("*")
            .eq("building_id", building_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None

    # ============================================
    # ECM Results
    # ============================================

    def insert_ecm_result(self, data: dict) -> dict:
        """Insert ECM analysis result."""
        result = self.client.table("ecm_results").insert(data).execute()
        return result.data[0] if result.data else None

    def insert_ecm_results_batch(self, results: list) -> list:
        """Insert multiple ECM results."""
        result = self.client.table("ecm_results").insert(results).execute()
        return result.data

    def get_ecm_results(
        self,
        building_id: str,
        applicable_only: bool = True,
    ) -> list:
        """Get all ECM results for a building."""
        query = (
            self.client.table("ecm_results")
            .select("*")
            .eq("building_id", building_id)
        )

        if applicable_only:
            query = query.eq("is_applicable", True)

        result = query.order("simple_payback_years").execute()
        return result.data

    def get_top_ecms(
        self,
        building_id: str,
        limit: int = 10,
        order_by: str = "simple_payback_years",
    ) -> list:
        """Get top ECMs by payback or savings."""
        result = (
            self.client.table("ecm_results")
            .select("*")
            .eq("building_id", building_id)
            .eq("is_applicable", True)
            .order(order_by)
            .limit(limit)
            .execute()
        )
        return result.data

    # ============================================
    # Packages
    # ============================================

    def insert_package(self, data: dict) -> dict:
        """Insert ECM package."""
        result = self.client.table("ecm_packages").insert(data).execute()
        return result.data[0] if result.data else None

    def get_packages(self, building_id: str) -> list:
        """Get all packages for a building."""
        result = (
            self.client.table("ecm_packages")
            .select("*")
            .eq("building_id", building_id)
            .order("simple_payback_years")
            .execute()
        )
        return result.data

    # ============================================
    # Reports
    # ============================================

    def insert_report(self, data: dict) -> dict:
        """Insert analysis report."""
        result = self.client.table("analysis_reports").insert(data).execute()
        return result.data[0] if result.data else None

    def get_latest_report(self, building_id: str) -> Optional[dict]:
        """Get latest report for building."""
        result = (
            self.client.table("analysis_reports")
            .select("*")
            .eq("building_id", building_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None

    # ============================================
    # Views and Aggregations
    # ============================================

    def get_building_summary(self, building_id: str) -> Optional[dict]:
        """Get building with analysis summary."""
        result = (
            self.client.table("building_summary")
            .select("*")
            .eq("id", building_id)
            .execute()
        )
        return result.data[0] if result.data else None

    def get_top_ecms_global(self, limit: int = 20) -> list:
        """Get top performing ECMs across all buildings."""
        result = (
            self.client.table("top_ecms")
            .select("*")
            .limit(limit)
            .execute()
        )
        return result.data


@lru_cache(maxsize=1)
def get_client() -> SupabaseClient:
    """Get singleton Supabase client instance."""
    return SupabaseClient()


def check_connection() -> bool:
    """Check if Supabase connection works."""
    try:
        client = get_client()
        # Try a simple query
        client.client.table("buildings").select("id").limit(1).execute()
        return True
    except Exception as e:
        print(f"Supabase connection error: {e}")
        return False
