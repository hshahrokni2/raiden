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

    # ============================================
    # Full Analysis Storage
    # ============================================

    def store_analysis(self, data: dict) -> Optional[str]:
        """
        Store complete analysis results from FullPipelineAnalyzer.

        Handles:
        - Building record (upsert by address)
        - Baseline simulation results
        - ECM analysis results (batch)
        - ECM packages
        - Analysis report metadata

        Args:
            data: Dict with keys:
                - address, construction_year, atemp_m2, energy_class, etc.
                - baseline_kwh_m2, calibration_gap
                - ecm_results: List of ECM dicts
                - packages: List of package dicts
                - data_sources, confidence

        Returns:
            Building ID (UUID) or None on error
        """
        import logging
        logger = logging.getLogger(__name__)

        try:
            # 1. Upsert building record
            building_data = {
                "address": data.get("address"),
                "construction_year": data.get("construction_year"),
                "heated_area_m2": data.get("atemp_m2"),
                "energy_class": data.get("energy_class"),
                "heating_system": data.get("heating_system"),
                "ventilation_system": data.get("ventilation_system"),
                "facade_material": data.get("facade_material"),
                "num_floors": data.get("num_floors"),
                "num_apartments": data.get("num_apartments"),
                "declared_energy_kwh_m2": data.get("declared_energy_kwh_m2"),
                "data_sources": data.get("data_sources"),
                "data_confidence": data.get("confidence"),
            }
            # Remove None values
            building_data = {k: v for k, v in building_data.items() if v is not None}

            # Check if building exists
            existing = self.get_building_by_address(data.get("address", ""))
            if existing:
                building_id = existing["id"]
                self.update_building(building_id, building_data)
                logger.info(f"Updated existing building: {building_id}")
            else:
                result = self.insert_building(building_data)
                building_id = result["id"] if result else None
                logger.info(f"Created new building: {building_id}")

            if not building_id:
                return None

            # 2. Store baseline simulation
            baseline_data = {
                "building_id": building_id,
                "archetype_id": data.get("archetype_id"),
                "heating_kwh_m2": data.get("baseline_kwh_m2"),
                "calibration_gap": data.get("calibration_gap"),
                "calibrated_infiltration": data.get("calibrated_infiltration"),
                "calibrated_heat_recovery": data.get("calibrated_heat_recovery"),
                "calibrated_window_u": data.get("calibrated_window_u"),
            }
            baseline_data = {k: v for k, v in baseline_data.items() if v is not None}
            if len(baseline_data) > 1:  # More than just building_id
                self.insert_baseline(baseline_data)

            # 3. Store ECM results (batch)
            ecm_results = data.get("ecm_results", [])
            if ecm_results:
                ecm_records = []
                for ecm in ecm_results:
                    ecm_record = {
                        "building_id": building_id,
                        "ecm_id": ecm.get("ecm_id"),
                        "ecm_name": ecm.get("ecm_name"),
                        "ecm_category": ecm.get("category"),  # Schema uses ecm_category
                        "is_applicable": ecm.get("is_applicable", True),
                        "baseline_kwh_m2": ecm.get("baseline_kwh_m2"),
                        "heating_kwh_m2": ecm.get("heating_kwh_m2"),  # Primary column in schema
                        "result_kwh_m2": ecm.get("result_kwh_m2") or ecm.get("heating_kwh_m2"),
                        "savings_kwh_m2": ecm.get("savings_kwh_m2"),
                        "savings_percent": ecm.get("savings_percent"),
                        "heating_savings_percent": ecm.get("savings_percent"),  # Schema column
                        "investment_sek": ecm.get("investment_sek"),
                        "net_cost": ecm.get("investment_sek"),  # Schema uses net_cost
                        "annual_savings_sek": ecm.get("annual_savings_sek"),
                        "simple_payback_years": ecm.get("simple_payback_years"),
                        "simulated": ecm.get("simulated", False),
                    }
                    ecm_record = {k: v for k, v in ecm_record.items() if v is not None}
                    ecm_records.append(ecm_record)

                if ecm_records:
                    self.insert_ecm_results_batch(ecm_records)
                    logger.info(f"Stored {len(ecm_records)} ECM results")

            # 4. Store packages
            packages = data.get("packages", [])
            for pkg in packages:
                package_data = {
                    "building_id": building_id,
                    "package_name": pkg.get("name") or pkg.get("package_name"),
                    "package_type": pkg.get("package_type"),  # basic, standard, premium
                    "ecm_ids": pkg.get("ecm_ids"),
                    "combined_heating_kwh_m2": pkg.get("combined_kwh_m2"),  # Schema column
                    "combined_savings_kwh_m2": pkg.get("combined_kwh_m2"),  # Alias
                    "combined_savings_percent": pkg.get("savings_percent"),
                    "total_cost": pkg.get("total_investment_sek"),  # Schema uses total_cost
                    "net_cost": pkg.get("total_investment_sek"),  # After deductions
                    "annual_savings_sek": pkg.get("annual_savings_sek"),
                    "simple_payback_years": pkg.get("simple_payback_years"),
                }
                package_data = {k: v for k, v in package_data.items() if v is not None}
                self.insert_package(package_data)

            logger.info(f"Analysis stored successfully for building {building_id}")
            return building_id

        except Exception as e:
            logger.error(f"Failed to store analysis: {e}")
            raise

    def store_portfolio_result(self, portfolio_id: str, building_result: dict) -> Optional[str]:
        """
        Store a single building result within a portfolio analysis.

        Args:
            portfolio_id: UUID of the portfolio
            building_result: Dict with building analysis data including:
                - tier: Analysis tier (skip, fast, standard, deep)
                - confidence: Overall confidence score
                - baseline_kwh_m2: Baseline energy consumption
                - savings_kwh_m2: Estimated savings
                - savings_percent: Savings as percentage
                - investment_sek: Total investment
                - payback_years: Simple payback
                - archetype_id: Matched archetype
                - qc_triggers: List of QC triggers if any

        Returns:
            Portfolio-building junction record ID
        """
        try:
            # First store the building analysis
            building_id = self.store_analysis(building_result)
            if not building_id:
                return None

            # Link to portfolio with full result data
            junction_data = {
                "portfolio_id": portfolio_id,
                "building_id": building_id,
                "tier": building_result.get("tier", "standard"),  # Schema column
                "analysis_status": "completed",
                # Results
                "baseline_kwh_m2": building_result.get("baseline_kwh_m2"),
                "savings_kwh_m2": building_result.get("savings_kwh_m2"),
                "savings_percent": building_result.get("savings_percent"),
                "investment_sek": building_result.get("investment_sek"),
                "payback_years": building_result.get("payback_years"),
                "npv_sek": building_result.get("npv_sek"),
                # Archetype
                "archetype_id": building_result.get("archetype_id"),
                "archetype_confidence": building_result.get("confidence"),
                # QC
                "needs_qc": bool(building_result.get("qc_triggers")),
                "qc_triggers": building_result.get("qc_triggers"),
                # Recommended ECMs
                "recommended_ecms": building_result.get("ecm_results", [])[:10],  # Top 10
                # Uncertainty
                "savings_uncertainty_kwh_m2": building_result.get("savings_std"),
                "savings_ci_90_lower": building_result.get("savings_ci_90", (None, None))[0] if building_result.get("savings_ci_90") else None,
                "savings_ci_90_upper": building_result.get("savings_ci_90", (None, None))[1] if building_result.get("savings_ci_90") else None,
            }
            # Remove None values
            junction_data = {k: v for k, v in junction_data.items() if v is not None}

            result = (
                self.client.table("portfolio_buildings")
                .insert(junction_data)
                .execute()
            )
            return result.data[0]["id"] if result.data else None

        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Failed to store portfolio result: {e}")
            return None


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
