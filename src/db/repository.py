"""
Repository pattern for database access.

Provides high-level methods for storing and retrieving analysis results.
"""

from typing import Optional, List, Dict, Any
from dataclasses import asdict

from .client import get_client, SupabaseClient
from .models import (
    BuildingRecord,
    BaselineRecord,
    ECMResultRecord,
    PackageRecord,
    ReportRecord,
)


class BuildingRepository:
    """Repository for building records."""

    def __init__(self, client: Optional[SupabaseClient] = None):
        self._client = client or get_client()

    def create(self, building: BuildingRecord) -> BuildingRecord:
        """Create a new building record."""
        data = building.to_dict()
        result = self._client.insert_building(data)
        return BuildingRecord.from_dict(result) if result else None

    def get(self, building_id: str) -> Optional[BuildingRecord]:
        """Get building by ID."""
        result = self._client.get_building(building_id)
        return BuildingRecord.from_dict(result) if result else None

    def find_by_address(self, address: str) -> Optional[BuildingRecord]:
        """Find building by address."""
        result = self._client.get_building_by_address(address)
        return BuildingRecord.from_dict(result) if result else None

    def find_or_create(self, building: BuildingRecord) -> BuildingRecord:
        """Find existing building or create new one."""
        existing = self.find_by_address(building.address)
        if existing:
            return existing
        return self.create(building)

    def list(
        self,
        region: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[BuildingRecord]:
        """List buildings with optional filtering."""
        results = self._client.list_buildings(region, limit, offset)
        return [BuildingRecord.from_dict(r) for r in results]

    def update(self, building_id: str, updates: Dict[str, Any]) -> BuildingRecord:
        """Update building record."""
        result = self._client.update_building(building_id, updates)
        return BuildingRecord.from_dict(result) if result else None

    def delete(self, building_id: str) -> bool:
        """Delete building and all related records."""
        return self._client.delete_building(building_id)


class ECMResultRepository:
    """Repository for ECM analysis results."""

    def __init__(self, client: Optional[SupabaseClient] = None):
        self._client = client or get_client()

    def create(self, result: ECMResultRecord) -> ECMResultRecord:
        """Create a new ECM result record."""
        data = result.to_dict()
        db_result = self._client.insert_ecm_result(data)
        return ECMResultRecord.from_dict(db_result) if db_result else None

    def create_batch(self, results: List[ECMResultRecord]) -> List[ECMResultRecord]:
        """Create multiple ECM results."""
        data = [r.to_dict() for r in results]
        db_results = self._client.insert_ecm_results_batch(data)
        return [ECMResultRecord.from_dict(r) for r in db_results]

    def get_for_building(
        self,
        building_id: str,
        applicable_only: bool = True,
    ) -> List[ECMResultRecord]:
        """Get all ECM results for a building."""
        results = self._client.get_ecm_results(building_id, applicable_only)
        return [ECMResultRecord.from_dict(r) for r in results]

    def get_top_ecms(
        self,
        building_id: str,
        limit: int = 10,
        order_by: str = "simple_payback_years",
    ) -> List[ECMResultRecord]:
        """Get top ECMs for a building."""
        results = self._client.get_top_ecms(building_id, limit, order_by)
        return [ECMResultRecord.from_dict(r) for r in results]

    def save_analysis_results(
        self,
        building_id: str,
        baseline: BaselineRecord,
        ecm_results: List[ECMResultRecord],
    ) -> Dict[str, Any]:
        """
        Save complete analysis results for a building.

        Returns dict with saved record counts.
        """
        # Save baseline
        baseline.building_id = building_id
        baseline_data = baseline.to_dict()
        saved_baseline = self._client.insert_baseline(baseline_data)

        # Update ECM results with building and baseline IDs
        for result in ecm_results:
            result.building_id = building_id
            if saved_baseline:
                result.baseline_id = saved_baseline.get("id")

        # Batch insert ECM results
        saved_ecms = self.create_batch(ecm_results)

        return {
            "building_id": building_id,
            "baseline_id": saved_baseline.get("id") if saved_baseline else None,
            "ecm_count": len(saved_ecms),
        }


class PackageRepository:
    """Repository for ECM packages."""

    def __init__(self, client: Optional[SupabaseClient] = None):
        self._client = client or get_client()

    def create(self, package: PackageRecord) -> PackageRecord:
        """Create a new package record."""
        data = package.to_dict()
        result = self._client.insert_package(data)
        return PackageRecord.from_dict(result) if result else None

    def get_for_building(self, building_id: str) -> List[PackageRecord]:
        """Get all packages for a building."""
        results = self._client.get_packages(building_id)
        return [PackageRecord.from_dict(r) for r in results]

    def save_packages(
        self,
        building_id: str,
        packages: List[PackageRecord],
    ) -> List[PackageRecord]:
        """Save multiple packages for a building."""
        saved = []
        for package in packages:
            package.building_id = building_id
            saved.append(self.create(package))
        return saved


class ReportRepository:
    """Repository for analysis reports."""

    def __init__(self, client: Optional[SupabaseClient] = None):
        self._client = client or get_client()

    def create(self, report: ReportRecord) -> ReportRecord:
        """Create a new report record."""
        data = report.to_dict()
        result = self._client.insert_report(data)
        return ReportRecord.from_dict(result) if result else None

    def get_latest(self, building_id: str) -> Optional[ReportRecord]:
        """Get latest report for building."""
        result = self._client.get_latest_report(building_id)
        return ReportRecord.from_dict(result) if result else None


# ============================================
# Convenience Functions
# ============================================

def save_full_analysis(
    building: BuildingRecord,
    baseline: BaselineRecord,
    ecm_results: List[ECMResultRecord],
    packages: List[PackageRecord],
    report: Optional[ReportRecord] = None,
) -> Dict[str, Any]:
    """
    Save complete analysis results to database.

    Convenience function that handles all record types.
    """
    building_repo = BuildingRepository()
    ecm_repo = ECMResultRepository()
    package_repo = PackageRepository()
    report_repo = ReportRepository()

    # Create or find building
    saved_building = building_repo.find_or_create(building)
    building_id = saved_building.id

    # Save analysis results
    analysis_result = ecm_repo.save_analysis_results(
        building_id, baseline, ecm_results
    )

    # Save packages
    saved_packages = package_repo.save_packages(building_id, packages)

    # Save report if provided
    saved_report = None
    if report:
        report.building_id = building_id
        saved_report = report_repo.create(report)

    return {
        "building_id": building_id,
        "baseline_id": analysis_result.get("baseline_id"),
        "ecm_count": analysis_result.get("ecm_count"),
        "package_count": len(saved_packages),
        "report_id": saved_report.id if saved_report else None,
    }


def get_building_analysis(building_id: str) -> Dict[str, Any]:
    """
    Get complete analysis for a building.

    Returns building info, ECM results, packages, and latest report.
    """
    client = get_client()

    building = client.get_building(building_id)
    baseline = client.get_baseline(building_id)
    ecm_results = client.get_ecm_results(building_id)
    packages = client.get_packages(building_id)
    report = client.get_latest_report(building_id)

    return {
        "building": building,
        "baseline": baseline,
        "ecm_results": ecm_results,
        "packages": packages,
        "report": report,
    }
