"""
Tests for Supabase database integration.

These tests require a live Supabase connection.
Run with: pytest tests/test_database.py -v
"""

import pytest
import os
from uuid import uuid4
from dotenv import load_dotenv

# Load .env for tests
load_dotenv()

# Skip all tests if Supabase not configured
pytestmark = pytest.mark.skipif(
    not os.getenv("SUPABASE_URL"),
    reason="SUPABASE_URL not set - skipping database tests"
)


class TestSupabaseClient:
    """Test Supabase client connection."""

    def test_client_creation(self):
        """Client creates successfully."""
        from src.db import get_client
        client = get_client()
        assert client is not None

    def test_client_singleton(self):
        """Client is singleton."""
        from src.db import get_client
        client1 = get_client()
        client2 = get_client()
        assert client1 is client2


class TestBuildingRepository:
    """Test building CRUD operations."""

    @pytest.fixture
    def repo(self):
        from src.db import BuildingRepository
        return BuildingRepository()

    @pytest.fixture
    def test_building(self):
        from src.db import BuildingRecord
        return BuildingRecord(
            address=f"Test Building {uuid4().hex[:8]}",
            name="Test Building",
            construction_year=2000,
            heated_area_m2=1000,
            num_apartments=10,
            region="stockholm",
            owner_type="brf",
        )

    def test_create_building(self, repo, test_building):
        """Create building record."""
        saved = repo.create(test_building)
        assert saved is not None
        assert saved.id is not None
        assert saved.address == test_building.address

        # Cleanup
        repo.delete(saved.id)

    def test_get_building(self, repo, test_building):
        """Get building by ID."""
        saved = repo.create(test_building)

        retrieved = repo.get(saved.id)
        assert retrieved is not None
        assert retrieved.id == saved.id
        assert retrieved.name == test_building.name

        # Cleanup
        repo.delete(saved.id)

    def test_find_by_address(self, repo, test_building):
        """Find building by address."""
        saved = repo.create(test_building)

        found = repo.find_by_address(test_building.address)
        assert found is not None
        assert found.id == saved.id

        # Cleanup
        repo.delete(saved.id)

    def test_find_or_create_existing(self, repo, test_building):
        """Find existing building."""
        saved = repo.create(test_building)

        # Should find existing, not create new
        found = repo.find_or_create(test_building)
        assert found.id == saved.id

        # Cleanup
        repo.delete(saved.id)

    def test_update_building(self, repo, test_building):
        """Update building record."""
        saved = repo.create(test_building)

        updated = repo.update(saved.id, {"num_apartments": 20})
        assert updated.num_apartments == 20

        # Cleanup
        repo.delete(saved.id)

    def test_delete_building(self, repo, test_building):
        """Delete building record."""
        saved = repo.create(test_building)
        repo.delete(saved.id)

        retrieved = repo.get(saved.id)
        assert retrieved is None

    def test_list_buildings(self, repo, test_building):
        """List buildings."""
        saved = repo.create(test_building)

        buildings = repo.list(limit=10)
        assert len(buildings) > 0
        assert any(b.id == saved.id for b in buildings)

        # Cleanup
        repo.delete(saved.id)


class TestECMResultRepository:
    """Test ECM result operations."""

    @pytest.fixture
    def building_repo(self):
        from src.db import BuildingRepository
        return BuildingRepository()

    @pytest.fixture
    def ecm_repo(self):
        from src.db import ECMResultRepository
        return ECMResultRepository()

    @pytest.fixture
    def test_building(self, building_repo):
        from src.db import BuildingRecord
        building = BuildingRecord(
            address=f"ECM Test Building {uuid4().hex[:8]}",
            heated_area_m2=1000,
        )
        saved = building_repo.create(building)
        yield saved
        building_repo.delete(saved.id)

    def test_create_ecm_result(self, ecm_repo, test_building):
        """Create ECM result."""
        from src.db import ECMResultRecord

        result = ECMResultRecord(
            building_id=test_building.id,
            ecm_id="wall_external_insulation",
            ecm_name="External Wall Insulation",
            ecm_category="envelope",
            heating_savings_percent=15.0,
            total_cost=500000,
            simple_payback_years=12.5,
        )

        saved = ecm_repo.create(result)
        assert saved is not None
        assert saved.ecm_id == "wall_external_insulation"

    def test_get_ecm_results(self, ecm_repo, test_building):
        """Get ECM results for building."""
        from src.db import ECMResultRecord

        # Create multiple results
        for ecm_id in ["roof_insulation", "window_replacement"]:
            result = ECMResultRecord(
                building_id=test_building.id,
                ecm_id=ecm_id,
                heating_savings_percent=10.0,
                simple_payback_years=10.0,
            )
            ecm_repo.create(result)

        results = ecm_repo.get_for_building(test_building.id)
        assert len(results) >= 2

    def test_batch_create(self, ecm_repo, test_building):
        """Batch create ECM results."""
        from src.db import ECMResultRecord

        results = [
            ECMResultRecord(
                building_id=test_building.id,
                ecm_id=f"test_ecm_{i}",
                heating_savings_percent=5.0 * i,
            )
            for i in range(3)
        ]

        saved = ecm_repo.create_batch(results)
        assert len(saved) == 3


class TestPackageRepository:
    """Test package operations."""

    @pytest.fixture
    def building_repo(self):
        from src.db import BuildingRepository
        return BuildingRepository()

    @pytest.fixture
    def package_repo(self):
        from src.db import PackageRepository
        return PackageRepository()

    @pytest.fixture
    def test_building(self, building_repo):
        from src.db import BuildingRecord
        building = BuildingRecord(
            address=f"Package Test Building {uuid4().hex[:8]}",
            heated_area_m2=1000,
        )
        saved = building_repo.create(building)
        yield saved
        building_repo.delete(saved.id)

    def test_create_package(self, package_repo, test_building):
        """Create package."""
        from src.db import PackageRecord

        package = PackageRecord(
            building_id=test_building.id,
            package_name="Grundpaket",
            package_type="basic",
            ecm_ids=["air_sealing", "smart_thermostats"],
            combined_savings_percent=15.0,
            simple_payback_years=5.0,
        )

        saved = package_repo.create(package)
        assert saved is not None
        assert saved.package_name == "Grundpaket"

    def test_get_packages(self, package_repo, test_building):
        """Get packages for building."""
        from src.db import PackageRecord

        for name in ["Grundpaket", "Standardpaket"]:
            package = PackageRecord(
                building_id=test_building.id,
                package_name=name,
                simple_payback_years=10.0,
            )
            package_repo.create(package)

        packages = package_repo.get_for_building(test_building.id)
        assert len(packages) >= 2


class TestConvenienceFunctions:
    """Test high-level convenience functions."""

    @pytest.fixture
    def building_repo(self):
        from src.db import BuildingRepository
        return BuildingRepository()

    def test_save_full_analysis(self, building_repo):
        """Save complete analysis."""
        from src.db import (
            BuildingRecord,
            BaselineRecord,
            ECMResultRecord,
            PackageRecord,
        )
        from src.db.repository import save_full_analysis

        building = BuildingRecord(
            address=f"Full Analysis Test {uuid4().hex[:8]}",
            heated_area_m2=2000,
            num_apartments=20,
        )

        baseline = BaselineRecord(
            building_id="",  # Will be set by function
            heating_kwh_m2=95.0,
            is_calibrated=True,
        )

        ecm_results = [
            ECMResultRecord(
                building_id="",
                ecm_id="air_sealing",
                heating_savings_percent=10.0,
                simple_payback_years=3.0,
            ),
            ECMResultRecord(
                building_id="",
                ecm_id="roof_insulation",
                heating_savings_percent=8.0,
                simple_payback_years=15.0,
            ),
        ]

        packages = [
            PackageRecord(
                building_id="",
                package_name="Grundpaket",
                ecm_ids=["air_sealing"],
                simple_payback_years=3.0,
            ),
        ]

        result = save_full_analysis(
            building=building,
            baseline=baseline,
            ecm_results=ecm_results,
            packages=packages,
        )

        assert result["building_id"] is not None
        assert result["baseline_id"] is not None
        assert result["ecm_count"] == 2
        assert result["package_count"] == 1

        # Cleanup
        building_repo.delete(result["building_id"])

    def test_get_building_analysis(self, building_repo):
        """Get complete building analysis."""
        from src.db import BuildingRecord
        from src.db.repository import get_building_analysis

        building = BuildingRecord(
            address=f"Get Analysis Test {uuid4().hex[:8]}",
            heated_area_m2=1500,
        )
        saved = building_repo.create(building)

        analysis = get_building_analysis(saved.id)

        assert analysis["building"] is not None
        assert analysis["building"]["id"] == saved.id

        # Cleanup
        building_repo.delete(saved.id)
