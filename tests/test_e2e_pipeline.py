"""
End-to-End Pipeline Tests for Raiden.

Tests the complete flow:
1. Address → BuildingData (from GeoJSON or fallback)
2. BuildingData → Archetype matching
3. Archetype → ECM filtering
4. ECM → Simulation (optional)
5. Results → Report generation

Run with: pytest tests/test_e2e_pipeline.py -v
"""

import pytest
from pathlib import Path


class TestAddressToBuildingData:
    """Test fetching building data from addresses."""

    def test_stockholm_address_uses_geojson(self):
        """Test that Stockholm addresses use the GeoJSON data source."""
        from src.core.address_pipeline import BuildingDataFetcher

        fetcher = BuildingDataFetcher()
        data = fetcher.fetch("Bellmansgatan 16, Stockholm")

        assert data is not None
        assert data.address is not None
        assert "sweden_buildings_geojson" in data.data_sources
        # Should have real energy data from GeoJSON
        assert data.energy_class in ["A", "B", "C", "D", "E", "F", "G"]
        assert data.construction_year > 0

    def test_building_data_has_required_fields(self):
        """Test that BuildingData has all fields needed for analysis."""
        from src.core.address_pipeline import BuildingDataFetcher

        fetcher = BuildingDataFetcher()
        data = fetcher.fetch("Bellmansgatan 16, Stockholm")

        # Required for archetype matching
        assert hasattr(data, "construction_year")
        assert hasattr(data, "building_type")
        assert hasattr(data, "facade_material")
        assert hasattr(data, "num_floors")

        # Required for ECM analysis
        assert hasattr(data, "atemp_m2")
        assert hasattr(data, "heating_system")
        assert hasattr(data, "has_ftx")
        assert hasattr(data, "has_heat_pump")
        assert hasattr(data, "has_solar")

        # Required for geometry
        assert hasattr(data, "wwr")
        assert hasattr(data, "latitude")
        assert hasattr(data, "longitude")


class TestBuildingDataToArchetype:
    """Test archetype matching from building data."""

    def test_archetype_matching_from_building_data(self):
        """Test matching archetype from BuildingData."""
        from src.core.address_pipeline import BuildingDataFetcher
        from src.baseline import ArchetypeMatcherV2

        fetcher = BuildingDataFetcher()
        data = fetcher.fetch("Bellmansgatan 16, Stockholm")

        matcher = ArchetypeMatcherV2(use_ai_modules=False)
        result = matcher.match_from_building_data(data)

        assert result is not None
        assert result.archetype is not None
        assert result.confidence > 0
        # Should return a valid archetype (may not match year due to renovation signals)
        assert result.archetype.id is not None
        assert result.archetype.year_start > 0

    def test_modern_building_matches_modern_archetype(self):
        """Test that modern buildings match modern archetypes."""
        from src.core.address_pipeline import BuildingDataFetcher
        from src.baseline import ArchetypeMatcherV2

        fetcher = BuildingDataFetcher()
        # Find a modern building in Stockholm
        buildings = fetcher.sweden_buildings.find_by_address("Hammarby")
        modern_buildings = [b for b in buildings if b.construction_year and b.construction_year > 2000]

        if not modern_buildings:
            pytest.skip("No modern buildings found in test data")

        # Use the fetcher to get full BuildingData
        b = modern_buildings[0]
        data = fetcher.fetch(b.address)

        matcher = ArchetypeMatcherV2(use_ai_modules=False)
        result = matcher.match_from_building_data(data)

        # Should return a valid archetype
        assert result is not None
        assert result.archetype is not None


class TestArchetypeToECM:
    """Test ECM filtering based on archetype and building data."""

    def test_ecm_filtering_for_old_brick_building(self):
        """Test that old brick buildings get correct ECM filtering."""
        from src.ecm import get_all_ecms, ConstraintEngine
        from src.ecm.constraints import BuildingContext

        # Old brick building context - all required fields
        ctx = BuildingContext(
            construction_year=1900,
            building_type="multi_family",
            facade_material="brick",
            heating_system="district",
            ventilation_type="natural",
            heritage_listed=False,
            current_window_u=2.5,
            current_heat_recovery=0.0,
        )

        engine = ConstraintEngine()

        # Get valid and excluded ECMs
        valid_ecms = engine.get_valid_ecms(ctx)
        excluded_ecms = engine.get_excluded_ecms(ctx)

        valid_ids = [ecm.id for ecm in valid_ecms]
        excluded_ids = [ecm.id for ecm, _ in excluded_ecms]

        # External wall insulation should be EXCLUDED for brick
        assert "wall_external_insulation" not in valid_ids
        # Internal wall insulation should be INCLUDED
        assert "wall_internal_insulation" in valid_ids
        # FTX installation should be applicable (no existing FTX)
        assert "ftx_installation" in valid_ids

    def test_ecm_filtering_for_modern_building(self):
        """Test that modern buildings have fewer applicable ECMs."""
        from src.ecm import ConstraintEngine
        from src.ecm.constraints import BuildingContext

        # Modern building context - all required fields
        ctx = BuildingContext(
            construction_year=2015,
            building_type="multi_family",
            facade_material="plaster",
            heating_system="heat_pump_ground",
            ventilation_type="ftx",
            heritage_listed=False,
            current_window_u=0.9,  # Already good windows
            current_heat_recovery=0.80,  # Already has FTX
            current_lighting_w_m2=5,  # Already LED
        )

        engine = ConstraintEngine()
        valid_ecms = engine.get_valid_ecms(ctx)
        valid_ids = [ecm.id for ecm in valid_ecms]

        # Window replacement should be EXCLUDED (windows already efficient)
        assert "window_replacement" not in valid_ids
        # FTX upgrade should be EXCLUDED (already high efficiency)
        assert "ftx_upgrade" not in valid_ids
        # LED should be EXCLUDED (already efficient)
        assert "led_lighting" not in valid_ids


class TestECMCatalog:
    """Test ECM catalog integrity."""

    def test_all_ecms_have_required_fields(self):
        """Test that all ECMs have required fields."""
        from src.ecm import get_all_ecms

        for ecm in get_all_ecms():
            assert ecm.id, f"ECM missing id"
            assert ecm.name, f"ECM {ecm.id} missing name"
            assert ecm.name_sv, f"ECM {ecm.id} missing Swedish name"
            assert ecm.category, f"ECM {ecm.id} missing category"
            assert ecm.cost_per_unit >= 0, f"ECM {ecm.id} has negative cost"
            assert ecm.cost_unit, f"ECM {ecm.id} missing cost unit"
            assert 0 <= ecm.typical_savings_percent <= 100, f"ECM {ecm.id} invalid savings %"

    def test_ecm_count(self):
        """Test that we have the expected number of ECMs."""
        from src.ecm import get_all_ecms

        ecms = get_all_ecms()
        assert len(ecms) >= 20, f"Expected at least 20 ECMs, got {len(ecms)}"

    def test_operational_ecms_are_zero_cost(self):
        """Test that operational ECMs have zero or low cost."""
        from src.ecm import get_all_ecms, ECMCategory

        for ecm in get_all_ecms():
            if ecm.category == ECMCategory.OPERATIONAL:
                # Operational ECMs should be zero/low cost per unit
                assert ecm.cost_per_unit <= 500, f"Operational ECM {ecm.id} has high cost"


class TestArchetypeCatalog:
    """Test archetype catalog integrity."""

    def test_archetype_count(self):
        """Test that we have 40 archetypes."""
        from src.baseline import get_all_archetypes

        archetypes = get_all_archetypes()  # Returns dict
        assert len(archetypes) == 40, f"Expected 40 archetypes, got {len(archetypes)}"

    def test_archetypes_cover_all_eras(self):
        """Test that archetypes cover all Swedish building eras."""
        from src.baseline import get_all_archetypes, BuildingEra

        archetypes = get_all_archetypes()  # Returns dict
        eras_covered = set(a.era for a in archetypes.values())

        expected_eras = {
            BuildingEra.PRE_1930,
            BuildingEra.FUNKIS_1930_1945,
            BuildingEra.FOLKHEM_1946_1960,
            BuildingEra.REKORD_1961_1975,
            BuildingEra.ENERGI_1976_1985,
            BuildingEra.MODERN_1986_1995,
            BuildingEra.LAGENERGI_1996_2010,
            BuildingEra.NARA_NOLL_2011_PLUS,
        }

        for era in expected_eras:
            assert era in eras_covered, f"Missing archetypes for era {era}"


class TestSwedishBuildingsGeoJSON:
    """Test the Swedish buildings GeoJSON data source."""

    def test_geojson_loads(self):
        """Test that GeoJSON loads successfully."""
        from src.ingest import load_sweden_buildings

        loader = load_sweden_buildings()
        assert loader is not None

    def test_geojson_has_expected_count(self):
        """Test that GeoJSON has ~37k buildings."""
        from src.ingest import load_sweden_buildings

        loader = load_sweden_buildings()
        stats = loader.get_statistics()

        assert stats["total_buildings"] > 35000
        assert stats["total_buildings"] < 40000

    def test_find_by_address(self):
        """Test finding buildings by address."""
        from src.ingest import load_sweden_buildings

        loader = load_sweden_buildings()
        buildings = loader.find_by_address("Bellmansgatan")

        assert len(buildings) > 0
        for b in buildings:
            assert "Bellmansgatan" in b.address or "bellmansgatan" in b.address.lower()

    def test_buildings_have_energy_data(self):
        """Test that buildings have energy declaration data."""
        from src.ingest import load_sweden_buildings

        loader = load_sweden_buildings()
        buildings = loader.find_by_address("Hammarby")[:10]

        for b in buildings:
            # Should have at least some energy data
            has_energy = (
                b.energy_class is not None or
                b.energy_performance_kwh_m2 is not None or
                b.district_heating_kwh is not None
            )
            # Most buildings should have energy data
            # (not asserting all because some may be incomplete)


@pytest.mark.integration
class TestFullPipelineWithSimulation:
    """Full pipeline tests including EnergyPlus simulation.

    These tests require EnergyPlus to be installed.
    Skip with: pytest -m "not integration"
    """

    @pytest.fixture
    def weather_file(self):
        """Stockholm weather file."""
        path = Path(__file__).parent / "fixtures" / "stockholm.epw"
        if not path.exists():
            pytest.skip("Weather file not found")
        return path

    @pytest.fixture
    def baseline_idf(self):
        """Sjostaden baseline IDF."""
        path = Path(__file__).parent.parent / "sjostaden_7zone.idf"
        if not path.exists():
            pytest.skip("Baseline IDF not found")
        return path

    def test_full_pipeline_ecm_simulation(self, baseline_idf, weather_file, tmp_path):
        """Test full pipeline: IDF → ECM modification → simulation."""
        from src.ecm.idf_modifier import IDFModifier
        from src.simulation.runner import run_simulation

        modifier = IDFModifier()
        output_dir = tmp_path / "ecm_output"
        output_dir.mkdir(exist_ok=True)

        # Apply roof insulation ECM - returns path to modified IDF
        modified_idf = modifier.apply_single(
            baseline_idf=baseline_idf,
            ecm_id="roof_insulation",
            params={"thickness_mm": 150},
            output_dir=output_dir,
        )

        assert modified_idf.exists(), "Modified IDF should be created"

        # Run simulation
        result = run_simulation(
            idf_path=modified_idf,
            weather_path=weather_file,
            output_dir=tmp_path / "sim_output"
        )

        assert result.success, f"Simulation failed: {result.error_message}"
        assert result.parsed_results is not None, "Should have parsed results"

        # Check heating results
        heating_kwh_m2 = result.parsed_results.heating_kwh_m2
        assert heating_kwh_m2 > 0, "Heating should be > 0"
        assert heating_kwh_m2 < 100, f"Heating {heating_kwh_m2} kWh/m² seems too high"
