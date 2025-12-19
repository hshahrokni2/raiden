"""Tests for Swedish building archetypes."""
import pytest
from src.baseline.archetypes import (
    SWEDISH_ARCHETYPES,
    SwedishArchetype,
    ArchetypeMatcher,
    BuildingType,
    HeatingSystem,
    VentilationType,
    EnvelopeProperties,
    HVACProperties,
    InternalLoads,
)


class TestSwedishArchetypes:
    """Test archetype database integrity."""

    def test_all_archetypes_have_required_fields(self):
        """Test that all archetypes have required fields populated."""
        for key, archetype in SWEDISH_ARCHETYPES.items():
            assert archetype.name, f"{key} missing name"
            assert archetype.era_start > 0, f"{key} invalid era_start"
            assert archetype.era_end > archetype.era_start, f"{key} era_end <= era_start"
            assert archetype.envelope is not None, f"{key} missing envelope"
            assert archetype.hvac is not None, f"{key} missing hvac"
            assert archetype.loads is not None, f"{key} missing loads"

    def test_all_archetypes_have_valid_u_values(self):
        """Test that U-values are physically reasonable."""
        for key, archetype in SWEDISH_ARCHETYPES.items():
            env = archetype.envelope
            # U-values should be positive and less than 3.0 W/m²K
            assert 0 < env.wall_u_value < 3.0, f"{key} invalid wall_u_value"
            assert 0 < env.roof_u_value < 2.0, f"{key} invalid roof_u_value"
            assert 0 < env.floor_u_value < 2.0, f"{key} invalid floor_u_value"
            assert 0 < env.window_u_value < 4.0, f"{key} invalid window_u_value"

    def test_all_archetypes_have_valid_shgc(self):
        """Test that SHGC values are in valid range 0-1."""
        for key, archetype in SWEDISH_ARCHETYPES.items():
            shgc = archetype.envelope.window_shgc
            assert 0 < shgc <= 1.0, f"{key} invalid window_shgc: {shgc}"

    def test_all_archetypes_have_valid_infiltration(self):
        """Test that infiltration rates are reasonable."""
        for key, archetype in SWEDISH_ARCHETYPES.items():
            ach = archetype.envelope.infiltration_ach
            # Infiltration should be between 0.02 and 0.5 ACH
            assert 0.02 <= ach <= 0.5, f"{key} invalid infiltration_ach: {ach}"

    def test_all_archetypes_have_valid_heat_recovery(self):
        """Test heat recovery efficiency is in valid range."""
        for key, archetype in SWEDISH_ARCHETYPES.items():
            hr = archetype.hvac.heat_recovery_efficiency
            assert 0 <= hr <= 0.95, f"{key} invalid heat_recovery: {hr}"

    def test_newer_archetypes_have_better_insulation(self):
        """Test that newer buildings have better U-values (lower)."""
        pre_1945 = SWEDISH_ARCHETYPES["pre_1945_brick"]
        modern = SWEDISH_ARCHETYPES["1996_2010_modern"]
        low_energy = SWEDISH_ARCHETYPES["2011_plus_low_energy"]

        # Modern should be better than pre-1945
        assert modern.envelope.wall_u_value < pre_1945.envelope.wall_u_value
        assert modern.envelope.window_u_value < pre_1945.envelope.window_u_value

        # Low energy should be best
        assert low_energy.envelope.wall_u_value < modern.envelope.wall_u_value

    def test_million_program_archetype_exists(self):
        """Test that Miljonprogrammet archetype is defined."""
        assert "1961_1975_concrete" in SWEDISH_ARCHETYPES
        mp = SWEDISH_ARCHETYPES["1961_1975_concrete"]
        assert mp.era_start == 1961
        assert mp.era_end == 1975
        assert "concrete" in mp.facade_materials

    def test_archetype_era_coverage(self):
        """Test that archetypes cover all eras from 1900 to present."""
        years_covered = set()
        for archetype in SWEDISH_ARCHETYPES.values():
            for year in range(archetype.era_start, archetype.era_end + 1):
                years_covered.add(year)

        # Should cover 1900-2030
        assert 1900 in years_covered
        assert 1968 in years_covered  # Million program peak
        assert 2020 in years_covered


class TestArchetypeMatcher:
    """Test archetype matching logic."""

    @pytest.fixture
    def matcher(self):
        return ArchetypeMatcher()

    def test_match_million_program_building(self, matcher):
        """Test matching a 1968 concrete building."""
        archetype = matcher.match(
            construction_year=1968,
            building_type=BuildingType.MULTI_FAMILY,
            facade_material='concrete'
        )

        assert archetype is not None
        assert 1961 <= 1968 <= archetype.era_end
        assert "Miljonprogrammet" in archetype.name or "1961" in archetype.name

    def test_match_pre_war_building(self, matcher):
        """Test matching a pre-war brick building."""
        archetype = matcher.match(
            construction_year=1935,
            building_type=BuildingType.MULTI_FAMILY,
            facade_material='brick'
        )

        assert archetype is not None
        assert archetype.era_start <= 1935 <= archetype.era_end

    def test_match_modern_building(self, matcher):
        """Test matching a modern building."""
        archetype = matcher.match(
            construction_year=2015,
            building_type=BuildingType.MULTI_FAMILY
        )

        assert archetype is not None
        assert archetype.envelope.wall_u_value < 0.2  # Should be low energy

    def test_match_by_year_only(self, matcher):
        """Test matching with only construction year."""
        archetype = matcher.match(construction_year=1980)

        assert archetype is not None
        assert archetype.era_start <= 1980 <= archetype.era_end

    def test_match_returns_best_candidate(self, matcher):
        """Test that matching prioritizes facade material when given."""
        # Concrete facade should prefer Miljonprogrammet even at boundary years
        archetype = matcher.match(
            construction_year=1975,
            facade_material='concrete'
        )

        # Should match concrete panel archetype
        assert 'concrete' in archetype.facade_materials


class TestEnvelopeProperties:
    """Test envelope property dataclass."""

    def test_create_envelope(self):
        """Test creating envelope properties."""
        env = EnvelopeProperties(
            wall_u_value=0.20,
            roof_u_value=0.10,
            floor_u_value=0.15,
            window_u_value=1.0,
            window_shgc=0.50,
            infiltration_ach=0.06
        )

        assert env.wall_u_value == 0.20
        assert env.infiltration_ach == 0.06


class TestHVACProperties:
    """Test HVAC property dataclass."""

    def test_create_hvac_with_ftx(self):
        """Test creating HVAC with FTX system."""
        hvac = HVACProperties(
            heating_system=HeatingSystem.DISTRICT,
            ventilation_type=VentilationType.BALANCED,
            heat_recovery_efficiency=0.75,
            ventilation_rate_l_s_m2=0.35,
            sfp_kw_per_m3s=1.5
        )

        assert hvac.ventilation_type == VentilationType.BALANCED
        assert hvac.heat_recovery_efficiency == 0.75

    def test_create_hvac_natural_ventilation(self):
        """Test creating HVAC with natural ventilation."""
        hvac = HVACProperties(
            heating_system=HeatingSystem.DISTRICT,
            ventilation_type=VentilationType.NATURAL,
            heat_recovery_efficiency=0.0,
            ventilation_rate_l_s_m2=0.35,
            sfp_kw_per_m3s=0.0
        )

        assert hvac.ventilation_type == VentilationType.NATURAL
        assert hvac.heat_recovery_efficiency == 0.0


class TestInternalLoads:
    """Test internal loads dataclass."""

    def test_sveby_default_loads(self):
        """Test Sveby-based internal load defaults."""
        loads = InternalLoads(
            occupancy_m2_per_person=25,
            occupancy_heat_w_per_person=80,
            lighting_w_m2=8,
            equipment_w_m2=10,
            dhw_kwh_m2_year=25
        )

        # Sveby standard values
        assert loads.lighting_w_m2 == 8
        assert loads.equipment_w_m2 == 10


class TestBuildingTypes:
    """Test building type enum."""

    def test_multi_family_swedish_name(self):
        """Test that multi-family has correct Swedish name."""
        assert BuildingType.MULTI_FAMILY.value == "flerbostadshus"

    def test_single_family_swedish_name(self):
        """Test that single-family has correct Swedish name."""
        assert BuildingType.SINGLE_FAMILY.value == "småhus"


class TestHeatingSystem:
    """Test heating system enum."""

    def test_district_heating_swedish_name(self):
        """Test that district heating has correct Swedish name."""
        assert HeatingSystem.DISTRICT.value == "fjärrvärme"

    def test_heat_pump_ground_swedish_name(self):
        """Test that ground source HP has correct Swedish name."""
        assert HeatingSystem.HEAT_PUMP_GROUND.value == "bergvärme"


class TestVentilationType:
    """Test ventilation type enum."""

    def test_ftx_ventilation(self):
        """Test FTX ventilation type."""
        assert VentilationType.BALANCED.value == "ftx"

    def test_exhaust_ventilation(self):
        """Test exhaust (F-system) ventilation type."""
        assert VentilationType.EXHAUST.value == "frånluft"
