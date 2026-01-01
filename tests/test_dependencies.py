"""
Tests for ECM dependency and conflict matrix.
"""

import pytest

from src.ecm.dependencies import (
    ECMDependencyMatrix,
    ECMRelation,
    RelationType,
    get_dependency_matrix,
    validate_package,
    get_package_synergy,
    suggest_additions,
    CONFLICTS,
    DEPENDENCIES,
    SYNERGIES,
    ANTI_SYNERGIES,
)


class TestRelationDefinitions:
    """Test that relationships are properly defined."""

    def test_conflicts_defined(self):
        """Conflicts list has entries."""
        assert len(CONFLICTS) > 5

    def test_dependencies_defined(self):
        """Dependencies list has entries."""
        assert len(DEPENDENCIES) >= 2  # Reduced after removing DCV->FTX

    def test_synergies_defined(self):
        """Synergies list has entries."""
        assert len(SYNERGIES) > 3

    def test_anti_synergies_defined(self):
        """Anti-synergies list has entries."""
        assert len(ANTI_SYNERGIES) > 2

    def test_synergy_factors_positive(self):
        """Synergy factors are > 1."""
        for syn in SYNERGIES:
            assert syn.factor > 1.0, f"Synergy {syn.ecm_a}/{syn.ecm_b} factor should be > 1"

    def test_anti_synergy_factors_less_than_one(self):
        """Anti-synergy factors are < 1."""
        for anti in ANTI_SYNERGIES:
            assert anti.factor < 1.0, f"Anti-synergy {anti.ecm_a}/{anti.ecm_b} factor should be < 1"


class TestECMDependencyMatrix:
    """Test ECMDependencyMatrix class."""

    @pytest.fixture
    def matrix(self):
        """Get dependency matrix."""
        return ECMDependencyMatrix()

    def test_matrix_creation(self, matrix):
        """Matrix creates successfully."""
        assert matrix is not None

    def test_validate_valid_combination(self, matrix):
        """Valid combination passes validation."""
        # air_sealing and smart_thermostats have no conflict
        is_valid, issues = matrix.validate_combination([
            "air_sealing", "smart_thermostats"
        ])
        assert is_valid
        assert len([i for i in issues if i.startswith("Conflict:")]) == 0

    def test_validate_conflicting_combination(self, matrix):
        """Conflicting ECMs fail validation."""
        # wall_external and wall_internal conflict
        is_valid, issues = matrix.validate_combination([
            "wall_external_insulation", "wall_internal_insulation"
        ])
        assert not is_valid
        assert any("Conflict" in issue for issue in issues)

    def test_validate_missing_dependency(self, matrix):
        """Missing dependency fails validation."""
        # battery_storage requires solar_pv
        is_valid, issues = matrix.validate_combination([
            "battery_storage"
        ])
        assert not is_valid
        assert any("Dependency" in issue for issue in issues)

    def test_validate_with_dependency_satisfied(self, matrix):
        """Satisfied dependency passes."""
        # battery_storage with solar_pv
        is_valid, issues = matrix.validate_combination([
            "solar_pv", "battery_storage"
        ])
        # Should pass conflict/dependency checks (supersede warnings ok)
        conflict_errors = [i for i in issues if i.startswith("Conflict:") or i.startswith("Dependency:")]
        assert len(conflict_errors) == 0

    def test_supersedes_warning(self, matrix):
        """Superseded ECMs generate warnings."""
        # building_automation_system supersedes smart_thermostats
        is_valid, issues = matrix.validate_combination([
            "building_automation_system", "smart_thermostats"
        ])
        # Should be valid but with warning
        assert is_valid  # Not an error
        assert any("Redundant" in issue for issue in issues)


class TestSynergyCalculation:
    """Test synergy factor calculations."""

    @pytest.fixture
    def matrix(self):
        return ECMDependencyMatrix()

    def test_single_ecm_no_synergy(self, matrix):
        """Single ECM has synergy factor 1.0."""
        factor = matrix.calculate_synergy_factor(["air_sealing"])
        assert factor == 1.0

    def test_synergistic_pair(self, matrix):
        """Synergistic ECMs have factor > 1."""
        # air_sealing + ftx_installation have synergy
        factor = matrix.calculate_synergy_factor([
            "air_sealing", "ftx_installation"
        ])
        assert factor > 1.0

    def test_anti_synergistic_pair(self, matrix):
        """Anti-synergistic ECMs have factor < 1."""
        # smart_thermostats + heating_curve_adjustment have anti-synergy
        factor = matrix.calculate_synergy_factor([
            "smart_thermostats", "heating_curve_adjustment"
        ])
        assert factor < 1.0

    def test_no_interaction_pair(self, matrix):
        """Non-interacting ECMs have factor 1.0."""
        # led_lighting and roof_insulation - no defined interaction
        factor = matrix.calculate_synergy_factor([
            "led_lighting", "roof_insulation"
        ])
        assert factor == 1.0


class TestValidAdditions:
    """Test getting valid ECM additions."""

    @pytest.fixture
    def matrix(self):
        return ECMDependencyMatrix()

    def test_excludes_conflicting(self, matrix):
        """Conflicting ECMs not suggested."""
        all_ecms = [
            "wall_external_insulation",
            "wall_internal_insulation",
            "roof_insulation"
        ]
        current = ["wall_external_insulation"]

        valid = matrix.get_valid_additions(current, all_ecms)

        # wall_internal should be excluded
        assert "wall_internal_insulation" not in valid
        assert "roof_insulation" in valid

    def test_excludes_current(self, matrix):
        """Already selected ECMs not suggested."""
        all_ecms = ["air_sealing", "roof_insulation"]
        current = ["air_sealing"]

        valid = matrix.get_valid_additions(current, all_ecms)

        assert "air_sealing" not in valid
        assert "roof_insulation" in valid


class TestQueryMethods:
    """Test query methods."""

    @pytest.fixture
    def matrix(self):
        return ECMDependencyMatrix()

    def test_get_required_ecms(self, matrix):
        """Get dependencies for an ECM."""
        required = matrix.get_required_ecms("battery_storage")
        assert "solar_pv" in required

    def test_get_conflicting_ecms(self, matrix):
        """Get conflicts for an ECM."""
        conflicts = matrix.get_conflicting_ecms("wall_external_insulation")
        assert "wall_internal_insulation" in conflicts

    def test_get_synergistic_ecms(self, matrix):
        """Get synergies for an ECM."""
        synergies = matrix.get_synergistic_ecms("air_sealing")
        synergy_ids = [s[0] for s in synergies]
        assert "ftx_installation" in synergy_ids

    def test_suggest_complementary(self, matrix):
        """Suggest complementary ECMs."""
        available = [
            "ftx_installation", "roof_insulation", "led_lighting",
            "wall_external_insulation"
        ]
        suggestions = matrix.suggest_complementary_ecms(
            "air_sealing", available, max_suggestions=3
        )

        # Should suggest ftx_installation (synergy)
        suggestion_ids = [s[0] for s in suggestions]
        assert "ftx_installation" in suggestion_ids


class TestExport:
    """Test export functionality."""

    def test_to_dict(self):
        """Matrix exports to dict."""
        matrix = ECMDependencyMatrix()
        d = matrix.to_dict()

        assert "conflicts" in d
        assert "dependencies" in d
        assert "synergies" in d
        assert "anti_synergies" in d
        assert "supersedes" in d

        # Check structure
        assert len(d["conflicts"]) > 0
        assert "ecm_a" in d["conflicts"][0]
        assert "reason" in d["conflicts"][0]


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_get_dependency_matrix(self):
        """Global matrix is available."""
        matrix = get_dependency_matrix()
        assert matrix is not None

    def test_validate_package(self):
        """validate_package function works."""
        is_valid, issues = validate_package(["air_sealing", "roof_insulation"])
        assert is_valid

    def test_get_package_synergy(self):
        """get_package_synergy function works."""
        factor = get_package_synergy(["air_sealing", "ftx_installation"])
        assert factor > 1.0

    def test_suggest_additions(self):
        """suggest_additions function works."""
        all_ecms = ["air_sealing", "roof_insulation", "wall_external_insulation"]
        valid = suggest_additions([], all_ecms)
        assert len(valid) == 3


class TestIntegration:
    """Integration tests."""

    def test_full_package_validation(self):
        """Validate a realistic package."""
        package = [
            "air_sealing",
            "window_replacement",
            "ftx_installation",
            "smart_thermostats",
        ]

        is_valid, issues = validate_package(package)
        assert is_valid

        factor = get_package_synergy(package)
        # Should have some synergy (air_sealing + ftx)
        assert factor != 1.0

    def test_invalid_package_detected(self):
        """Invalid package is caught."""
        # Mix central and apartment ventilation
        package = [
            "ftx_installation",
            "apartment_ventilation_units",
        ]

        is_valid, issues = validate_package(package)
        assert not is_valid
        assert any("Conflict" in i for i in issues)
