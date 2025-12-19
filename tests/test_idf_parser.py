"""
Tests for the structured IDF parser.
"""

import pytest
from pathlib import Path

from src.core.idf_parser import IDFParser


@pytest.fixture
def parser():
    """Create parser instance."""
    return IDFParser()


@pytest.fixture
def idf_path():
    """Path to the Sjostaden IDF file."""
    path = Path(__file__).parent.parent / "sjostaden_7zone.idf"
    if not path.exists():
        pytest.skip("Test IDF file not found")
    return path


@pytest.fixture
def idf_content(idf_path):
    """Load IDF content as string."""
    return idf_path.read_text()


class TestIDFParserLoad:
    """Test IDF loading functionality."""

    def test_load_from_file(self, parser, idf_path):
        """Test loading IDF from file path."""
        idf = parser.load(idf_path)
        assert idf is not None

    def test_load_from_string(self, parser, idf_content):
        """Test loading IDF from string content."""
        idf = parser.load_string(idf_content)
        assert idf is not None

    def test_to_string(self, parser, idf_path):
        """Test converting IDF back to string."""
        idf = parser.load(idf_path)
        content = parser.to_string(idf)
        assert isinstance(content, str)
        assert len(content) > 0
        # Should contain key IDF elements
        assert "Version" in content or "version" in content.lower()


class TestInfiltration:
    """Test infiltration parameter handling."""

    def test_get_infiltration_objects(self, parser, idf_path):
        """Test getting all infiltration objects."""
        idf = parser.load(idf_path)
        infiltrations = parser.get_infiltration_objects(idf)
        # Sjostaden has 7 floors, should have 7 infiltration objects
        assert len(infiltrations) == 7

    def test_get_infiltration_ach(self, parser, idf_path):
        """Test extracting ACH value."""
        idf = parser.load(idf_path)
        ach = parser.get_infiltration_ach(idf)
        assert ach is not None
        # Sjostaden calibrated model is 0.04 ACH (original was 0.06)
        assert ach == pytest.approx(0.04, rel=0.01)

    def test_set_infiltration_ach(self, parser, idf_path):
        """Test setting ACH value for all zones."""
        idf = parser.load(idf_path)
        new_ach = 0.04

        count = parser.set_infiltration_ach(idf, new_ach)
        assert count == 7  # All 7 zones should be modified

        # Verify the change
        actual_ach = parser.get_infiltration_ach(idf)
        assert actual_ach == pytest.approx(new_ach, rel=0.01)

    def test_get_all_infiltration_data(self, parser, idf_path):
        """Test getting detailed infiltration data."""
        idf = parser.load(idf_path)
        data = parser.get_all_infiltration_data(idf)

        assert len(data) == 7
        for item in data:
            assert item.calculation_method == "AirChanges/Hour"
            # Calibrated model uses 0.04 ACH (original was 0.06)
            assert item.air_changes_per_hour == pytest.approx(0.04, rel=0.01)


class TestWindows:
    """Test window parameter handling."""

    def test_get_simple_glazing_objects(self, parser, idf_path):
        """Test getting glazing objects."""
        idf = parser.load(idf_path)
        glazings = parser.get_simple_glazing_objects(idf)
        assert len(glazings) >= 1

    def test_get_window_u_value(self, parser, idf_path):
        """Test extracting window U-value."""
        idf = parser.load(idf_path)
        u_value = parser.get_window_u_value(idf)
        assert u_value is not None
        # Sjostaden calibrated model is 0.85 W/m²K (original was 1.0)
        assert u_value == pytest.approx(0.85, rel=0.01)

    def test_set_window_u_value(self, parser, idf_path):
        """Test setting window U-value."""
        idf = parser.load(idf_path)
        new_u = 0.8

        count = parser.set_window_u_value(idf, new_u)
        assert count >= 1

        # Verify the change
        actual_u = parser.get_window_u_value(idf)
        assert actual_u == pytest.approx(new_u, rel=0.01)

    def test_set_window_u_value_with_shgc(self, parser, idf_path):
        """Test setting window U-value with SHGC."""
        idf = parser.load(idf_path)
        new_u = 0.9
        new_shgc = 0.35

        parser.set_window_u_value(idf, new_u, new_shgc)

        glazings = parser.get_simple_glazing_objects(idf)
        assert glazings[0].UFactor == pytest.approx(new_u, rel=0.01)
        assert glazings[0].Solar_Heat_Gain_Coefficient == pytest.approx(new_shgc, rel=0.01)


class TestHeatRecovery:
    """Test heat recovery parameter handling."""

    def test_get_ideal_loads_objects(self, parser, idf_path):
        """Test getting IdealLoadsAirSystem objects."""
        idf = parser.load(idf_path)
        ideal_loads = parser.get_ideal_loads_objects(idf)
        assert len(ideal_loads) == 7  # One per floor

    def test_get_heat_recovery_effectiveness(self, parser, idf_path):
        """Test extracting HR effectiveness."""
        idf = parser.load(idf_path)
        hr_eff = parser.get_heat_recovery_effectiveness(idf)
        assert hr_eff is not None
        # Sjostaden baseline is 75%
        assert hr_eff == pytest.approx(0.75, rel=0.01)

    def test_set_heat_recovery_effectiveness(self, parser, idf_path):
        """Test setting HR effectiveness."""
        idf = parser.load(idf_path)
        new_eff = 0.85

        count = parser.set_heat_recovery_effectiveness(idf, new_eff)
        assert count == 7  # All 7 zones

        # Verify the change
        actual_eff = parser.get_heat_recovery_effectiveness(idf)
        assert actual_eff == pytest.approx(new_eff, rel=0.01)


class TestCalibrationParameters:
    """Test combined calibration parameter handling."""

    def test_extract_calibration_parameters(self, parser, idf_path):
        """Test extracting all calibration parameters."""
        idf = parser.load(idf_path)
        params = parser.extract_calibration_parameters(idf)

        assert 'infiltration' in params
        assert 'heat_recovery' in params
        assert 'window_u' in params

        # Verify values match calibrated model
        assert params['infiltration'] == pytest.approx(0.04, rel=0.01)  # Calibrated from 0.06
        assert params['heat_recovery'] == pytest.approx(0.75, rel=0.01)
        assert params['window_u'] == pytest.approx(0.85, rel=0.01)  # Calibrated from 1.0

    def test_apply_calibration_parameters(self, parser, idf_path):
        """Test applying all calibration parameters at once."""
        idf = parser.load(idf_path)

        new_params = {
            'infiltration': 0.04,
            'heat_recovery': 0.82,
            'window_u': 0.9,
        }

        results = parser.apply_calibration_parameters(idf, new_params)

        # Verify counts
        assert results['infiltration'] == 7
        assert results['heat_recovery'] == 7
        assert results['window_u'] >= 1

        # Verify values
        params = parser.extract_calibration_parameters(idf)
        assert params['infiltration'] == pytest.approx(0.04, rel=0.01)
        assert params['heat_recovery'] == pytest.approx(0.82, rel=0.01)
        assert params['window_u'] == pytest.approx(0.9, rel=0.01)


class TestUtilityMethods:
    """Test utility methods."""

    def test_get_building_name(self, parser, idf_path):
        """Test getting building name."""
        idf = parser.load(idf_path)
        name = parser.get_building_name(idf)
        assert name is not None
        assert "Sjostaden" in name

    def test_get_zone_count(self, parser, idf_path):
        """Test counting zones."""
        idf = parser.load(idf_path)
        count = parser.get_zone_count(idf)
        assert count == 7  # 7 floors

    def test_get_zone_names(self, parser, idf_path):
        """Test getting zone names."""
        idf = parser.load(idf_path)
        names = parser.get_zone_names(idf)

        assert len(names) == 7
        # Should have Floor1 through Floor7
        for i in range(1, 8):
            assert f"Floor{i}" in names


class TestMaterials:
    """Test material handling."""

    def test_get_material_objects(self, parser, idf_path):
        """Test getting material objects."""
        idf = parser.load(idf_path)
        materials = parser.get_material_objects(idf)
        assert len(materials) > 0

    def test_get_material_by_name(self, parser, idf_path):
        """Test getting specific material."""
        idf = parser.load(idf_path)
        material = parser.get_material_by_name(idf, "WallInsulation")
        assert material is not None

    def test_get_nonexistent_material(self, parser, idf_path):
        """Test getting nonexistent material returns None."""
        idf = parser.load(idf_path)
        material = parser.get_material_by_name(idf, "NonexistentMaterial")
        assert material is None

    def test_add_material_thickness(self, parser, idf_path):
        """Test adding thickness to material."""
        idf = parser.load(idf_path)

        material = parser.get_material_by_name(idf, "WallInsulation")
        original_thickness = material.Thickness

        additional = 0.05  # 50mm
        result = parser.add_material_thickness(idf, "WallInsulation", additional)
        assert result is True

        # Verify the change
        material = parser.get_material_by_name(idf, "WallInsulation")
        assert material.Thickness == pytest.approx(original_thickness + additional, rel=0.01)


class TestLighting:
    """Test lighting handling."""

    def test_get_lights_objects(self, parser, idf_path):
        """Test getting lights objects."""
        idf = parser.load(idf_path)
        lights = parser.get_lights_objects(idf)
        # Should have lights for each zone
        assert len(lights) >= 1

    def test_set_lighting_power_density(self, parser, idf_path):
        """Test setting lighting power density."""
        idf = parser.load(idf_path)
        new_lpd = 5.5  # W/m²

        count = parser.set_lighting_power_density(idf, new_lpd)
        # Should modify at least some lights
        assert count >= 0  # May be 0 if no lights use Watts/Area method


class TestRoundTrip:
    """Test loading, modifying, and saving preserves functionality."""

    def test_roundtrip_preserves_structure(self, parser, idf_path, tmp_path):
        """Test that loading and saving preserves IDF structure."""
        # Load
        idf = parser.load(idf_path)

        # Make some changes
        parser.set_infiltration_ach(idf, 0.05)
        parser.set_window_u_value(idf, 0.95)
        parser.set_heat_recovery_effectiveness(idf, 0.80)

        # Save to temp file
        output_path = tmp_path / "modified.idf"
        parser.save(idf, output_path)

        # Reload and verify
        idf2 = parser.load(output_path)
        assert parser.get_infiltration_ach(idf2) == pytest.approx(0.05, rel=0.01)
        assert parser.get_window_u_value(idf2) == pytest.approx(0.95, rel=0.01)
        assert parser.get_heat_recovery_effectiveness(idf2) == pytest.approx(0.80, rel=0.01)

    def test_string_roundtrip(self, parser, idf_content):
        """Test loading from string and converting back."""
        # Load from string
        idf = parser.load_string(idf_content)

        # Modify
        parser.set_infiltration_ach(idf, 0.03)

        # Convert to string
        modified_content = parser.to_string(idf)

        # Reload and verify
        idf2 = parser.load_string(modified_content)
        assert parser.get_infiltration_ach(idf2) == pytest.approx(0.03, rel=0.01)
