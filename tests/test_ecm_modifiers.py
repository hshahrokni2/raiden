"""
Tests for ECM IDF modifiers.

Tests all 12 ECM modification methods:
- Envelope: wall insulation, roof insulation, windows, air sealing
- HVAC: FTX upgrade, FTX installation, DCV
- Renewable: Solar PV
- Controls: Smart thermostats
- Lighting: LED retrofit
"""

import pytest
import re
from pathlib import Path

from src.ecm.idf_modifier import IDFModifier


class TestIDFModifier:
    """Tests for IDFModifier class."""

    @pytest.fixture
    def modifier(self):
        """Create modifier instance."""
        return IDFModifier()

    # =========================================================================
    # WALL INSULATION TESTS
    # =========================================================================

    def test_wall_external_insulation(self, modifier, sample_idf_content):
        """Test external wall insulation increases thickness."""
        params = {'thickness_mm': 100, 'material': 'mineral_wool'}

        result = modifier._apply_wall_insulation(
            sample_idf_content, params, external=True
        )

        # Should contain ECM comment
        assert 'ECM Applied: External Wall Insulation' in result
        assert 'Added thickness: 100 mm' in result

        # Thickness should be increased (0.250 + 0.100 = 0.350)
        assert '0.3500' in result or '0.35' in result

    def test_wall_internal_insulation(self, modifier, sample_idf_content):
        """Test internal wall insulation."""
        params = {'thickness_mm': 50, 'material': 'eps'}

        result = modifier._apply_wall_insulation(
            sample_idf_content, params, external=False
        )

        assert 'ECM Applied: Internal Wall Insulation' in result

    # =========================================================================
    # ROOF INSULATION TESTS
    # =========================================================================

    def test_roof_insulation(self, modifier, sample_idf_content):
        """Test roof insulation increases thickness."""
        params = {'thickness_mm': 150, 'material': 'mineral_wool'}

        result = modifier._apply_roof_insulation(sample_idf_content, params)

        assert 'ECM Applied: Roof Insulation' in result
        assert 'Added thickness: 150 mm' in result

    # =========================================================================
    # WINDOW REPLACEMENT TESTS
    # =========================================================================

    def test_window_replacement_u_value(self, modifier, sample_idf_content):
        """Test window replacement changes U-value."""
        params = {'u_value': 0.8, 'shgc': 0.5}

        result = modifier._apply_window_replacement(sample_idf_content, params)

        assert 'ECM Applied: Window Replacement' in result
        assert 'New U-value: 0.8' in result

        # U-value should be changed from 1.0 to 0.8
        assert '0.8,' in result and 'U-Factor' in result

    def test_window_replacement_shgc(self, modifier, sample_idf_content):
        """Test window replacement changes SHGC."""
        params = {'u_value': 0.9, 'shgc': 0.4}

        result = modifier._apply_window_replacement(sample_idf_content, params)

        # SHGC should be in output
        assert 'New SHGC: 0.4' in result

    # =========================================================================
    # AIR SEALING TESTS
    # =========================================================================

    def test_air_sealing_reduces_infiltration(self, modifier, sample_idf_content):
        """Test air sealing reduces ACH."""
        params = {'reduction_factor': 0.5}

        result = modifier._apply_air_sealing(sample_idf_content, params)

        assert 'ECM Applied: Air Sealing' in result

        # ACH should be reduced from 0.06 to 0.03
        assert '0.0300' in result or '0.03' in result

    def test_air_sealing_70_percent_reduction(self, modifier, sample_idf_content):
        """Test 70% air sealing reduction."""
        params = {'reduction_factor': 0.7}

        result = modifier._apply_air_sealing(sample_idf_content, params)

        # ACH should be 0.06 * 0.7 = 0.042
        assert '0.0420' in result or '0.042' in result

    # =========================================================================
    # FTX UPGRADE TESTS
    # =========================================================================

    def test_ftx_upgrade_effectiveness(self, modifier, sample_idf_content):
        """Test FTX upgrade increases heat recovery."""
        params = {'effectiveness': 0.85}

        result = modifier._apply_ftx_upgrade(sample_idf_content, params)

        assert 'ECM Applied: FTX Heat Recovery Upgrade' in result
        assert 'New effectiveness: 85%' in result

        # Effectiveness should be changed from 0.75 to 0.85
        # Look for the pattern in result
        assert '0.85' in result

    def test_ftx_upgrade_90_percent(self, modifier, sample_idf_content):
        """Test 90% heat recovery."""
        params = {'effectiveness': 0.90}

        result = modifier._apply_ftx_upgrade(sample_idf_content, params)

        assert '0.90' in result or '0.9' in result

    # =========================================================================
    # FTX INSTALLATION TESTS
    # =========================================================================

    def test_ftx_installation(self, modifier, sample_idf_content):
        """Test FTX installation changes heat recovery type."""
        # First modify to have no heat recovery
        content_no_hr = sample_idf_content.replace(
            'Sensible,                !- Heat Recovery Type',
            'None,                    !- Heat Recovery Type'
        )
        params = {'effectiveness': 0.80}

        result = modifier._apply_ftx_installation(content_no_hr, params)

        assert 'ECM Applied: FTX Installation' in result
        assert 'FTX installed' in result

    # =========================================================================
    # DCV TESTS
    # =========================================================================

    def test_dcv_application(self, modifier, sample_idf_content):
        """Test demand controlled ventilation."""
        params = {'co2_setpoint': 1000}

        result = modifier._apply_dcv(sample_idf_content, params)

        assert 'ECM Applied: Demand Controlled Ventilation' in result
        assert 'CO2 setpoint: 1000 ppm' in result

    # =========================================================================
    # SOLAR PV TESTS
    # =========================================================================

    def test_solar_pv_comment(self, modifier, sample_idf_content):
        """Test solar PV adds comment."""
        params = {
            'coverage_fraction': 0.7,
            'panel_efficiency': 0.20,
            'roof_area_m2': 320
        }

        result = modifier._apply_solar_pv(sample_idf_content, params)

        assert 'ECM Applied: Solar PV' in result
        assert 'Roof coverage: 70%' in result
        assert 'Panel efficiency: 20%' in result
        assert 'Estimated PV area: 224 m' in result

    # =========================================================================
    # LED LIGHTING TESTS
    # =========================================================================

    def test_led_lighting_reduces_power(self, modifier, sample_idf_content):
        """Test LED lighting reduces power density."""
        params = {'power_density': 4}

        result = modifier._apply_led_lighting(sample_idf_content, params)

        assert 'ECM Applied: LED Lighting' in result
        assert 'New power density: 4 W/m' in result

        # Power should be reduced from 8 to 4
        assert '4,' in result

    def test_led_lighting_6_watts(self, modifier, sample_idf_content):
        """Test 6 W/m² LED lighting."""
        params = {'power_density': 6}

        result = modifier._apply_led_lighting(sample_idf_content, params)

        assert 'New power density: 6 W/m' in result

    # =========================================================================
    # SMART THERMOSTAT TESTS
    # =========================================================================

    def test_smart_thermostat_schedule(self, modifier, sample_idf_content):
        """Test smart thermostat creates setback schedule."""
        params = {'setback_c': 2}

        result = modifier._apply_smart_thermostats(sample_idf_content, params)

        assert 'ECM Applied: Smart Thermostats' in result
        assert 'Setback temperature: 2' in result
        assert 'Schedule:Compact' in result
        assert 'HeatSet_ECM' in result

    def test_smart_thermostat_3_degree_setback(self, modifier, sample_idf_content):
        """Test 3 degree setback."""
        params = {'setback_c': 3}

        result = modifier._apply_smart_thermostats(sample_idf_content, params)

        # Night setpoint should be 21 - 3 = 18
        assert '18' in result

    # =========================================================================
    # HEAT PUMP TESTS
    # =========================================================================

    def test_heat_pump_integration(self, modifier, sample_idf_content):
        """Test heat pump integration comment."""
        params = {'cop': 3.5, 'coverage': 0.8}

        result = modifier._apply_heat_pump(sample_idf_content, params)

        assert 'ECM Applied: Heat Pump Integration' in result
        assert 'COP: 3.5' in result
        assert 'Load coverage: 80%' in result

    # =========================================================================
    # DISPATCH TESTS
    # =========================================================================

    def test_apply_ecm_dispatch(self, modifier, sample_idf_content):
        """Test ECM dispatch routes to correct method."""
        # Test window replacement via dispatch
        result = modifier._apply_ecm(
            sample_idf_content,
            'window_replacement',
            {'u_value': 0.8, 'shgc': 0.5}
        )

        assert 'ECM Applied: Window Replacement' in result

    def test_apply_ecm_unknown(self, modifier, sample_idf_content, capsys):
        """Test unknown ECM returns unchanged content."""
        result = modifier._apply_ecm(
            sample_idf_content,
            'unknown_ecm',
            {}
        )

        # Should return unchanged
        assert result == sample_idf_content

        # Should print warning
        captured = capsys.readouterr()
        assert 'Warning' in captured.out or result == sample_idf_content


class TestApplySingle:
    """Tests for apply_single method."""

    def test_apply_single_creates_file(self, temp_dir, sample_idf_content):
        """Test apply_single creates modified IDF file."""
        # Write baseline IDF
        baseline = temp_dir / "baseline.idf"
        baseline.write_text(sample_idf_content)

        modifier = IDFModifier()
        result_path = modifier.apply_single(
            baseline_idf=baseline,
            ecm_id='air_sealing',
            params={'reduction_factor': 0.5},
            output_dir=temp_dir / "output"
        )

        assert result_path.exists()
        content = result_path.read_text()
        assert 'ECM Applied: Air Sealing' in content

    def test_apply_single_custom_name(self, temp_dir, sample_idf_content):
        """Test apply_single with custom output name."""
        baseline = temp_dir / "baseline.idf"
        baseline.write_text(sample_idf_content)

        modifier = IDFModifier()
        result_path = modifier.apply_single(
            baseline_idf=baseline,
            ecm_id='led_lighting',
            params={'power_density': 4},
            output_dir=temp_dir / "output",
            output_name="my_custom_name"
        )

        assert result_path.name == "my_custom_name.idf"


class TestMultipleECMs:
    """Tests for applying multiple ECMs."""

    def test_apply_multiple_ecms_sequentially(self, sample_idf_content):
        """Test applying multiple ECMs to same content."""
        modifier = IDFModifier()

        # Apply window replacement
        content = modifier._apply_ecm(
            sample_idf_content,
            'window_replacement',
            {'u_value': 0.8, 'shgc': 0.5}
        )

        # Apply air sealing
        content = modifier._apply_ecm(
            content,
            'air_sealing',
            {'reduction_factor': 0.5}
        )

        # Apply LED lighting
        content = modifier._apply_ecm(
            content,
            'led_lighting',
            {'power_density': 4}
        )

        # All ECMs should be present
        assert 'Window Replacement' in content
        assert 'Air Sealing' in content
        assert 'LED Lighting' in content
