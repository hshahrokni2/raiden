"""
Tests for baseline calibrator.

Tests:
- Parameter extraction from IDF
- Parameter adjustment calculations
- IDF modification for calibration
- Calibration bounds checking
"""

import pytest
from pathlib import Path

from src.baseline.calibrator import (
    BaselineCalibrator,
    CalibrationResult,
    CalibrationParameter,
)


class TestCalibrationParameter:
    """Tests for CalibrationParameter dataclass."""

    def test_create_parameter(self):
        """Test creating a calibration parameter."""
        param = CalibrationParameter(
            name='infiltration',
            current_value=0.06,
            min_value=0.02,
            max_value=0.15,
            sensitivity=80.0
        )

        assert param.name == 'infiltration'
        assert param.current_value == 0.06
        assert param.sensitivity == 80.0


class TestParameterExtraction:
    """Tests for parameter extraction from IDF."""

    @pytest.fixture
    def calibrator(self):
        """Create calibrator without E+ path check."""
        # Mock the E+ finder to avoid requirement
        class MockCalibrator(BaselineCalibrator):
            def __init__(self):
                self.runner = None
                self.parser = None

        return MockCalibrator()

    def test_extract_infiltration(self, calibrator, sample_idf_content):
        """Test infiltration extraction from IDF."""
        params = calibrator._extract_parameters(sample_idf_content)

        assert 'infiltration' in params
        assert params['infiltration'] == 0.06

    def test_extract_heat_recovery(self, calibrator, sample_idf_content):
        """Test heat recovery extraction from IDF."""
        params = calibrator._extract_parameters(sample_idf_content)

        assert 'heat_recovery' in params
        assert params['heat_recovery'] == 0.75

    def test_extract_window_u(self, calibrator, sample_idf_content):
        """Test window U-value extraction from IDF."""
        params = calibrator._extract_parameters(sample_idf_content)

        assert 'window_u' in params
        assert params['window_u'] == 1.0

    def test_extract_defaults_when_missing(self, calibrator):
        """Test default values when IDF lacks parameters."""
        minimal_idf = "Version, 25.1;"

        params = calibrator._extract_parameters(minimal_idf)

        # Should return defaults
        assert params['infiltration'] == 0.06
        assert params['heat_recovery'] == 0.75
        assert params['window_u'] == 1.0


class TestParameterAdjustment:
    """Tests for parameter adjustment calculations."""

    @pytest.fixture
    def calibrator(self):
        """Create calibrator for adjustment tests."""
        class MockCalibrator(BaselineCalibrator):
            def __init__(self):
                pass

        return MockCalibrator()

    def test_adjustment_to_reduce_heating(self, calibrator):
        """Test adjustments to reduce heating demand."""
        current = {
            'infiltration': 0.06,
            'heat_recovery': 0.75,
            'window_u': 1.0
        }

        # Need to reduce heating by 10 kWh/m²
        delta_kwh_m2 = -10  # Negative means reduce

        new_params = calibrator._calculate_adjustments(
            current_params=current,
            delta_kwh_m2=delta_kwh_m2,
            damping=1.0  # No damping for test
        )

        # Heat recovery should increase (reduces heating)
        assert new_params['heat_recovery'] > current['heat_recovery']

        # Infiltration should decrease (reduces heating)
        assert new_params['infiltration'] < current['infiltration']

        # Window U should decrease (reduces heating)
        assert new_params['window_u'] < current['window_u']

    def test_adjustment_bounds_respected(self, calibrator):
        """Test that adjustments stay within bounds."""
        current = {
            'infiltration': 0.03,  # Already low
            'heat_recovery': 0.88,  # Already high
            'window_u': 0.75  # Already low
        }

        # Try to reduce heating a lot
        new_params = calibrator._calculate_adjustments(
            current_params=current,
            delta_kwh_m2=-50,  # Large reduction
            damping=1.0
        )

        # Should be clamped to bounds
        assert new_params['infiltration'] >= calibrator.PARAM_BOUNDS['infiltration'][0]
        assert new_params['heat_recovery'] <= calibrator.PARAM_BOUNDS['heat_recovery'][1]
        assert new_params['window_u'] >= calibrator.PARAM_BOUNDS['window_u'][0]

    def test_adjustment_damping(self, calibrator):
        """Test that damping reduces adjustment magnitude."""
        current = {
            'infiltration': 0.06,
            'heat_recovery': 0.75,
            'window_u': 1.0
        }

        # Same delta, different damping
        no_damping = calibrator._calculate_adjustments(
            current_params=current,
            delta_kwh_m2=-10,
            damping=1.0
        )

        with_damping = calibrator._calculate_adjustments(
            current_params=current,
            delta_kwh_m2=-10,
            damping=0.5
        )

        # Changes should be smaller with damping
        hr_change_no_damp = abs(no_damping['heat_recovery'] - current['heat_recovery'])
        hr_change_damped = abs(with_damping['heat_recovery'] - current['heat_recovery'])

        assert hr_change_damped < hr_change_no_damp


class TestIDFModification:
    """Tests for IDF modification during calibration."""

    @pytest.fixture
    def calibrator(self):
        """Create calibrator for modification tests."""
        class MockCalibrator(BaselineCalibrator):
            def __init__(self):
                pass

        return MockCalibrator()

    def test_modify_infiltration(self, calibrator, sample_idf_content):
        """Test infiltration modification in IDF."""
        params = {
            'infiltration': 0.04,
            'heat_recovery': 0.75,
            'window_u': 1.0
        }

        modified = calibrator._modify_idf(sample_idf_content, params)

        # Should have new infiltration value
        assert '0.0400' in modified or '0.04' in modified

    def test_modify_heat_recovery(self, calibrator, sample_idf_content):
        """Test heat recovery modification in IDF."""
        params = {
            'infiltration': 0.06,
            'heat_recovery': 0.82,
            'window_u': 1.0
        }

        modified = calibrator._modify_idf(sample_idf_content, params)

        # Should have new HR value
        assert '0.82' in modified

    def test_modify_multiple_parameters(self, calibrator, sample_idf_content):
        """Test modifying all parameters simultaneously."""
        params = {
            'infiltration': 0.04,
            'heat_recovery': 0.85,
            'window_u': 0.9
        }

        modified = calibrator._modify_idf(sample_idf_content, params)

        # All values should be present
        assert '0.04' in modified or '0.0400' in modified
        assert '0.85' in modified
        assert '0.9' in modified or '0.90' in modified


class TestCalibrationBounds:
    """Tests for calibration parameter bounds."""

    def test_infiltration_bounds(self):
        """Test infiltration bounds are physically reasonable."""
        bounds = BaselineCalibrator.PARAM_BOUNDS['infiltration']

        assert bounds[0] >= 0.01  # Minimum airtightness
        assert bounds[1] <= 0.20  # Maximum leakiness
        assert bounds[0] < bounds[1]

    def test_heat_recovery_bounds(self):
        """Test heat recovery bounds are reasonable."""
        bounds = BaselineCalibrator.PARAM_BOUNDS['heat_recovery']

        assert bounds[0] >= 0.50  # Minimum reasonable HR
        assert bounds[1] <= 0.95  # Maximum physical HR
        assert bounds[0] < bounds[1]

    def test_window_u_bounds(self):
        """Test window U-value bounds are reasonable."""
        bounds = BaselineCalibrator.PARAM_BOUNDS['window_u']

        assert bounds[0] >= 0.5   # Very good windows
        assert bounds[1] <= 3.0   # Old single glazing
        assert bounds[0] < bounds[1]


class TestCalibrationSensitivities:
    """Tests for parameter sensitivities."""

    def test_sensitivity_signs(self):
        """Test that sensitivities have correct signs."""
        sens = BaselineCalibrator.PARAM_SENSITIVITIES

        # Higher infiltration = more heating (positive)
        assert sens['infiltration'] > 0

        # Higher heat recovery = less heating (negative)
        assert sens['heat_recovery'] < 0

        # Higher window U = more heating (positive)
        assert sens['window_u'] > 0

    def test_sensitivity_magnitudes_reasonable(self):
        """Test that sensitivity magnitudes are reasonable."""
        sens = BaselineCalibrator.PARAM_SENSITIVITIES

        # Infiltration: ~80 kWh/m² per ACH
        assert 50 < abs(sens['infiltration']) < 150

        # Heat recovery: ~50 kWh/m² per 0.1 HR
        assert 30 < abs(sens['heat_recovery']) < 100

        # Window U: ~8 kWh/m² per W/m²K
        assert 3 < abs(sens['window_u']) < 20


class TestCalibrationResult:
    """Tests for CalibrationResult dataclass."""

    def test_success_result(self):
        """Test successful calibration result."""
        result = CalibrationResult(
            success=True,
            iterations=3,
            final_error_percent=5.0,
            adjusted_infiltration_ach=0.04,
            adjusted_heat_recovery=0.82,
            adjusted_window_u=0.9,
            measured_kwh_m2=33.0,
            initial_kwh_m2=42.0,
            calibrated_kwh_m2=34.0,
            calibrated_idf_path=Path('/tmp/calibrated.idf')
        )

        assert result.success
        assert result.iterations == 3
        assert abs(result.final_error_percent) < 10

    def test_failed_result(self):
        """Test failed calibration result."""
        result = CalibrationResult(
            success=False,
            iterations=10,
            final_error_percent=25.0,
            adjusted_infiltration_ach=0.02,
            adjusted_heat_recovery=0.90,
            adjusted_window_u=0.7,
            measured_kwh_m2=33.0,
            initial_kwh_m2=50.0,
            calibrated_kwh_m2=41.0,
        )

        assert not result.success
        assert result.final_error_percent > 10
