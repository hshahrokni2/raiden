"""
Integration tests with actual EnergyPlus simulation.

These tests require:
- EnergyPlus 25.1.0 installed
- Weather file in tests/fixtures/stockholm.epw

Skip with: pytest -m "not integration"
"""
import pytest
from pathlib import Path
import shutil

from src.simulation.runner import SimulationRunner, run_simulation
from src.simulation.results import ResultsParser
from src.baseline.calibrator import BaselineCalibrator, CalibrationResult
from src.ecm.idf_modifier import IDFModifier


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def weather_file():
    """Stockholm weather file for testing."""
    weather_path = Path(__file__).parent / "fixtures" / "stockholm.epw"
    if not weather_path.exists():
        pytest.skip("Weather file not found: tests/fixtures/stockholm.epw")
    return weather_path


@pytest.fixture
def model_file():
    """Sjostaden 7-zone model for testing (calibrated version)."""
    model_path = Path(__file__).parent.parent / "sjostaden_7zone.idf"
    if not model_path.exists():
        pytest.skip("Model file not found: sjostaden_7zone.idf")
    return model_path


@pytest.fixture
def original_model_file():
    """Sjostaden 7-zone model - original uncalibrated version for calibration tests."""
    model_path = Path(__file__).parent.parent / "sjostaden_7zone_original.idf"
    if not model_path.exists():
        pytest.skip("Original model file not found: sjostaden_7zone_original.idf")
    return model_path


@pytest.fixture
def energyplus_available():
    """Check if EnergyPlus is available."""
    if not shutil.which('energyplus'):
        pytest.skip("EnergyPlus not installed")
    return True


class TestEndToEndSimulation:
    """End-to-end simulation tests."""

    def test_full_simulation_runs(self, model_file, weather_file, energyplus_available, tmp_path):
        """Test that full simulation completes successfully."""
        output_dir = tmp_path / "output"

        runner = SimulationRunner()
        result = runner.run(
            idf_path=model_file,
            weather_path=weather_file,
            output_dir=output_dir,
            timeout_seconds=300
        )

        assert result.success, f"Simulation failed: {result.error_message}"
        assert result.runtime_seconds > 0
        assert (output_dir / "eplusout.err").exists()

    def test_results_parsing_after_simulation(self, model_file, weather_file, energyplus_available, tmp_path):
        """Test that results can be parsed after simulation."""
        output_dir = tmp_path / "output"

        result = run_simulation(
            idf_path=model_file,
            weather_path=weather_file,
            output_dir=output_dir,
            parse_results=True
        )

        assert result.success, f"Simulation failed: {result.error_message}"
        assert result.parsed_results is not None

        # Check parsed values are reasonable
        parsed = result.parsed_results
        assert parsed.floor_area_m2 > 0
        assert parsed.heating_kwh > 0
        assert parsed.heating_kwh_m2 > 0

    def test_heating_intensity_reasonable(self, model_file, weather_file, energyplus_available, tmp_path):
        """Test that heating intensity is in reasonable Swedish range."""
        output_dir = tmp_path / "output"

        result = run_simulation(
            idf_path=model_file,
            weather_path=weather_file,
            output_dir=output_dir,
            parse_results=True
        )

        assert result.success
        parsed = result.parsed_results

        # Swedish multi-family buildings: typically 30-80 kWh/m²/year heating
        assert 20 < parsed.heating_kwh_m2 < 100, \
            f"Heating {parsed.heating_kwh_m2:.1f} kWh/m² outside reasonable range"

    def test_floor_area_matches_expected(self, model_file, weather_file, energyplus_available, tmp_path):
        """Test that floor area matches Sjostaden specs (2240 m²)."""
        output_dir = tmp_path / "output"

        result = run_simulation(
            idf_path=model_file,
            weather_path=weather_file,
            output_dir=output_dir,
            parse_results=True
        )

        assert result.success
        parsed = result.parsed_results

        # Sjostaden: 7 floors × 320 m² = 2240 m²
        assert 2000 < parsed.floor_area_m2 < 2500, \
            f"Floor area {parsed.floor_area_m2:.0f} m² doesn't match expected ~2240 m²"


class TestSimulationErrorHandling:
    """Test error handling in simulations."""

    def test_invalid_idf_detected(self, weather_file, energyplus_available, tmp_path):
        """Test that invalid IDF is detected."""
        # Create an invalid IDF
        bad_idf = tmp_path / "bad_model.idf"
        bad_idf.write_text("This is not a valid IDF file")

        output_dir = tmp_path / "output"

        runner = SimulationRunner()
        result = runner.run(
            idf_path=bad_idf,
            weather_path=weather_file,
            output_dir=output_dir,
            timeout_seconds=60
        )

        # Should fail due to invalid IDF
        assert not result.success
        assert result.error_message is not None


class TestResultsParserIntegration:
    """Test results parser with real output files."""

    def test_parse_existing_output(self, model_file, weather_file, energyplus_available, tmp_path):
        """Test parsing results from completed simulation."""
        output_dir = tmp_path / "output"

        # Run simulation
        runner = SimulationRunner()
        sim_result = runner.run(model_file, weather_file, output_dir)
        assert sim_result.success

        # Parse results separately
        parser = ResultsParser()
        results = parser.parse(output_dir)

        assert results is not None
        assert results.total_site_energy_kwh > 0

        # Verify to_dict works
        results_dict = results.to_dict()
        assert "energy_kwh" in results_dict
        assert "intensity_kwh_m2" in results_dict


class TestCalibrationWorkflow:
    """Test calibration workflow with actual simulations."""

    def test_calibrator_initial_simulation(self, model_file, weather_file, energyplus_available, tmp_path):
        """Test that calibrator can run initial simulation."""
        output_dir = tmp_path / "calibration"

        calibrator = BaselineCalibrator()

        # Test with target close to expected result (~42 kWh/m²)
        # This should converge quickly since target is close to baseline
        result = calibrator.calibrate(
            idf_path=model_file,
            weather_path=weather_file,
            measured_heating_kwh_m2=42.0,  # Close to baseline
            output_dir=output_dir
        )

        assert isinstance(result, CalibrationResult)
        assert result.initial_kwh_m2 > 0
        # Should be close to target since we set target near baseline
        assert result.iterations <= 2

    def test_calibrator_reduces_heating(self, original_model_file, weather_file, energyplus_available, tmp_path):
        """Test that calibrator can reduce heating towards target."""
        output_dir = tmp_path / "calibration"

        calibrator = BaselineCalibrator()

        # Use original uncalibrated model (~42 kWh/m²) with target of 35 kWh/m²
        # This tests the calibration direction
        result = calibrator.calibrate(
            idf_path=original_model_file,
            weather_path=weather_file,
            measured_heating_kwh_m2=35.0,  # Lower target
            output_dir=output_dir
        )

        assert isinstance(result, CalibrationResult)
        # Calibration should attempt to reduce heating from ~42 towards 35
        assert result.calibrated_kwh_m2 < result.initial_kwh_m2

    def test_calibrator_creates_output_idf(self, model_file, weather_file, energyplus_available, tmp_path):
        """Test that calibrator creates calibrated IDF file."""
        output_dir = tmp_path / "calibration"

        calibrator = BaselineCalibrator()
        result = calibrator.calibrate(
            idf_path=model_file,
            weather_path=weather_file,
            measured_heating_kwh_m2=40.0,
            output_dir=output_dir
        )

        # Should create calibrated IDF
        assert result.calibrated_idf_path is not None
        assert result.calibrated_idf_path.exists()
        assert result.calibrated_idf_path.suffix == '.idf'

    def test_calibrator_parameter_bounds_respected(self, model_file, weather_file, energyplus_available, tmp_path):
        """Test that calibrator respects physical parameter bounds."""
        output_dir = tmp_path / "calibration"

        calibrator = BaselineCalibrator()
        result = calibrator.calibrate(
            idf_path=model_file,
            weather_path=weather_file,
            measured_heating_kwh_m2=25.0,  # Very aggressive target
            output_dir=output_dir
        )

        # Check parameters stay within bounds
        assert 0.02 <= result.adjusted_infiltration_ach <= 0.15
        assert 0.60 <= result.adjusted_heat_recovery <= 0.90
        assert 0.7 <= result.adjusted_window_u <= 1.5


class TestECMModifiersOnRealIDF:
    """Test ECM modifiers produce valid, runnable IDF files."""

    def test_air_sealing_produces_valid_idf(self, model_file, weather_file, energyplus_available, tmp_path):
        """Test air sealing ECM produces runnable IDF."""
        output_dir = tmp_path / "ecm_output"
        output_dir.mkdir()

        # Apply air sealing using apply_single
        modifier = IDFModifier()
        modified_idf = modifier.apply_single(
            baseline_idf=model_file,
            ecm_id='air_sealing',
            params={'reduction_factor': 0.5},
            output_dir=output_dir,
            output_name='air_sealed.idf'
        )

        runner = SimulationRunner()
        result = runner.run(modified_idf, weather_file, output_dir / "sim")

        assert result.success, f"Air sealing ECM produced invalid IDF: {result.error_message}"

    def test_window_upgrade_produces_valid_idf(self, model_file, weather_file, energyplus_available, tmp_path):
        """Test window upgrade ECM produces runnable IDF."""
        output_dir = tmp_path / "ecm_output"
        output_dir.mkdir()

        modifier = IDFModifier()
        modified_idf = modifier.apply_single(
            baseline_idf=model_file,
            ecm_id='window_replacement',
            params={'u_value': 0.8, 'shgc': 0.5},
            output_dir=output_dir,
            output_name='window_upgrade.idf'
        )

        runner = SimulationRunner()
        result = runner.run(modified_idf, weather_file, output_dir / "sim")

        assert result.success, f"Window ECM produced invalid IDF: {result.error_message}"

    def test_led_lighting_produces_valid_idf(self, model_file, weather_file, energyplus_available, tmp_path):
        """Test LED lighting ECM produces runnable IDF."""
        output_dir = tmp_path / "ecm_output"
        output_dir.mkdir()

        modifier = IDFModifier()
        modified_idf = modifier.apply_single(
            baseline_idf=model_file,
            ecm_id='led_lighting',
            params={'watts_per_m2': 6.0},
            output_dir=output_dir,
            output_name='led_lighting.idf'
        )

        runner = SimulationRunner()
        result = runner.run(modified_idf, weather_file, output_dir / "sim")

        assert result.success, f"LED ECM produced invalid IDF: {result.error_message}"

    def test_air_sealing_reduces_heating(self, model_file, weather_file, energyplus_available, tmp_path):
        """Test that air sealing actually reduces heating demand."""
        output_dir = tmp_path / "ecm_compare"
        output_dir.mkdir()

        # Run baseline
        runner = SimulationRunner()
        baseline_result = runner.run_and_parse(model_file, weather_file, output_dir / "baseline")
        assert baseline_result.success

        # Apply air sealing
        modifier = IDFModifier()
        modified_idf = modifier.apply_single(
            baseline_idf=model_file,
            ecm_id='air_sealing',
            params={'reduction_factor': 0.5},
            output_dir=output_dir,
            output_name='air_sealed.idf'
        )

        ecm_result = runner.run_and_parse(modified_idf, weather_file, output_dir / "ecm")
        assert ecm_result.success

        # Air sealing should reduce heating
        baseline_heating = baseline_result.parsed_results.heating_kwh_m2
        ecm_heating = ecm_result.parsed_results.heating_kwh_m2

        assert ecm_heating < baseline_heating, \
            f"Air sealing should reduce heating: {ecm_heating:.1f} >= {baseline_heating:.1f}"

    def test_multiple_ecms_sequentially(self, model_file, weather_file, energyplus_available, tmp_path):
        """Test that multiple ECMs can be applied sequentially."""
        output_dir = tmp_path / "multi_ecm"
        output_dir.mkdir()

        modifier = IDFModifier()

        # Apply first ECM
        step1_idf = modifier.apply_single(
            baseline_idf=model_file,
            ecm_id='air_sealing',
            params={'reduction_factor': 0.5},
            output_dir=output_dir,
            output_name='step1.idf'
        )

        # Apply second ECM to result of first
        step2_idf = modifier.apply_single(
            baseline_idf=step1_idf,
            ecm_id='window_replacement',
            params={'u_value': 0.8, 'shgc': 0.5},
            output_dir=output_dir,
            output_name='step2.idf'
        )

        # Apply third ECM
        final_idf = modifier.apply_single(
            baseline_idf=step2_idf,
            ecm_id='led_lighting',
            params={'watts_per_m2': 6.0},
            output_dir=output_dir,
            output_name='final.idf'
        )

        # Should still produce valid IDF
        runner = SimulationRunner()
        result = runner.run(final_idf, weather_file, output_dir / "sim")

        assert result.success, f"Multiple ECMs produced invalid IDF: {result.error_message}"
