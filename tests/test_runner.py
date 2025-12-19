"""Tests for SimulationRunner file validation."""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.simulation.runner import SimulationRunner, SimulationResult


class TestFileValidation:
    """Test file existence validation."""

    def test_missing_idf_raises_error(self, tmp_path):
        """Test that missing IDF file raises FileNotFoundError."""
        weather_file = tmp_path / "weather.epw"
        weather_file.write_text("dummy weather")

        with patch.object(SimulationRunner, '_find_energyplus', return_value='/usr/bin/energyplus'):
            runner = SimulationRunner()

            with pytest.raises(FileNotFoundError, match="IDF file not found"):
                runner._validate_input_files(
                    Path("/nonexistent/model.idf"),
                    weather_file
                )

    def test_missing_weather_raises_error(self, tmp_path):
        """Test that missing weather file raises FileNotFoundError."""
        idf_file = tmp_path / "model.idf"
        idf_file.write_text("dummy idf")

        with patch.object(SimulationRunner, '_find_energyplus', return_value='/usr/bin/energyplus'):
            runner = SimulationRunner()

            with pytest.raises(FileNotFoundError, match="Weather file not found"):
                runner._validate_input_files(
                    idf_file,
                    Path("/nonexistent/weather.epw")
                )

    def test_idf_directory_raises_error(self, tmp_path):
        """Test that passing a directory as IDF raises ValueError."""
        idf_dir = tmp_path / "model_dir"
        idf_dir.mkdir()
        weather_file = tmp_path / "weather.epw"
        weather_file.write_text("dummy")

        with patch.object(SimulationRunner, '_find_energyplus', return_value='/usr/bin/energyplus'):
            runner = SimulationRunner()

            with pytest.raises(ValueError, match="not a file"):
                runner._validate_input_files(idf_dir, weather_file)

    def test_weather_directory_raises_error(self, tmp_path):
        """Test that passing a directory as weather raises ValueError."""
        idf_file = tmp_path / "model.idf"
        idf_file.write_text("dummy")
        weather_dir = tmp_path / "weather_dir"
        weather_dir.mkdir()

        with patch.object(SimulationRunner, '_find_energyplus', return_value='/usr/bin/energyplus'):
            runner = SimulationRunner()

            with pytest.raises(ValueError, match="not a file"):
                runner._validate_input_files(idf_file, weather_dir)

    def test_valid_files_pass(self, tmp_path):
        """Test that valid files pass validation."""
        idf_file = tmp_path / "model.idf"
        idf_file.write_text("dummy idf")
        weather_file = tmp_path / "weather.epw"
        weather_file.write_text("dummy weather")

        with patch.object(SimulationRunner, '_find_energyplus', return_value='/usr/bin/energyplus'):
            runner = SimulationRunner()
            # Should not raise
            runner._validate_input_files(idf_file, weather_file)

    def test_imf_extension_accepted(self, tmp_path):
        """Test that .imf extension is accepted."""
        idf_file = tmp_path / "model.imf"
        idf_file.write_text("dummy imf")
        weather_file = tmp_path / "weather.epw"
        weather_file.write_text("dummy weather")

        with patch.object(SimulationRunner, '_find_energyplus', return_value='/usr/bin/energyplus'):
            runner = SimulationRunner()
            # Should not raise
            runner._validate_input_files(idf_file, weather_file)


class TestBatchValidation:
    """Test batch simulation validation."""

    def test_empty_batch_raises_error(self, tmp_path):
        """Test that empty IDF list raises ValueError."""
        weather_file = tmp_path / "weather.epw"
        weather_file.write_text("dummy")

        with patch.object(SimulationRunner, '_find_energyplus', return_value='/usr/bin/energyplus'):
            runner = SimulationRunner()

            with pytest.raises(ValueError, match="No IDF files provided"):
                runner.run_batch([], weather_file, tmp_path / "output")

    def test_batch_missing_weather_raises_error(self, tmp_path):
        """Test that missing weather file raises error in batch."""
        idf_file = tmp_path / "model.idf"
        idf_file.write_text("dummy")

        with patch.object(SimulationRunner, '_find_energyplus', return_value='/usr/bin/energyplus'):
            runner = SimulationRunner()

            with pytest.raises(FileNotFoundError, match="Weather file not found"):
                runner.run_batch(
                    [idf_file],
                    Path("/nonexistent/weather.epw"),
                    tmp_path / "output"
                )


class TestSimulationResult:
    """Test SimulationResult dataclass."""

    def test_success_result(self, tmp_path):
        """Test successful simulation result."""
        result = SimulationResult(
            idf_path=tmp_path / "model.idf",
            output_dir=tmp_path / "output",
            success=True,
            runtime_seconds=45.2,
        )

        assert result.success is True
        assert result.runtime_seconds == 45.2
        assert result.error_message is None
        assert result.parsed_results is None

    def test_failed_result(self, tmp_path):
        """Test failed simulation result."""
        result = SimulationResult(
            idf_path=tmp_path / "model.idf",
            output_dir=tmp_path / "output",
            success=False,
            runtime_seconds=5.0,
            error_message="Severe error in IDF"
        )

        assert result.success is False
        assert result.error_message == "Severe error in IDF"


class TestEnergyPlusDetection:
    """Test EnergyPlus executable detection."""

    def test_custom_path_used(self):
        """Test that custom EnergyPlus path is used."""
        runner = SimulationRunner(energyplus_path="/custom/path/energyplus")
        assert runner.energyplus_path == "/custom/path/energyplus"

    def test_auto_detect_raises_when_not_found(self):
        """Test that auto-detect raises error when E+ not found."""
        with patch('shutil.which', return_value=None):
            with patch('pathlib.Path.exists', return_value=False):
                with pytest.raises(RuntimeError, match="EnergyPlus not found"):
                    SimulationRunner()


class TestRunValidationIntegration:
    """Test that run() calls validation."""

    def test_run_validates_files(self, tmp_path):
        """Test that run() validates files before execution."""
        with patch.object(SimulationRunner, '_find_energyplus', return_value='/usr/bin/energyplus'):
            runner = SimulationRunner()

            # Should raise before even trying to run EnergyPlus
            with pytest.raises(FileNotFoundError, match="IDF file not found"):
                runner.run(
                    Path("/nonexistent/model.idf"),
                    tmp_path / "weather.epw",
                    tmp_path / "output"
                )
