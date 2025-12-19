"""
Tests for simulation results parser.

Tests:
- CSV parsing (eplustbl.csv)
- Floor area extraction
- End use energy extraction
- Intensity calculations
"""

import pytest
from pathlib import Path

from src.simulation.results import ResultsParser, AnnualResults, parse_results


class TestResultsParser:
    """Tests for ResultsParser class."""

    def test_parse_sample_csv(self, sample_csv_file):
        """Test parsing sample CSV content."""
        parser = ResultsParser()
        results = parser.parse(sample_csv_file)

        assert results is not None
        assert isinstance(results, AnnualResults)

    def test_floor_area_extraction(self, sample_csv_file):
        """Test floor area is correctly extracted."""
        parser = ResultsParser()
        results = parser.parse(sample_csv_file)

        assert results.floor_area_m2 == 2240.0

    def test_heating_energy_extraction(self, sample_csv_file):
        """Test heating energy is correctly extracted."""
        parser = ResultsParser()
        results = parser.parse(sample_csv_file)

        assert results.heating_kwh == 93765.0

    def test_lighting_energy_extraction(self, sample_csv_file):
        """Test lighting energy is correctly extracted."""
        parser = ResultsParser()
        results = parser.parse(sample_csv_file)

        assert results.lighting_kwh == 41207.0

    def test_equipment_energy_extraction(self, sample_csv_file):
        """Test equipment energy is correctly extracted."""
        parser = ResultsParser()
        results = parser.parse(sample_csv_file)

        assert results.equipment_kwh == 59358.0

    def test_intensity_calculations(self, sample_csv_file):
        """Test energy intensities are calculated correctly."""
        parser = ResultsParser()
        results = parser.parse(sample_csv_file)

        # Heating: 93765 / 2240 = 41.86 kWh/m²
        assert abs(results.heating_kwh_m2 - 41.86) < 0.1

        # Lighting: 41207 / 2240 = 18.40 kWh/m²
        assert abs(results.lighting_kwh_m2 - 18.40) < 0.1

    def test_parse_nonexistent_directory(self, temp_dir):
        """Test parsing nonexistent directory returns None."""
        parser = ResultsParser()
        results = parser.parse(temp_dir / "nonexistent")

        assert results is None

    def test_parse_empty_directory(self, temp_dir):
        """Test parsing empty directory returns None."""
        parser = ResultsParser()
        results = parser.parse(temp_dir)

        assert results is None

    def test_to_dict(self, sample_csv_file):
        """Test conversion to dictionary."""
        parser = ResultsParser()
        results = parser.parse(sample_csv_file)
        result_dict = results.to_dict()

        assert 'energy_kwh' in result_dict
        assert 'intensity_kwh_m2' in result_dict
        assert result_dict['energy_kwh']['heating'] == 93765.0
        assert result_dict['floor_area_m2'] == 2240.0


class TestParseResultsFunction:
    """Tests for convenience function."""

    def test_parse_results_function(self, sample_csv_file):
        """Test convenience function."""
        results = parse_results(sample_csv_file)

        assert results is not None
        assert results.heating_kwh == 93765.0


class TestRealOutputParsing:
    """Tests using real simulation output (if available)."""

    def test_parse_real_output(self, output_dir):
        """Test parsing actual EnergyPlus output."""
        if not (output_dir / "eplustbl.csv").exists():
            pytest.skip("No real output available")

        parser = ResultsParser()
        results = parser.parse(output_dir)

        assert results is not None
        assert results.floor_area_m2 > 0
        assert results.heating_kwh > 0
        # Sjostaden should be around 42 kWh/m²
        assert 30 < results.heating_kwh_m2 < 60


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_floor_area_handling(self, temp_dir):
        """Test handling of zero floor area."""
        # Create CSV with zero floor area
        csv_content = '''
,Total Building Area,0.00
End Uses
,,Electricity [kWh],Natural Gas [kWh],Gasoline [kWh],Diesel [kWh],Coal [kWh],Fuel Oil No 1 [kWh],Fuel Oil No 2 [kWh],Propane [kWh],Other Fuel 1 [kWh],Other Fuel 2 [kWh],District Cooling [kWh],District Heating Water [kWh],District Heating Steam [kWh],Water [m3]
,Heating,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1000.00,0.00,0.00
'''
        csv_path = temp_dir / "eplustbl.csv"
        csv_path.write_text(csv_content)

        parser = ResultsParser()
        results = parser.parse(temp_dir)

        # Should handle gracefully (use 1.0 as divisor)
        if results:
            assert results.heating_kwh == 1000.0

    def test_missing_end_uses_section(self, temp_dir):
        """Test handling of missing end uses."""
        csv_content = '''
,Total Building Area,2240.00
'''
        csv_path = temp_dir / "eplustbl.csv"
        csv_path.write_text(csv_content)

        parser = ResultsParser()
        results = parser.parse(temp_dir)

        # Should return results with zero energy
        if results:
            assert results.heating_kwh == 0.0
