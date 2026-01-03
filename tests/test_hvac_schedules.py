"""
Tests for HVAC and Schedules modules (2026 Roadmap Phase 1-2).

Tests:
- HVAC system selection from building data
- Occupancy pattern generation
- EnergyPlus IDF snippet generation
- Integration with generator_v2
"""

import pytest
from pathlib import Path

# =============================================================================
# HVAC MODULE TESTS
# =============================================================================

class TestSwedishHVACSystems:
    """Test Swedish HVAC system definitions."""

    def test_hvac_system_enum_values(self):
        """All Swedish HVAC systems are defined."""
        from src.hvac import SwedishHVACSystem

        systems = list(SwedishHVACSystem)
        assert len(systems) >= 6

        # Key systems
        assert SwedishHVACSystem.DISTRICT_HEATING in systems
        assert SwedishHVACSystem.GROUND_SOURCE_HP in systems
        assert SwedishHVACSystem.EXHAUST_AIR_HP in systems
        assert SwedishHVACSystem.AIR_SOURCE_HP in systems
        assert SwedishHVACSystem.DIRECT_ELECTRIC in systems

    def test_ventilation_system_enum_values(self):
        """All Swedish ventilation systems are defined."""
        from src.hvac import VentilationSystem

        systems = list(VentilationSystem)
        assert VentilationSystem.FTX in systems  # Balanced with HR
        assert VentilationSystem.F_SYSTEM in systems  # Exhaust only
        assert VentilationSystem.NATURAL in systems  # Natural draft

    def test_hvac_selection_dataclass(self):
        """HVACSelection dataclass has required fields."""
        from src.hvac import HVACSelection, SwedishHVACSystem, VentilationSystem

        selection = HVACSelection(
            primary_heating=SwedishHVACSystem.DISTRICT_HEATING,
            ventilation=VentilationSystem.FTX,
            heat_recovery_eff=0.80,
            detected_from="test",
            confidence=0.95,
        )

        assert selection.primary_heating == SwedishHVACSystem.DISTRICT_HEATING
        assert selection.ventilation == VentilationSystem.FTX
        assert selection.heat_recovery_eff == 0.80
        assert selection.confidence == 0.95


class TestHVACSelector:
    """Test HVAC auto-selection from building data."""

    def test_hvac_from_era_pre_1975(self):
        """Pre-1975 buildings default to district heating."""
        from src.hvac import select_hvac_system

        selection = select_hvac_system(construction_year=1970)

        # Most pre-1975 MFH had district heating
        assert selection.primary_heating.value == "district_heating"
        assert selection.detected_from == "era_inference"

    def test_hvac_from_era_post_2000(self):
        """Post-2000 buildings should have FTX."""
        from src.hvac import select_hvac_system, VentilationSystem

        selection = select_hvac_system(construction_year=2005)

        # Should have some heat recovery
        assert selection.heat_recovery_eff >= 0.7

    def test_hvac_default_when_no_data(self):
        """Default to district heating when no data available."""
        from src.hvac import select_hvac_system, SwedishHVACSystem

        selection = select_hvac_system()

        assert selection.primary_heating == SwedishHVACSystem.DISTRICT_HEATING
        assert selection.confidence == 0.5
        assert selection.detected_from == "default"


class TestHVACIDFGeneration:
    """Test EnergyPlus IDF snippet generation."""

    def test_generate_district_heating_idf(self):
        """Generate district heating IDF snippet."""
        from src.hvac import generate_hvac_idf, SwedishHVACSystem

        idf_snippet = generate_hvac_idf(
            system_type=SwedishHVACSystem.DISTRICT_HEATING,
            zone_names=["Floor1", "Floor2"],
            design_heating_load_w=50000,
        )

        assert "DistrictHeating" in idf_snippet or "District" in idf_snippet
        assert "Floor1" in idf_snippet
        assert "Floor2" in idf_snippet

    def test_generate_heat_pump_idf(self):
        """Generate heat pump IDF snippet."""
        from src.hvac import generate_hvac_idf, SwedishHVACSystem

        idf_snippet = generate_hvac_idf(
            system_type=SwedishHVACSystem.GROUND_SOURCE_HP,
            zone_names=["Zone1"],
            design_heating_load_w=30000,
        )

        assert "HeatPump" in idf_snippet or "GroundSource" in idf_snippet
        assert "Zone1" in idf_snippet


# =============================================================================
# SCHEDULES MODULE TESTS
# =============================================================================

class TestSwedishOccupancyPatterns:
    """Test Swedish occupancy pattern definitions."""

    def test_occupant_profile_enum(self):
        """All occupant profiles are defined."""
        from src.schedules import OccupantProfile

        profiles = list(OccupantProfile)
        assert len(profiles) >= 6

        # Residential
        assert OccupantProfile.FAMILIES in profiles
        assert OccupantProfile.ELDERLY in profiles
        assert OccupantProfile.STUDENTS in profiles

        # Commercial
        assert OccupantProfile.OFFICE_STANDARD in profiles
        assert OccupantProfile.RETAIL in profiles

    def test_hourly_profile_24_values(self):
        """Hourly profile has exactly 24 values."""
        from src.schedules import HourlyProfile

        profile = HourlyProfile([0.5] * 24)
        assert len(profile.values) == 24

        with pytest.raises(ValueError):
            HourlyProfile([0.5] * 23)  # Should fail

    def test_hourly_profile_clamps_values(self):
        """Hourly profile clamps values to 0-1."""
        from src.schedules import HourlyProfile

        profile = HourlyProfile([1.5, -0.5] + [0.5] * 22)
        assert profile.values[0] == 1.0  # Clamped from 1.5
        assert profile.values[1] == 0.0  # Clamped from -0.5

    def test_residential_patterns_available(self):
        """Residential patterns are defined."""
        from src.schedules import RESIDENTIAL_PATTERNS, OccupantProfile

        assert OccupantProfile.FAMILIES in RESIDENTIAL_PATTERNS
        assert OccupantProfile.ELDERLY in RESIDENTIAL_PATTERNS
        assert OccupantProfile.STUDENTS in RESIDENTIAL_PATTERNS

    def test_commercial_patterns_available(self):
        """Commercial patterns are defined."""
        from src.schedules import COMMERCIAL_PATTERNS, OccupantProfile

        assert OccupantProfile.OFFICE_STANDARD in COMMERCIAL_PATTERNS
        assert OccupantProfile.RETAIL in COMMERCIAL_PATTERNS
        assert OccupantProfile.RESTAURANT in COMMERCIAL_PATTERNS


class TestPatternSelection:
    """Test pattern selection functions."""

    def test_get_pattern_for_residential(self):
        """Get pattern for residential building."""
        from src.schedules import get_pattern_for_building

        pattern = get_pattern_for_building(
            building_type="residential",
            occupant_profile="families",
        )

        assert "Families" in pattern.name or "families" in pattern.name.lower()
        assert pattern.occupant_density_m2_person > 0

    def test_get_pattern_for_commercial(self):
        """Get pattern for commercial building."""
        from src.schedules import get_pattern_for_building

        pattern = get_pattern_for_building(
            building_type="commercial",
            commercial_type="office",
        )

        assert "Office" in pattern.name or "office" in pattern.name.lower()
        # Commercial has lower density (more people per m²)
        assert pattern.occupant_density_m2_person < 40

    def test_get_default_pattern(self):
        """Get default mixed residential pattern."""
        from src.schedules import get_pattern_for_building

        pattern = get_pattern_for_building()

        assert pattern is not None
        assert pattern.occupant_density_m2_person > 0


class TestScheduleGeneration:
    """Test EnergyPlus schedule IDF generation."""

    def test_generate_schedule_idf(self):
        """Generate schedule IDF from pattern."""
        from src.schedules import get_pattern_for_building, generate_schedule_idf

        pattern = get_pattern_for_building(
            building_type="residential",
            occupant_profile="families",
        )

        idf_snippet = generate_schedule_idf(pattern, zone_name="Zone1")

        assert "Schedule" in idf_snippet
        assert "Zone1" in idf_snippet
        assert "Occupancy" in idf_snippet or "Fraction" in idf_snippet


class TestBlendPatterns:
    """Test pattern blending for mixed buildings."""

    def test_blend_two_patterns(self):
        """Blend two patterns with weights."""
        from src.schedules import RESIDENTIAL_PATTERNS, OccupantProfile, blend_patterns

        patterns_weights = [
            (RESIDENTIAL_PATTERNS[OccupantProfile.FAMILIES], 0.6),
            (RESIDENTIAL_PATTERNS[OccupantProfile.ELDERLY], 0.4),
        ]

        blended = blend_patterns(patterns_weights)

        assert blended is not None
        assert blended.occupant_density_m2_person > 0
        # Blended density should be between the two
        families_density = RESIDENTIAL_PATTERNS[OccupantProfile.FAMILIES].occupant_density_m2_person
        elderly_density = RESIDENTIAL_PATTERNS[OccupantProfile.ELDERLY].occupant_density_m2_person
        assert min(families_density, elderly_density) <= blended.occupant_density_m2_person <= max(families_density, elderly_density)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestGeneratorIntegration:
    """Test integration with generator_v2."""

    def test_generate_enhanced_imports(self):
        """generate_enhanced function is importable."""
        from src.baseline import generate_enhanced
        assert generate_enhanced is not None

    def test_hvac_module_importable(self):
        """HVAC module is importable from baseline."""
        from src.hvac import (
            SwedishHVACSystem,
            VentilationSystem,
            HVACSelection,
            select_hvac_system,
            generate_hvac_idf,
        )
        assert SwedishHVACSystem is not None
        assert select_hvac_system is not None

    def test_schedules_module_importable(self):
        """Schedules module is importable."""
        from src.schedules import (
            OccupantProfile,
            get_pattern_for_building,
            generate_schedule_idf,
            RESIDENTIAL_PATTERNS,
            COMMERCIAL_PATTERNS,
        )
        assert OccupantProfile is not None
        assert get_pattern_for_building is not None


class TestSeasonalAdjustments:
    """Test Swedish seasonal adjustments."""

    def test_summer_vacation_adjustment(self):
        """July (industrisemester) has lower occupancy."""
        from src.schedules import SEASONAL_ADJUSTMENTS, SeasonalPattern, OccupantProfile

        if OccupantProfile.FAMILIES in SEASONAL_ADJUSTMENTS:
            adjustments = SEASONAL_ADJUSTMENTS[OccupantProfile.FAMILIES]
            summer_factor = adjustments.get(SeasonalPattern.SUMMER_VACATION, 1.0)
            assert summer_factor < 1.0  # Reduced occupancy in July

    def test_christmas_adjustment(self):
        """Christmas has different occupancy."""
        from src.schedules import SEASONAL_ADJUSTMENTS, SeasonalPattern, OccupantProfile

        if OccupantProfile.FAMILIES in SEASONAL_ADJUSTMENTS:
            adjustments = SEASONAL_ADJUSTMENTS[OccupantProfile.FAMILIES]
            christmas_factor = adjustments.get(SeasonalPattern.CHRISTMAS, 1.0)
            # Families tend to be home more at Christmas
            assert christmas_factor >= 1.0


class TestASHRAEMetrics:
    """Test ASHRAE metrics are available (Phase 3)."""

    def test_calibration_metrics_import(self):
        """CalibrationMetrics is importable."""
        from src.calibration.metrics import CalibrationMetrics
        assert CalibrationMetrics is not None

    def test_metrics_from_annual_data(self):
        """Compute metrics from annual data."""
        from src.calibration.metrics import CalibrationMetrics

        metrics = CalibrationMetrics.from_annual_data(
            measured_kwh_m2=100.0,
            simulated_kwh_m2=95.0,
        )

        assert metrics.annual_error_percent == pytest.approx(5.0, abs=0.1)
        assert metrics.data_resolution == "annual"

    def test_metrics_from_monthly_data(self):
        """Compute metrics from monthly data."""
        from src.calibration.metrics import CalibrationMetrics

        measured = [10, 12, 10, 8, 6, 4, 3, 4, 6, 8, 10, 12]
        simulated = [9.5, 11.5, 9.8, 7.8, 5.9, 3.9, 2.9, 3.9, 5.8, 7.9, 9.7, 11.8]

        metrics = CalibrationMetrics.from_monthly_data(measured, simulated)

        assert metrics.data_resolution == "monthly"
        assert metrics.n_points == 12
        assert abs(metrics.nmbe) < 10  # Should be close


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
