"""Tests for PV potential calculator."""
import pytest
from src.geometry.pv_potential import (
    PVPotentialCalculator,
    PVPotential,
    calculate_pv_potential,
)


class TestPVPotentialCalculator:
    """Test PVPotentialCalculator class."""

    def test_init_stockholm(self):
        """Test initialization with Stockholm latitude."""
        calc = PVPotentialCalculator(latitude=59.3)
        assert calc.latitude == 59.3
        assert calc.base_irradiance == 950  # Stockholm value

    def test_init_malmo(self):
        """Test initialization with Malmö latitude."""
        calc = PVPotentialCalculator(latitude=55.0)
        assert calc.base_irradiance == 1050  # Higher irradiance in south

    def test_init_kiruna(self):
        """Test initialization with Kiruna latitude."""
        calc = PVPotentialCalculator(latitude=67.8)
        assert calc.base_irradiance == 850  # Lower irradiance in north


class TestIrradianceInterpolation:
    """Test irradiance interpolation."""

    def test_exact_latitude_match(self):
        """Test when latitude matches exactly."""
        calc = PVPotentialCalculator(latitude=59.3)
        assert calc.base_irradiance == 950

    def test_interpolation_between_cities(self):
        """Test interpolation between known latitudes."""
        # Between Stockholm (59.3, 950) and Gothenburg (57.7, 1000)
        calc = PVPotentialCalculator(latitude=58.5)
        assert 950 < calc.base_irradiance < 1000

    def test_below_minimum_latitude(self):
        """Test latitude below minimum uses lowest entry."""
        calc = PVPotentialCalculator(latitude=50.0)
        assert calc.base_irradiance == 1050  # Malmö value

    def test_above_maximum_latitude(self):
        """Test latitude above maximum uses highest entry."""
        calc = PVPotentialCalculator(latitude=70.0)
        assert calc.base_irradiance == 850  # Kiruna value


class TestCalculate:
    """Test main calculate method."""

    def test_flat_roof_basic(self):
        """Test basic flat roof calculation."""
        calc = PVPotentialCalculator(latitude=59.3)
        result = calc.calculate(roof_area_m2=320, roof_type='flat')

        assert isinstance(result, PVPotential)
        assert result.available_roof_area_m2 == 320 * 0.70  # 70% utilization
        assert result.max_capacity_kwp > 0
        assert result.effective_annual_yield_kwh > 0

    def test_pitched_roof_higher_utilization(self):
        """Test pitched roof has higher utilization."""
        calc = PVPotentialCalculator(latitude=59.3)

        flat = calc.calculate(roof_area_m2=320, roof_type='flat')
        pitched = calc.calculate(
            roof_area_m2=320,
            roof_type='pitched',
            roof_slope_deg=30,
            roof_azimuth_deg=180
        )

        assert flat.roof_utilization_factor == 0.70
        assert pitched.roof_utilization_factor == 0.85

    def test_optimal_tilt_calculation(self):
        """Test optimal tilt is roughly latitude - 12 degrees."""
        calc = PVPotentialCalculator(latitude=59.3)
        result = calc.calculate(roof_area_m2=320)

        expected_tilt = 59.3 - 12.0
        assert abs(result.optimal_tilt_deg - expected_tilt) < 1.0

    def test_optimal_azimuth_is_south(self):
        """Test optimal azimuth is 180 (south)."""
        calc = PVPotentialCalculator(latitude=59.3)
        result = calc.calculate(roof_area_m2=320)

        assert result.optimal_azimuth_deg == 180.0

    def test_capacity_proportional_to_area(self):
        """Test capacity scales with roof area."""
        calc = PVPotentialCalculator(latitude=59.3)

        small = calc.calculate(roof_area_m2=100)
        large = calc.calculate(roof_area_m2=400)

        assert large.max_capacity_kwp == pytest.approx(small.max_capacity_kwp * 4, rel=0.01)

    def test_system_losses_applied(self):
        """Test system losses are included."""
        calc = PVPotentialCalculator(latitude=59.3)
        result = calc.calculate(roof_area_m2=320)

        assert result.inverter_losses == 0.04
        assert result.soiling_losses == 0.02


class TestOrientationFactor:
    """Test orientation factor calculation."""

    def test_south_facing_optimal(self):
        """Test south-facing (180) has highest factor."""
        calc = PVPotentialCalculator(latitude=59.3)

        south = calc._orientation_factor(tilt=47.0, azimuth=180)
        southeast = calc._orientation_factor(tilt=47.0, azimuth=135)

        assert south > southeast

    def test_north_facing_reduced(self):
        """Test north-facing has significantly reduced factor."""
        calc = PVPotentialCalculator(latitude=59.3)

        south = calc._orientation_factor(tilt=47.0, azimuth=180)
        north = calc._orientation_factor(tilt=47.0, azimuth=0)

        assert north < south * 0.6  # North should be much worse

    def test_east_west_symmetric(self):
        """Test east and west have similar factors."""
        calc = PVPotentialCalculator(latitude=59.3)

        east = calc._orientation_factor(tilt=30.0, azimuth=90)
        west = calc._orientation_factor(tilt=30.0, azimuth=270)

        assert east == pytest.approx(west, rel=0.01)

    def test_optimal_tilt_best(self):
        """Test optimal tilt has highest factor."""
        calc = PVPotentialCalculator(latitude=59.3)
        optimal_tilt = calc.latitude - 12.0

        optimal = calc._orientation_factor(tilt=optimal_tilt, azimuth=180)
        flat = calc._orientation_factor(tilt=0, azimuth=180)
        vertical = calc._orientation_factor(tilt=90, azimuth=180)

        assert optimal > flat
        assert optimal > vertical


class TestShadingLoss:
    """Test shading loss calculation."""

    def test_no_shading_zero_loss(self):
        """Test no shading objects gives zero loss."""
        calc = PVPotentialCalculator(latitude=59.3)

        loss = calc._calculate_shading_loss([])
        assert loss == 0.0

        loss = calc._calculate_shading_loss(None)
        assert loss == 0.0

    def test_south_facing_obstruction_highest_impact(self):
        """Test south-facing obstructions have most impact."""
        calc = PVPotentialCalculator(latitude=59.3)

        south_obj = {'height_m': 10, 'distance_m': 10, 'width_m': 10, 'azimuth_deg': 180}
        north_obj = {'height_m': 10, 'distance_m': 10, 'width_m': 10, 'azimuth_deg': 0}

        south_loss = calc._calculate_shading_loss([south_obj])
        north_loss = calc._calculate_shading_loss([north_obj])

        assert south_loss > north_loss

    def test_closer_objects_more_shading(self):
        """Test closer objects cause more shading."""
        calc = PVPotentialCalculator(latitude=59.3)

        close = {'height_m': 10, 'distance_m': 5, 'width_m': 10, 'azimuth_deg': 180}
        far = {'height_m': 10, 'distance_m': 20, 'width_m': 10, 'azimuth_deg': 180}

        close_loss = calc._calculate_shading_loss([close])
        far_loss = calc._calculate_shading_loss([far])

        assert close_loss > far_loss

    def test_taller_objects_more_shading(self):
        """Test taller objects cause more shading."""
        calc = PVPotentialCalculator(latitude=59.3)

        tall = {'height_m': 20, 'distance_m': 10, 'width_m': 10, 'azimuth_deg': 180}
        short = {'height_m': 5, 'distance_m': 10, 'width_m': 10, 'azimuth_deg': 180}

        tall_loss = calc._calculate_shading_loss([tall])
        short_loss = calc._calculate_shading_loss([short])

        assert tall_loss > short_loss

    def test_tree_reduced_impact(self):
        """Test trees have 30% reduced impact vs buildings."""
        calc = PVPotentialCalculator(latitude=59.3)

        building = {'type': 'building', 'height_m': 10, 'distance_m': 10, 'width_m': 10, 'azimuth_deg': 180}
        tree = {'type': 'tree', 'height_m': 10, 'distance_m': 10, 'width_m': 10, 'azimuth_deg': 180}

        building_loss = calc._calculate_shading_loss([building])
        tree_loss = calc._calculate_shading_loss([tree])

        assert tree_loss == pytest.approx(building_loss * 0.7, rel=0.01)

    def test_max_shading_capped(self):
        """Test shading loss capped at 50%."""
        calc = PVPotentialCalculator(latitude=59.3)

        # Many large obstructions
        obstructions = [
            {'height_m': 30, 'distance_m': 5, 'width_m': 20, 'azimuth_deg': 180},
            {'height_m': 30, 'distance_m': 5, 'width_m': 20, 'azimuth_deg': 150},
            {'height_m': 30, 'distance_m': 5, 'width_m': 20, 'azimuth_deg': 210},
        ]

        loss = calc._calculate_shading_loss(obstructions)
        assert loss <= 0.5


class TestConvenienceFunction:
    """Test the calculate_pv_potential convenience function."""

    def test_basic_usage(self):
        """Test basic convenience function usage."""
        result = calculate_pv_potential(
            roof_area_m2=320,
            latitude=59.3,
            roof_type='flat'
        )

        assert isinstance(result, PVPotential)
        assert result.available_roof_area_m2 > 0
        assert result.max_capacity_kwp > 0

    def test_with_all_parameters(self):
        """Test with all parameters specified."""
        result = calculate_pv_potential(
            roof_area_m2=320,
            latitude=57.7,
            roof_type='pitched',
            roof_slope_deg=30,
            roof_azimuth_deg=190,
            shading_objects=[{'height_m': 5, 'distance_m': 20, 'width_m': 10, 'azimuth_deg': 180}]
        )

        assert result.shading_loss_factor > 0


class TestRealisticScenarios:
    """Test with realistic Swedish building scenarios."""

    def test_sjostaden_flat_roof(self):
        """Test Sjostaden 7-floor building roof (~320 m2)."""
        result = calculate_pv_potential(
            roof_area_m2=320,
            latitude=59.3,  # Stockholm
            roof_type='flat'
        )

        # Expected: ~45 kWp capacity on 224m² usable area
        assert 40 < result.max_capacity_kwp < 50
        # Expected: ~900-1000 kWh/kWp/year in Stockholm
        assert 800 < result.annual_yield_kwh_per_kwp < 1100
        # Expected: ~35-45 MWh/year total
        assert 35000 < result.effective_annual_yield_kwh < 50000

    def test_gothenburg_pitched_roof(self):
        """Test Gothenburg building with pitched roof."""
        result = calculate_pv_potential(
            roof_area_m2=150,
            latitude=57.7,  # Gothenburg
            roof_type='pitched',
            roof_slope_deg=35,
            roof_azimuth_deg=180  # South-facing
        )

        # Higher irradiance in Gothenburg than Stockholm
        assert result.annual_yield_kwh_per_kwp > 950

    def test_malmo_south_facing(self):
        """Test Malmö with optimal south-facing setup."""
        result = calculate_pv_potential(
            roof_area_m2=200,
            latitude=55.0,  # Malmö - southernmost Sweden
            roof_type='pitched',
            roof_slope_deg=43,  # Near optimal for 55° latitude
            roof_azimuth_deg=180
        )

        # Malmö has highest irradiance in Sweden
        calc = PVPotentialCalculator(latitude=55.0)
        assert calc.base_irradiance == 1050
