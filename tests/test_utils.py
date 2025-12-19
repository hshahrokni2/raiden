"""
Tests for utility modules: logging, validation, retry.

Run with: pytest tests/test_utils.py -v
"""

import pytest
import time
from unittest.mock import Mock, patch

from src.utils import (
    get_logger,
    setup_logging,
    validate_address,
    validate_coordinates,
    validate_construction_year,
    validate_building_area,
    validate_energy_class,
    validate_facade_material,
    validate_heating_system,
    ValidationError,
    AddressComponents,
    retry_with_backoff,
    RetryConfig,
)


class TestLogging:
    """Tests for logging configuration."""

    def test_get_logger(self):
        """Test getting a logger instance."""
        logger = get_logger("test_module")
        assert logger is not None
        assert logger.name == "test_module"

    def test_logger_has_handlers(self):
        """Test that logging is set up with handlers."""
        setup_logging()
        logger = get_logger("test_handlers")
        # Root logger should have at least one handler
        import logging
        root = logging.getLogger()
        assert len(root.handlers) > 0


class TestAddressValidation:
    """Tests for address validation."""

    def test_valid_address_with_city(self):
        """Test validating a complete Swedish address."""
        result = validate_address("Bellmansgatan 16, Stockholm")
        assert result.street_name == "Bellmansgatan"
        assert result.street_number == "16"
        assert result.city == "Stockholm"

    def test_valid_address_without_city(self):
        """Test validating address without city."""
        result = validate_address("Storgatan 1")
        assert result.street_name == "Storgatan"
        assert result.street_number == "1"

    def test_empty_address_raises_error(self):
        """Test that empty address raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_address("")
        assert "empty" in str(exc_info.value).lower()

    def test_too_short_address(self):
        """Test that very short address raises error."""
        with pytest.raises(ValidationError):
            validate_address("AB")

    def test_numeric_only_address(self):
        """Test that numeric-only address raises error."""
        with pytest.raises(ValidationError):
            validate_address("12345")

    def test_validation_error_has_suggestions(self):
        """Test that ValidationError includes suggestions."""
        with pytest.raises(ValidationError) as exc_info:
            validate_address("")
        assert len(exc_info.value.suggestions) > 0


class TestCoordinateValidation:
    """Tests for coordinate validation."""

    def test_valid_stockholm_coordinates(self):
        """Test validating Stockholm coordinates."""
        lat, lon = validate_coordinates(59.3293, 18.0686)
        assert lat == 59.3293
        assert lon == 18.0686

    def test_invalid_latitude_range(self):
        """Test that out-of-range latitude raises error."""
        with pytest.raises(ValidationError):
            validate_coordinates(100.0, 18.0)

    def test_invalid_longitude_range(self):
        """Test that out-of-range longitude raises error."""
        with pytest.raises(ValidationError):
            validate_coordinates(59.0, 200.0)

    def test_coordinates_outside_sweden(self):
        """Test that coordinates outside Sweden raise error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_coordinates(40.0, -74.0, require_sweden=True)  # NYC
        assert "outside Sweden" in str(exc_info.value)

    def test_coordinates_outside_sweden_allowed(self):
        """Test that require_sweden=False allows non-Swedish coords."""
        lat, lon = validate_coordinates(40.0, -74.0, require_sweden=False)
        assert lat == 40.0


class TestConstructionYearValidation:
    """Tests for construction year validation."""

    def test_valid_year(self):
        """Test validating a reasonable construction year."""
        year = validate_construction_year(1965)
        assert year == 1965

    def test_too_old_year(self):
        """Test that year before 1800 raises error."""
        with pytest.raises(ValidationError):
            validate_construction_year(1500)

    def test_future_year(self):
        """Test that future year raises error."""
        with pytest.raises(ValidationError):
            validate_construction_year(2050)

    def test_string_conversion(self):
        """Test that string years are converted."""
        year = validate_construction_year("1990")
        assert year == 1990


class TestBuildingAreaValidation:
    """Tests for building area validation."""

    def test_valid_area(self):
        """Test validating a reasonable area."""
        area = validate_building_area(2500.0)
        assert area == 2500.0

    def test_too_small_area(self):
        """Test that very small area raises error."""
        with pytest.raises(ValidationError):
            validate_building_area(10.0)

    def test_too_large_area(self):
        """Test that very large area raises error."""
        with pytest.raises(ValidationError):
            validate_building_area(500000.0)


class TestEnergyClassValidation:
    """Tests for energy class validation."""

    def test_valid_energy_classes(self):
        """Test all valid energy classes."""
        for cls in ["A", "B", "C", "D", "E", "F", "G"]:
            assert validate_energy_class(cls) == cls

    def test_lowercase_normalized(self):
        """Test that lowercase is normalized."""
        assert validate_energy_class("c") == "C"

    def test_unknown_default(self):
        """Test empty returns Unknown."""
        assert validate_energy_class("") == "Unknown"

    def test_invalid_class(self):
        """Test invalid class raises error."""
        with pytest.raises(ValidationError):
            validate_energy_class("X")


class TestFacadeMaterialValidation:
    """Tests for facade material validation."""

    def test_valid_materials(self):
        """Test valid facade materials."""
        assert validate_facade_material("brick") == "brick"
        assert validate_facade_material("concrete") == "concrete"
        assert validate_facade_material("plaster") == "plaster"

    def test_swedish_names(self):
        """Test Swedish material names."""
        assert validate_facade_material("tegel") == "brick"
        assert validate_facade_material("betong") == "concrete"
        assert validate_facade_material("puts") == "plaster"

    def test_default_material(self):
        """Test empty returns default."""
        assert validate_facade_material("") == "concrete"


class TestHeatingSystemValidation:
    """Tests for heating system validation."""

    def test_valid_systems(self):
        """Test valid heating systems."""
        assert validate_heating_system("district") == "district"
        assert validate_heating_system("heat_pump") == "heat_pump"

    def test_swedish_names(self):
        """Test Swedish heating names."""
        assert validate_heating_system("fjärrvärme") == "district"
        assert validate_heating_system("värmepump") == "heat_pump"


class TestRetryLogic:
    """Tests for retry with backoff."""

    def test_successful_call_no_retry(self):
        """Test that successful calls don't retry."""
        call_count = 0

        @retry_with_backoff(max_retries=3)
        def success():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = success()
        assert result == "ok"
        assert call_count == 1

    def test_retry_on_connection_error(self):
        """Test retrying on ConnectionError."""
        call_count = 0

        @retry_with_backoff(
            config=RetryConfig(
                max_retries=2,
                base_delay=0.01,  # Fast for testing
            )
        )
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection failed")
            return "ok"

        result = flaky()
        assert result == "ok"
        assert call_count == 3

    def test_max_retries_exceeded(self):
        """Test that max retries are respected."""
        call_count = 0

        @retry_with_backoff(
            config=RetryConfig(
                max_retries=2,
                base_delay=0.01,
            )
        )
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Always fails")

        with pytest.raises(ConnectionError):
            always_fails()

        assert call_count == 3  # Initial + 2 retries

    def test_non_retryable_exception_not_retried(self):
        """Test that non-retryable exceptions aren't retried."""
        call_count = 0

        @retry_with_backoff(max_retries=3)
        def raises_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not retryable")

        with pytest.raises(ValueError):
            raises_value_error()

        assert call_count == 1  # No retries

    def test_retry_config_custom_exceptions(self):
        """Test custom retryable exceptions."""
        call_count = 0

        config = RetryConfig(
            max_retries=2,
            base_delay=0.01,
            retryable_exceptions=(ValueError,),
        )

        @retry_with_backoff(config=config)
        def custom_retry():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Custom retryable")
            return "ok"

        result = custom_retry()
        assert result == "ok"
        assert call_count == 2
