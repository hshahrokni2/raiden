"""
Input validation utilities for Raiden.

Provides validation for addresses, coordinates, and building data.

Usage:
    from src.utils.validation import (
        validate_address,
        validate_coordinates,
        ValidationError,
    )

    address = validate_address("Bellmansgatan 16, Stockholm")
    lat, lon = validate_coordinates(59.3293, 18.0686)
"""

import re
from dataclasses import dataclass
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class ValidationError(ValueError):
    """Raised when input validation fails."""

    def __init__(self, message: str, field: str = "", suggestions: Optional[List[str]] = None):
        super().__init__(message)
        self.field = field
        self.suggestions = suggestions or []


@dataclass
class AddressComponents:
    """Parsed address components."""

    street_name: str
    street_number: Optional[str]
    city: Optional[str]
    postal_code: Optional[str]
    original: str


# Swedish address patterns
SWEDISH_ADDRESS_PATTERN = re.compile(
    r"^(?P<street>[A-Za-zÀ-ÿ\s\-]+)\s*"
    r"(?P<number>\d+\s*[A-Za-z]?)?\s*"
    r"(?:,\s*(?P<postal>\d{3}\s?\d{2})?\s*(?P<city>[A-Za-zÀ-ÿ\s]+)?)?$",
    re.UNICODE,
)

# Valid Swedish cities (major ones for validation)
MAJOR_SWEDISH_CITIES = {
    "stockholm",
    "göteborg",
    "malmö",
    "uppsala",
    "linköping",
    "västerås",
    "örebro",
    "helsingborg",
    "norrköping",
    "jönköping",
    "umeå",
    "lund",
    "borås",
    "gävle",
    "södertälje",
    "eskilstuna",
    "halmstad",
    "växjö",
    "karlstad",
    "sundsvall",
}

# Sweden coordinate bounds
SWEDEN_BOUNDS = {
    "min_lat": 55.0,
    "max_lat": 69.5,
    "min_lon": 10.5,
    "max_lon": 24.5,
}

# Stockholm bounds (for GeoJSON coverage check)
STOCKHOLM_BOUNDS = {
    "min_lat": 59.0,
    "max_lat": 59.7,
    "min_lon": 17.5,
    "max_lon": 18.5,
}


def validate_address(
    address: str,
    require_city: bool = False,
    require_number: bool = False,
) -> AddressComponents:
    """
    Validate and parse a Swedish address.

    Args:
        address: Address string to validate
        require_city: If True, require city in address
        require_number: If True, require street number

    Returns:
        AddressComponents with parsed data

    Raises:
        ValidationError: If address is invalid
    """
    if not address:
        raise ValidationError(
            "Address cannot be empty",
            field="address",
            suggestions=["Enter a Swedish street address like 'Storgatan 1, Stockholm'"],
        )

    # Clean up whitespace
    address = " ".join(address.split())

    # Check minimum length
    if len(address) < 5:
        raise ValidationError(
            f"Address too short: '{address}'",
            field="address",
            suggestions=["Include at least street name and number"],
        )

    # Check for obvious non-addresses
    if address.isdigit():
        raise ValidationError(
            "Address cannot be just numbers",
            field="address",
            suggestions=["Include street name with the number"],
        )

    # Try to parse with pattern
    match = SWEDISH_ADDRESS_PATTERN.match(address)

    if not match:
        # Still accept the address but log warning
        logger.warning(f"Address format not recognized: {address}")
        return AddressComponents(
            street_name=address,
            street_number=None,
            city=None,
            postal_code=None,
            original=address,
        )

    components = AddressComponents(
        street_name=match.group("street").strip() if match.group("street") else "",
        street_number=match.group("number").strip() if match.group("number") else None,
        city=match.group("city").strip() if match.group("city") else None,
        postal_code=match.group("postal").strip() if match.group("postal") else None,
        original=address,
    )

    # Validate requirements
    if require_number and not components.street_number:
        raise ValidationError(
            f"Street number required but missing: '{address}'",
            field="address",
            suggestions=["Add street number like 'Storgatan 1'"],
        )

    if require_city and not components.city:
        raise ValidationError(
            f"City required but missing: '{address}'",
            field="address",
            suggestions=["Add city like 'Storgatan 1, Stockholm'"],
        )

    return components


def validate_coordinates(
    latitude: float,
    longitude: float,
    require_sweden: bool = True,
    warn_outside_stockholm: bool = True,
) -> Tuple[float, float]:
    """
    Validate geographic coordinates.

    Args:
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees
        require_sweden: If True, coordinates must be within Sweden
        warn_outside_stockholm: If True, warn if outside Stockholm coverage

    Returns:
        Tuple of (latitude, longitude)

    Raises:
        ValidationError: If coordinates are invalid
    """
    # Basic range check
    if not (-90 <= latitude <= 90):
        raise ValidationError(
            f"Invalid latitude {latitude}: must be between -90 and 90",
            field="latitude",
        )

    if not (-180 <= longitude <= 180):
        raise ValidationError(
            f"Invalid longitude {longitude}: must be between -180 and 180",
            field="longitude",
        )

    # Sweden bounds check
    if require_sweden:
        if not (
            SWEDEN_BOUNDS["min_lat"] <= latitude <= SWEDEN_BOUNDS["max_lat"]
            and SWEDEN_BOUNDS["min_lon"] <= longitude <= SWEDEN_BOUNDS["max_lon"]
        ):
            raise ValidationError(
                f"Coordinates ({latitude}, {longitude}) are outside Sweden",
                field="coordinates",
                suggestions=[
                    "Swedish coordinates should be roughly lat 55-70, lon 10-25",
                    "For Stockholm area: lat 59.0-59.7, lon 17.5-18.5",
                ],
            )

    # Stockholm coverage warning
    if warn_outside_stockholm:
        in_stockholm = (
            STOCKHOLM_BOUNDS["min_lat"] <= latitude <= STOCKHOLM_BOUNDS["max_lat"]
            and STOCKHOLM_BOUNDS["min_lon"] <= longitude <= STOCKHOLM_BOUNDS["max_lon"]
        )
        if not in_stockholm:
            logger.warning(
                f"Coordinates ({latitude}, {longitude}) are outside Stockholm - "
                "building data from GeoJSON may not be available"
            )

    return (latitude, longitude)


def validate_construction_year(
    year: int,
    min_year: int = 1800,
    max_year: int = 2030,
) -> int:
    """
    Validate building construction year.

    Args:
        year: Construction year to validate
        min_year: Minimum valid year
        max_year: Maximum valid year

    Returns:
        Validated year

    Raises:
        ValidationError: If year is invalid
    """
    if not isinstance(year, int):
        try:
            year = int(year)
        except (ValueError, TypeError):
            raise ValidationError(
                f"Construction year must be a number: got '{year}'",
                field="construction_year",
            )

    if year < min_year:
        raise ValidationError(
            f"Construction year {year} seems too old (minimum: {min_year})",
            field="construction_year",
            suggestions=["Swedish building archetypes start from 1800s"],
        )

    if year > max_year:
        raise ValidationError(
            f"Construction year {year} is in the future (maximum: {max_year})",
            field="construction_year",
        )

    return year


def validate_building_area(
    atemp_m2: float,
    min_area: float = 50.0,
    max_area: float = 100000.0,
) -> float:
    """
    Validate building heated area (Atemp).

    Args:
        atemp_m2: Area in square meters
        min_area: Minimum valid area
        max_area: Maximum valid area

    Returns:
        Validated area

    Raises:
        ValidationError: If area is invalid
    """
    if not isinstance(atemp_m2, (int, float)):
        try:
            atemp_m2 = float(atemp_m2)
        except (ValueError, TypeError):
            raise ValidationError(
                f"Building area must be a number: got '{atemp_m2}'",
                field="atemp_m2",
            )

    if atemp_m2 < min_area:
        raise ValidationError(
            f"Building area {atemp_m2} m² seems too small",
            field="atemp_m2",
            suggestions=[f"Minimum area is {min_area} m²"],
        )

    if atemp_m2 > max_area:
        raise ValidationError(
            f"Building area {atemp_m2} m² seems too large",
            field="atemp_m2",
            suggestions=[f"Maximum area is {max_area} m²"],
        )

    return atemp_m2


def validate_energy_class(energy_class: str) -> str:
    """
    Validate Swedish energy class.

    Args:
        energy_class: Energy class (A-G or Unknown)

    Returns:
        Validated energy class

    Raises:
        ValidationError: If energy class is invalid
    """
    valid_classes = {"A", "B", "C", "D", "E", "F", "G", "Unknown"}

    if not energy_class:
        return "Unknown"

    normalized = energy_class.upper().strip()

    if normalized not in valid_classes:
        raise ValidationError(
            f"Invalid energy class '{energy_class}'",
            field="energy_class",
            suggestions=[f"Valid classes are: {', '.join(sorted(valid_classes))}"],
        )

    return normalized


def validate_facade_material(material: str) -> str:
    """
    Validate facade material type.

    Args:
        material: Facade material name

    Returns:
        Normalized material name

    Raises:
        ValidationError: If material is invalid
    """
    valid_materials = {
        "brick": ["brick", "tegel", "murad"],
        "concrete": ["concrete", "betong", "element"],
        "plaster": ["plaster", "puts", "rendered"],
        "wood": ["wood", "trä", "timber"],
        "metal": ["metal", "metall", "aluminum", "stål"],
        "glass": ["glass", "glas"],
        "stone": ["stone", "sten", "natural stone"],
    }

    if not material:
        return "concrete"  # Default

    material_lower = material.lower().strip()

    for canonical, variants in valid_materials.items():
        if material_lower in variants or material_lower == canonical:
            return canonical

    # Log warning but accept unknown materials
    logger.warning(f"Unknown facade material: {material}")
    return material_lower


def validate_heating_system(system: str) -> str:
    """
    Validate heating system type.

    Args:
        system: Heating system name

    Returns:
        Normalized system name

    Raises:
        ValidationError: If system is invalid
    """
    valid_systems = {
        "district": ["district", "fjärrvärme", "district_heating"],
        "heat_pump": ["heat_pump", "värmepump", "hp"],
        "heat_pump_ground": ["heat_pump_ground", "bergvärme", "ground_source"],
        "heat_pump_air": ["heat_pump_air", "luftvärmepump", "air_source"],
        "electric": ["electric", "el", "direktel", "electrical"],
        "gas": ["gas", "naturgas", "natural_gas"],
        "oil": ["oil", "olja"],
        "pellets": ["pellets", "biobränsle"],
    }

    if not system:
        return "district"  # Default for Swedish multi-family

    system_lower = system.lower().strip()

    for canonical, variants in valid_systems.items():
        if system_lower in variants or system_lower == canonical:
            return canonical

    # Log warning but accept unknown systems
    logger.warning(f"Unknown heating system: {system}")
    return system_lower
