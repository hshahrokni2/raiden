"""
Swedish Occupancy & Schedule Patterns for EnergyPlus.

This module provides realistic occupancy, lighting, and equipment schedules
for Swedish buildings, replacing generic Sveby defaults with building-specific
patterns that can be learned and optimized.

Key Features:
- Building-type-specific patterns (residential families, elderly, students, etc.)
- Swedish seasonal patterns (summer vacation, Christmas, etc.)
- Occupant density by building type (BBR-compliant)
- Schedule generation for EnergyPlus IDF

Usage:
    from src.schedules import (
        SwedishOccupancyPattern,
        get_pattern_for_building,
        generate_schedule_idf,
        RESIDENTIAL_PATTERNS,
        COMMERCIAL_PATTERNS,
    )

    # Get pattern for building type
    pattern = get_pattern_for_building(
        building_type="residential",
        occupant_profile="families",
        num_apartments=50,
    )

    # Generate EnergyPlus schedule objects
    idf_snippet = generate_schedule_idf(pattern)
"""

from .swedish_patterns import (
    # Enums
    OccupantProfile,
    SeasonalPattern,
    DayType,
    # Data classes
    HourlyProfile,
    DailySchedule,
    WeeklySchedule,
    AnnualSchedule,
    SwedishOccupancyPattern,
    # Pattern libraries
    RESIDENTIAL_PATTERNS,
    COMMERCIAL_PATTERNS,
    SEASONAL_ADJUSTMENTS,
    SWEDISH_HOLIDAYS,
    # Functions
    get_pattern_for_building,
    generate_schedule_idf,
    create_custom_pattern,
    blend_patterns,
)

__all__ = [
    # Enums
    "OccupantProfile",
    "SeasonalPattern",
    "DayType",
    # Data classes
    "HourlyProfile",
    "DailySchedule",
    "WeeklySchedule",
    "AnnualSchedule",
    "SwedishOccupancyPattern",
    # Pattern libraries
    "RESIDENTIAL_PATTERNS",
    "COMMERCIAL_PATTERNS",
    "SEASONAL_ADJUSTMENTS",
    "SWEDISH_HOLIDAYS",
    # Functions
    "get_pattern_for_building",
    "generate_schedule_idf",
    "create_custom_pattern",
    "blend_patterns",
]
