"""
Swedish Occupancy Patterns for Building Energy Simulation.

Provides realistic occupancy, lighting, and equipment schedules based on:
1. Swedish living patterns (Boverket, Sveby)
2. Building-type-specific profiles
3. Seasonal variations (Swedish summer vacation, Christmas)

References:
- Sveby (Standardisera och verifiera energiprestanda för byggnader)
- BBR (Boverkets byggregler)
- SCB (Statistiska centralbyrån) time use surveys
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class OccupantProfile(Enum):
    """Occupant profiles for Swedish residential buildings."""
    # Residential
    FAMILIES = "families"                    # Working parents + children
    FAMILIES_SMALL_CHILDREN = "families_small_children"  # Parents home more
    ELDERLY = "elderly"                      # Pensioners, home most of day
    STUDENTS = "students"                    # Irregular schedules
    YOUNG_PROFESSIONALS = "young_professionals"  # Out during day
    MIXED_RESIDENTIAL = "mixed_residential"  # Typical MFH mix

    # Commercial
    OFFICE_STANDARD = "office_standard"      # 8-17, weekdays
    OFFICE_FLEX = "office_flex"              # Flexible hours, some WFH
    RETAIL = "retail"                        # Shop hours
    RESTAURANT = "restaurant"                # Evening peak
    SCHOOL = "school"                        # 8-16, school year
    GROCERY = "grocery"                      # Extended hours
    HEALTHCARE = "healthcare"                # 24/7 operation


class SeasonalPattern(Enum):
    """Swedish seasonal patterns affecting occupancy."""
    NORMAL = "normal"
    SUMMER_VACATION = "summer_vacation"      # July (industrisemester)
    CHRISTMAS = "christmas"                  # Dec 23 - Jan 6
    EASTER = "easter"                        # Week around Easter
    MIDSOMMAR = "midsommar"                  # Midsummer week


class DayType(Enum):
    """Day types for schedule differentiation."""
    WEEKDAY = "weekday"
    SATURDAY = "saturday"
    SUNDAY = "sunday"
    HOLIDAY = "holiday"


@dataclass
class HourlyProfile:
    """24-hour profile (0-1 fractions for each hour)."""
    values: List[float]  # 24 values, one per hour (0-23)

    def __post_init__(self):
        if len(self.values) != 24:
            raise ValueError(f"HourlyProfile requires 24 values, got {len(self.values)}")
        # Clamp values to 0-1
        self.values = [max(0.0, min(1.0, v)) for v in self.values]

    def to_idf_list(self) -> str:
        """Convert to EnergyPlus Schedule:Day:List format."""
        return ",\n    ".join([f"{v:.2f}" for v in self.values])

    def average(self) -> float:
        """Average value across all hours."""
        return sum(self.values) / 24

    def peak_hour(self) -> int:
        """Hour with highest value."""
        return self.values.index(max(self.values))


@dataclass
class DailySchedule:
    """Schedule for different day types."""
    weekday: HourlyProfile
    saturday: HourlyProfile
    sunday: HourlyProfile
    holiday: Optional[HourlyProfile] = None

    def __post_init__(self):
        if self.holiday is None:
            self.holiday = self.sunday  # Default holiday to Sunday pattern

    def get_profile(self, day_type: DayType) -> HourlyProfile:
        """Get profile for specific day type."""
        profiles = {
            DayType.WEEKDAY: self.weekday,
            DayType.SATURDAY: self.saturday,
            DayType.SUNDAY: self.sunday,
            DayType.HOLIDAY: self.holiday,
        }
        return profiles[day_type]


@dataclass
class WeeklySchedule:
    """Full week schedule with optional per-day override."""
    default: DailySchedule
    per_day: Optional[Dict[int, HourlyProfile]] = None  # 0=Monday, 6=Sunday

    def get_profile_for_day(self, day_of_week: int, is_holiday: bool = False) -> HourlyProfile:
        """Get profile for specific day of week."""
        if is_holiday:
            return self.default.holiday

        if self.per_day and day_of_week in self.per_day:
            return self.per_day[day_of_week]

        # Map day of week to day type
        if day_of_week < 5:
            return self.default.weekday
        elif day_of_week == 5:
            return self.default.saturday
        else:
            return self.default.sunday


@dataclass
class AnnualSchedule:
    """Annual schedule with seasonal variations."""
    base: WeeklySchedule
    seasonal_adjustments: Dict[SeasonalPattern, float] = field(default_factory=dict)

    def get_profile_for_date(
        self,
        month: int,
        day_of_week: int,
        is_holiday: bool = False
    ) -> Tuple[HourlyProfile, float]:
        """Get profile and seasonal factor for specific date."""
        base_profile = self.base.get_profile_for_day(day_of_week, is_holiday)

        # Determine seasonal pattern
        season = SeasonalPattern.NORMAL
        if month == 7:  # July
            season = SeasonalPattern.SUMMER_VACATION
        elif month == 12 and day_of_week >= 23:
            season = SeasonalPattern.CHRISTMAS

        factor = self.seasonal_adjustments.get(season, 1.0)
        return base_profile, factor


@dataclass
class SwedishOccupancyPattern:
    """Complete occupancy pattern for a Swedish building."""
    profile: OccupantProfile
    name: str
    description: str

    # Schedule components
    occupancy: AnnualSchedule
    lighting: AnnualSchedule
    equipment: AnnualSchedule
    dhw: AnnualSchedule  # Domestic hot water

    # Design parameters
    occupant_density_m2_person: float  # m² per person
    lighting_power_density_w_m2: float  # W/m²
    equipment_power_density_w_m2: float  # W/m²
    dhw_liters_per_person_day: float  # L/person/day

    # Thermal comfort
    heating_setpoint_occupied_c: float = 21.0
    heating_setpoint_unoccupied_c: float = 18.0
    cooling_setpoint_c: float = 26.0

    def get_annual_operating_hours(self) -> float:
        """Estimate annual operating hours (occupancy > 50%)."""
        # Simplified calculation
        weekday_hours = sum(1 for v in self.occupancy.base.default.weekday.values if v > 0.5)
        weekend_hours = sum(1 for v in self.occupancy.base.default.sunday.values if v > 0.5)
        return weekday_hours * 260 + weekend_hours * 105  # 260 weekdays, 105 weekend days


# =============================================================================
# SWEDISH RESIDENTIAL PATTERNS
# =============================================================================

# Typical Swedish family pattern (two working parents, school-age children)
FAMILY_OCCUPANCY_WEEKDAY = HourlyProfile([
    0.95, 0.95, 0.95, 0.95, 0.95, 0.95,  # 00-05: sleeping
    0.90, 0.70, 0.30, 0.20, 0.20, 0.20,  # 06-11: morning rush, leave for work/school
    0.20, 0.20, 0.30, 0.50, 0.70, 0.85,  # 12-17: mostly away, return afternoon
    0.90, 0.95, 0.95, 0.95, 0.95, 0.95,  # 18-23: dinner, evening at home
])

FAMILY_OCCUPANCY_WEEKEND = HourlyProfile([
    0.95, 0.95, 0.95, 0.95, 0.95, 0.95,  # 00-05: sleeping
    0.95, 0.95, 0.90, 0.80, 0.70, 0.60,  # 06-11: late morning, some activities
    0.50, 0.50, 0.60, 0.70, 0.80, 0.85,  # 12-17: mix of home/away
    0.90, 0.95, 0.95, 0.95, 0.95, 0.95,  # 18-23: evening at home
])

# Elderly pattern (mostly at home)
ELDERLY_OCCUPANCY_WEEKDAY = HourlyProfile([
    0.98, 0.98, 0.98, 0.98, 0.98, 0.98,  # 00-05: sleeping
    0.95, 0.90, 0.85, 0.80, 0.70, 0.70,  # 06-11: morning, some shopping/errands
    0.75, 0.80, 0.85, 0.80, 0.75, 0.80,  # 12-17: afternoon, brief outings
    0.90, 0.95, 0.95, 0.98, 0.98, 0.98,  # 18-23: evening at home
])

ELDERLY_OCCUPANCY_WEEKEND = HourlyProfile([
    0.98, 0.98, 0.98, 0.98, 0.98, 0.98,  # 00-05: sleeping
    0.95, 0.95, 0.90, 0.85, 0.80, 0.80,  # 06-11: slow morning
    0.80, 0.85, 0.85, 0.85, 0.85, 0.90,  # 12-17: mostly at home
    0.95, 0.95, 0.98, 0.98, 0.98, 0.98,  # 18-23: evening at home
])

# Student pattern (irregular, late nights)
STUDENT_OCCUPANCY_WEEKDAY = HourlyProfile([
    0.90, 0.90, 0.85, 0.80, 0.80, 0.80,  # 00-05: late nights
    0.85, 0.80, 0.60, 0.40, 0.30, 0.30,  # 06-11: morning classes
    0.35, 0.40, 0.45, 0.50, 0.55, 0.60,  # 12-17: campus, library
    0.70, 0.80, 0.85, 0.90, 0.90, 0.90,  # 18-23: evening at home
])

STUDENT_OCCUPANCY_WEEKEND = HourlyProfile([
    0.85, 0.85, 0.85, 0.85, 0.85, 0.90,  # 00-05: late nights, sleeping in
    0.95, 0.95, 0.95, 0.90, 0.85, 0.75,  # 06-11: sleeping, brunch
    0.60, 0.55, 0.50, 0.55, 0.60, 0.65,  # 12-17: social activities
    0.70, 0.75, 0.80, 0.85, 0.85, 0.85,  # 18-23: evening activities
])

# Young professionals (out during day, social evenings)
YOUNG_PRO_OCCUPANCY_WEEKDAY = HourlyProfile([
    0.95, 0.95, 0.95, 0.95, 0.95, 0.90,  # 00-05: sleeping
    0.80, 0.50, 0.20, 0.10, 0.10, 0.10,  # 06-11: early out to work
    0.10, 0.10, 0.10, 0.15, 0.20, 0.40,  # 12-17: at work
    0.60, 0.70, 0.75, 0.80, 0.85, 0.90,  # 18-23: gym, social, home
])

YOUNG_PRO_OCCUPANCY_WEEKEND = HourlyProfile([
    0.90, 0.90, 0.90, 0.90, 0.90, 0.90,  # 00-05: sleeping (late nights)
    0.95, 0.95, 0.95, 0.90, 0.80, 0.60,  # 06-11: sleeping in, brunch out
    0.50, 0.45, 0.50, 0.55, 0.60, 0.65,  # 12-17: activities, shopping
    0.70, 0.65, 0.60, 0.70, 0.80, 0.85,  # 18-23: social evenings
])

# Families with small children (one parent often home)
SMALL_CHILDREN_OCCUPANCY_WEEKDAY = HourlyProfile([
    0.98, 0.98, 0.98, 0.98, 0.98, 0.95,  # 00-05: sleeping
    0.90, 0.75, 0.55, 0.50, 0.55, 0.60,  # 06-11: morning, daycare dropoff
    0.55, 0.50, 0.55, 0.65, 0.75, 0.85,  # 12-17: one parent often home
    0.95, 0.98, 0.98, 0.98, 0.98, 0.98,  # 18-23: bedtime routines
])

SMALL_CHILDREN_OCCUPANCY_WEEKEND = HourlyProfile([
    0.98, 0.98, 0.98, 0.98, 0.98, 0.95,  # 00-05: sleeping
    0.95, 0.95, 0.95, 0.90, 0.85, 0.80,  # 06-11: early mornings with kids
    0.75, 0.70, 0.75, 0.80, 0.85, 0.90,  # 12-17: family activities
    0.95, 0.98, 0.98, 0.98, 0.98, 0.98,  # 18-23: evening at home
])


# =============================================================================
# LIGHTING PATTERNS (correlates with occupancy but not identical)
# =============================================================================

RESIDENTIAL_LIGHTING_WEEKDAY = HourlyProfile([
    0.05, 0.05, 0.05, 0.05, 0.05, 0.10,  # 00-05: minimal (night lights)
    0.40, 0.50, 0.30, 0.15, 0.15, 0.15,  # 06-11: morning routines
    0.15, 0.15, 0.20, 0.30, 0.50, 0.70,  # 12-17: increasing as daylight fades
    0.85, 0.90, 0.85, 0.70, 0.50, 0.20,  # 18-23: evening peak
])

RESIDENTIAL_LIGHTING_WEEKEND = HourlyProfile([
    0.05, 0.05, 0.05, 0.05, 0.05, 0.05,  # 00-05: minimal
    0.10, 0.20, 0.30, 0.40, 0.35, 0.30,  # 06-11: late mornings
    0.25, 0.25, 0.30, 0.40, 0.55, 0.70,  # 12-17: afternoon
    0.85, 0.90, 0.85, 0.75, 0.55, 0.25,  # 18-23: evening peak
])


# =============================================================================
# EQUIPMENT PATTERNS (appliances, electronics)
# =============================================================================

RESIDENTIAL_EQUIPMENT_WEEKDAY = HourlyProfile([
    0.10, 0.10, 0.10, 0.10, 0.10, 0.15,  # 00-05: standby loads
    0.40, 0.60, 0.35, 0.20, 0.20, 0.25,  # 06-11: morning appliances
    0.30, 0.25, 0.25, 0.35, 0.45, 0.60,  # 12-17: afternoon cooking starts
    0.80, 0.85, 0.75, 0.55, 0.40, 0.20,  # 18-23: dinner peak, evening TV
])

RESIDENTIAL_EQUIPMENT_WEEKEND = HourlyProfile([
    0.10, 0.10, 0.10, 0.10, 0.10, 0.10,  # 00-05: standby
    0.15, 0.25, 0.40, 0.55, 0.50, 0.55,  # 06-11: brunch cooking
    0.60, 0.55, 0.50, 0.50, 0.55, 0.65,  # 12-17: afternoon activities
    0.80, 0.85, 0.80, 0.65, 0.45, 0.25,  # 18-23: dinner, entertainment
])


# =============================================================================
# DOMESTIC HOT WATER PATTERNS
# =============================================================================

DHW_WEEKDAY = HourlyProfile([
    0.05, 0.05, 0.05, 0.05, 0.05, 0.15,  # 00-05: minimal
    0.60, 0.90, 0.50, 0.25, 0.20, 0.20,  # 06-11: morning showers peak
    0.25, 0.20, 0.20, 0.25, 0.30, 0.45,  # 12-17: afternoon
    0.70, 0.65, 0.55, 0.45, 0.35, 0.15,  # 18-23: evening baths, dishes
])

DHW_WEEKEND = HourlyProfile([
    0.05, 0.05, 0.05, 0.05, 0.05, 0.05,  # 00-05: minimal
    0.10, 0.25, 0.50, 0.75, 0.70, 0.50,  # 06-11: late morning showers
    0.40, 0.35, 0.35, 0.40, 0.45, 0.55,  # 12-17: afternoon
    0.70, 0.65, 0.60, 0.50, 0.35, 0.15,  # 18-23: evening
])


# =============================================================================
# COMMERCIAL PATTERNS
# =============================================================================

# Standard office (8-17, weekdays only)
OFFICE_OCCUPANCY_WEEKDAY = HourlyProfile([
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00,  # 00-05: closed
    0.05, 0.15, 0.60, 0.90, 0.95, 0.85,  # 06-11: morning ramp-up, lunch dip
    0.90, 0.95, 0.95, 0.90, 0.70, 0.30,  # 12-17: afternoon, end of day
    0.10, 0.05, 0.00, 0.00, 0.00, 0.00,  # 18-23: closed
])

OFFICE_OCCUPANCY_WEEKEND = HourlyProfile([
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00,  # 00-05: closed
    0.00, 0.00, 0.00, 0.05, 0.05, 0.05,  # 06-11: minimal
    0.05, 0.05, 0.05, 0.05, 0.00, 0.00,  # 12-17: occasional
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00,  # 18-23: closed
])

# Retail (shop hours)
RETAIL_OCCUPANCY_WEEKDAY = HourlyProfile([
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00,  # 00-05: closed
    0.00, 0.00, 0.05, 0.30, 0.50, 0.70,  # 06-11: opening, building
    0.85, 0.80, 0.70, 0.75, 0.85, 0.90,  # 12-17: lunch rush, after work
    0.70, 0.40, 0.10, 0.00, 0.00, 0.00,  # 18-23: closing
])

RETAIL_OCCUPANCY_WEEKEND = HourlyProfile([
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00,  # 00-05: closed
    0.00, 0.00, 0.05, 0.20, 0.50, 0.75,  # 06-11: late opening Saturday
    0.90, 0.95, 0.90, 0.80, 0.60, 0.30,  # 12-17: peak shopping
    0.10, 0.00, 0.00, 0.00, 0.00, 0.00,  # 18-23: closed
])

# Restaurant (evening peak)
RESTAURANT_OCCUPANCY_WEEKDAY = HourlyProfile([
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00,  # 00-05: closed
    0.00, 0.00, 0.00, 0.05, 0.20, 0.60,  # 06-11: prep, lunch start
    0.90, 0.50, 0.20, 0.20, 0.30, 0.70,  # 12-17: lunch peak, afternoon lull
    0.95, 1.00, 0.95, 0.70, 0.40, 0.10,  # 18-23: dinner peak
])

RESTAURANT_OCCUPANCY_WEEKEND = HourlyProfile([
    0.05, 0.00, 0.00, 0.00, 0.00, 0.00,  # 00-05: late Friday/Saturday
    0.00, 0.00, 0.00, 0.10, 0.30, 0.50,  # 06-11: brunch
    0.70, 0.60, 0.40, 0.35, 0.45, 0.70,  # 12-17: lunch, afternoon
    0.95, 1.00, 1.00, 0.85, 0.50, 0.15,  # 18-23: dinner peak
])

# Grocery store (extended hours)
GROCERY_OCCUPANCY_WEEKDAY = HourlyProfile([
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00,  # 00-05: closed (most stores)
    0.05, 0.20, 0.40, 0.50, 0.60, 0.75,  # 06-11: morning shoppers
    0.80, 0.65, 0.55, 0.60, 0.80, 0.95,  # 12-17: lunch, after work rush
    0.85, 0.70, 0.50, 0.30, 0.15, 0.05,  # 18-23: evening winding down
])

GROCERY_OCCUPANCY_WEEKEND = HourlyProfile([
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00,  # 00-05: closed
    0.00, 0.05, 0.20, 0.50, 0.75, 0.90,  # 06-11: weekend rush
    0.95, 0.90, 0.80, 0.70, 0.50, 0.30,  # 12-17: tapering
    0.15, 0.05, 0.00, 0.00, 0.00, 0.00,  # 18-23: closing
])

# School (school year hours)
SCHOOL_OCCUPANCY_WEEKDAY = HourlyProfile([
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00,  # 00-05: closed
    0.05, 0.30, 0.85, 0.95, 0.95, 0.90,  # 06-11: arrival, morning classes
    0.80, 0.95, 0.95, 0.90, 0.50, 0.15,  # 12-17: lunch, afternoon, departure
    0.05, 0.02, 0.00, 0.00, 0.00, 0.00,  # 18-23: after-school activities
])

SCHOOL_OCCUPANCY_WEEKEND = HourlyProfile([
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00,  # 00-05: closed
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00,  # 06-11: closed
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00,  # 12-17: closed
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00,  # 18-23: closed
])


# =============================================================================
# SEASONAL ADJUSTMENTS FOR SWEDEN
# =============================================================================

SEASONAL_ADJUSTMENTS = {
    # Residential adjustments
    OccupantProfile.FAMILIES: {
        SeasonalPattern.SUMMER_VACATION: 0.60,    # July: many families travel
        SeasonalPattern.CHRISTMAS: 1.10,          # More time at home
        SeasonalPattern.EASTER: 0.85,             # Many travel
        SeasonalPattern.MIDSOMMAR: 0.50,          # Everyone at summer houses
    },
    OccupantProfile.ELDERLY: {
        SeasonalPattern.SUMMER_VACATION: 0.90,    # Many stay home
        SeasonalPattern.CHRISTMAS: 1.05,          # Family visits
        SeasonalPattern.EASTER: 0.95,
        SeasonalPattern.MIDSOMMAR: 0.80,
    },
    OccupantProfile.STUDENTS: {
        SeasonalPattern.SUMMER_VACATION: 0.30,    # Most go home or travel
        SeasonalPattern.CHRISTMAS: 0.20,          # Home for holidays
        SeasonalPattern.EASTER: 0.70,
        SeasonalPattern.MIDSOMMAR: 0.40,
    },
    OccupantProfile.YOUNG_PROFESSIONALS: {
        SeasonalPattern.SUMMER_VACATION: 0.50,
        SeasonalPattern.CHRISTMAS: 0.60,
        SeasonalPattern.EASTER: 0.75,
        SeasonalPattern.MIDSOMMAR: 0.40,
    },

    # Commercial adjustments
    OccupantProfile.OFFICE_STANDARD: {
        SeasonalPattern.SUMMER_VACATION: 0.40,    # July: skeleton crew
        SeasonalPattern.CHRISTMAS: 0.20,          # Closed or minimal
        SeasonalPattern.EASTER: 0.60,
        SeasonalPattern.MIDSOMMAR: 0.30,
    },
    OccupantProfile.RETAIL: {
        SeasonalPattern.SUMMER_VACATION: 0.70,
        SeasonalPattern.CHRISTMAS: 1.50,          # Shopping season!
        SeasonalPattern.EASTER: 0.90,
        SeasonalPattern.MIDSOMMAR: 0.60,
    },
}

# Swedish public holidays (dates vary by year, approximate)
SWEDISH_HOLIDAYS = [
    (1, 1),    # Nyårsdagen
    (1, 6),    # Trettondedag jul
    # (Easter varies)
    (5, 1),    # Första maj
    # (Ascension varies)
    (6, 6),    # Nationaldagen
    # (Midsommar varies, late June)
    (11, 1),   # Alla helgons dag (approx)
    (12, 24),  # Julafton
    (12, 25),  # Juldagen
    (12, 26),  # Annandag jul
    (12, 31),  # Nyårsafton
]


# =============================================================================
# PRE-BUILT PATTERNS LIBRARY
# =============================================================================

def _create_residential_pattern(
    profile: OccupantProfile,
    name: str,
    description: str,
    occupancy_weekday: HourlyProfile,
    occupancy_weekend: HourlyProfile,
    occupant_density: float,
    heating_setpoint: float = 21.0,
) -> SwedishOccupancyPattern:
    """Helper to create residential patterns with standard lighting/equipment."""
    occupancy_schedule = AnnualSchedule(
        base=WeeklySchedule(
            default=DailySchedule(
                weekday=occupancy_weekday,
                saturday=occupancy_weekend,
                sunday=occupancy_weekend,
            )
        ),
        seasonal_adjustments=SEASONAL_ADJUSTMENTS.get(profile, {}),
    )

    lighting_schedule = AnnualSchedule(
        base=WeeklySchedule(
            default=DailySchedule(
                weekday=RESIDENTIAL_LIGHTING_WEEKDAY,
                saturday=RESIDENTIAL_LIGHTING_WEEKEND,
                sunday=RESIDENTIAL_LIGHTING_WEEKEND,
            )
        ),
    )

    equipment_schedule = AnnualSchedule(
        base=WeeklySchedule(
            default=DailySchedule(
                weekday=RESIDENTIAL_EQUIPMENT_WEEKDAY,
                saturday=RESIDENTIAL_EQUIPMENT_WEEKEND,
                sunday=RESIDENTIAL_EQUIPMENT_WEEKEND,
            )
        ),
    )

    dhw_schedule = AnnualSchedule(
        base=WeeklySchedule(
            default=DailySchedule(
                weekday=DHW_WEEKDAY,
                saturday=DHW_WEEKEND,
                sunday=DHW_WEEKEND,
            )
        ),
    )

    return SwedishOccupancyPattern(
        profile=profile,
        name=name,
        description=description,
        occupancy=occupancy_schedule,
        lighting=lighting_schedule,
        equipment=equipment_schedule,
        dhw=dhw_schedule,
        occupant_density_m2_person=occupant_density,
        lighting_power_density_w_m2=8.0,  # LED residential (BBR)
        equipment_power_density_w_m2=4.0,  # Typical residential (Sveby)
        dhw_liters_per_person_day=40.0,   # Sveby default
        heating_setpoint_occupied_c=heating_setpoint,
        heating_setpoint_unoccupied_c=heating_setpoint - 2.0,
    )


RESIDENTIAL_PATTERNS: Dict[OccupantProfile, SwedishOccupancyPattern] = {
    OccupantProfile.FAMILIES: _create_residential_pattern(
        profile=OccupantProfile.FAMILIES,
        name="Swedish Families",
        description="Two working parents with school-age children. Out during weekdays 8-17.",
        occupancy_weekday=FAMILY_OCCUPANCY_WEEKDAY,
        occupancy_weekend=FAMILY_OCCUPANCY_WEEKEND,
        occupant_density=35.0,  # 35 m²/person (Sveby MFH)
    ),
    OccupantProfile.FAMILIES_SMALL_CHILDREN: _create_residential_pattern(
        profile=OccupantProfile.FAMILIES_SMALL_CHILDREN,
        name="Families with Small Children",
        description="Parents with pre-school children. Often one parent home during day.",
        occupancy_weekday=SMALL_CHILDREN_OCCUPANCY_WEEKDAY,
        occupancy_weekend=SMALL_CHILDREN_OCCUPANCY_WEEKEND,
        occupant_density=30.0,  # More people per unit
        heating_setpoint=22.0,  # Warmer for small children
    ),
    OccupantProfile.ELDERLY: _create_residential_pattern(
        profile=OccupantProfile.ELDERLY,
        name="Elderly Residents",
        description="Pensioners, mostly at home during day. Lower activity levels.",
        occupancy_weekday=ELDERLY_OCCUPANCY_WEEKDAY,
        occupancy_weekend=ELDERLY_OCCUPANCY_WEEKEND,
        occupant_density=45.0,  # Often single or couple
        heating_setpoint=22.0,  # Warmer preferred
    ),
    OccupantProfile.STUDENTS: _create_residential_pattern(
        profile=OccupantProfile.STUDENTS,
        name="Student Housing",
        description="Irregular schedules, late nights, often away. High July vacancy.",
        occupancy_weekday=STUDENT_OCCUPANCY_WEEKDAY,
        occupancy_weekend=STUDENT_OCCUPANCY_WEEKEND,
        occupant_density=25.0,  # Dense student housing
        heating_setpoint=20.0,  # Often lower setpoint
    ),
    OccupantProfile.YOUNG_PROFESSIONALS: _create_residential_pattern(
        profile=OccupantProfile.YOUNG_PROFESSIONALS,
        name="Young Professionals",
        description="Single or couples without children. Out during day, social evenings.",
        occupancy_weekday=YOUNG_PRO_OCCUPANCY_WEEKDAY,
        occupancy_weekend=YOUNG_PRO_OCCUPANCY_WEEKEND,
        occupant_density=40.0,  # Often single occupancy
        heating_setpoint=21.0,
    ),
}

# Note: MIXED_RESIDENTIAL pattern is created after blend_patterns() is defined below


def _create_commercial_pattern(
    profile: OccupantProfile,
    name: str,
    description: str,
    occupancy_weekday: HourlyProfile,
    occupancy_weekend: HourlyProfile,
    occupant_density: float,
    lighting_density: float,
    equipment_density: float,
) -> SwedishOccupancyPattern:
    """Helper to create commercial patterns."""
    # Use occupancy as proxy for all schedules (commercial correlation is high)
    occupancy_schedule = AnnualSchedule(
        base=WeeklySchedule(
            default=DailySchedule(
                weekday=occupancy_weekday,
                saturday=occupancy_weekend,
                sunday=occupancy_weekend,
            )
        ),
        seasonal_adjustments=SEASONAL_ADJUSTMENTS.get(profile, {}),
    )

    # Lighting tracks occupancy closely in commercial
    lighting_schedule = occupancy_schedule

    # Equipment also tracks occupancy
    equipment_schedule = occupancy_schedule

    # DHW lower in commercial
    dhw_schedule = AnnualSchedule(
        base=WeeklySchedule(
            default=DailySchedule(
                weekday=occupancy_weekday,
                saturday=occupancy_weekend,
                sunday=occupancy_weekend,
            )
        ),
    )

    return SwedishOccupancyPattern(
        profile=profile,
        name=name,
        description=description,
        occupancy=occupancy_schedule,
        lighting=lighting_schedule,
        equipment=equipment_schedule,
        dhw=dhw_schedule,
        occupant_density_m2_person=occupant_density,
        lighting_power_density_w_m2=lighting_density,
        equipment_power_density_w_m2=equipment_density,
        dhw_liters_per_person_day=10.0,  # Lower in commercial
        heating_setpoint_occupied_c=21.0,
        heating_setpoint_unoccupied_c=15.0,  # Night setback in commercial
    )


COMMERCIAL_PATTERNS: Dict[OccupantProfile, SwedishOccupancyPattern] = {
    OccupantProfile.OFFICE_STANDARD: _create_commercial_pattern(
        profile=OccupantProfile.OFFICE_STANDARD,
        name="Standard Office",
        description="Typical office 8-17 weekdays. Closed weekends.",
        occupancy_weekday=OFFICE_OCCUPANCY_WEEKDAY,
        occupancy_weekend=OFFICE_OCCUPANCY_WEEKEND,
        occupant_density=15.0,  # 15 m²/person (BBR office)
        lighting_density=10.0,  # W/m² (LED office)
        equipment_density=15.0,  # W/m² (computers, etc.)
    ),
    OccupantProfile.RETAIL: _create_commercial_pattern(
        profile=OccupantProfile.RETAIL,
        name="Retail Store",
        description="Shop hours, Saturday peak. Closed Sunday (many Swedish stores).",
        occupancy_weekday=RETAIL_OCCUPANCY_WEEKDAY,
        occupancy_weekend=RETAIL_OCCUPANCY_WEEKEND,
        occupant_density=5.0,   # 5 m²/person (dense retail)
        lighting_density=15.0,  # Higher for display
        equipment_density=5.0,
    ),
    OccupantProfile.RESTAURANT: _create_commercial_pattern(
        profile=OccupantProfile.RESTAURANT,
        name="Restaurant",
        description="Lunch and dinner peaks. Kitchen equipment heavy.",
        occupancy_weekday=RESTAURANT_OCCUPANCY_WEEKDAY,
        occupancy_weekend=RESTAURANT_OCCUPANCY_WEEKEND,
        occupant_density=3.0,   # Very dense
        lighting_density=12.0,
        equipment_density=40.0,  # Kitchen equipment
    ),
    OccupantProfile.GROCERY: _create_commercial_pattern(
        profile=OccupantProfile.GROCERY,
        name="Grocery Store",
        description="Extended hours, refrigeration always on.",
        occupancy_weekday=GROCERY_OCCUPANCY_WEEKDAY,
        occupancy_weekend=GROCERY_OCCUPANCY_WEEKEND,
        occupant_density=8.0,
        lighting_density=12.0,
        equipment_density=25.0,  # Refrigeration, freezers
    ),
    OccupantProfile.SCHOOL: _create_commercial_pattern(
        profile=OccupantProfile.SCHOOL,
        name="School",
        description="School hours only. Closed summers, weekends, holidays.",
        occupancy_weekday=SCHOOL_OCCUPANCY_WEEKDAY,
        occupancy_weekend=SCHOOL_OCCUPANCY_WEEKEND,
        occupant_density=4.0,   # Dense classrooms
        lighting_density=10.0,
        equipment_density=8.0,
    ),
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def blend_patterns(
    patterns_weights: List[Tuple[SwedishOccupancyPattern, float]]
) -> SwedishOccupancyPattern:
    """
    Blend multiple patterns with weights.

    Used for mixed-use buildings or diverse occupant populations.

    Args:
        patterns_weights: List of (pattern, weight) tuples

    Returns:
        Blended pattern
    """
    if not patterns_weights:
        raise ValueError("At least one pattern required")

    total_weight = sum(w for _, w in patterns_weights)
    if abs(total_weight - 1.0) > 0.01:
        logger.warning(f"Pattern weights sum to {total_weight}, normalizing")
        patterns_weights = [(p, w / total_weight) for p, w in patterns_weights]

    # Blend hourly profiles
    def blend_hourly(attr: str, schedule_type: str) -> HourlyProfile:
        blended = [0.0] * 24
        for pattern, weight in patterns_weights:
            schedule = getattr(pattern, attr)
            daily = schedule.base.default
            profile = getattr(daily, schedule_type)
            for i, v in enumerate(profile.values):
                blended[i] += v * weight
        return HourlyProfile(blended)

    def blend_daily(attr: str) -> DailySchedule:
        return DailySchedule(
            weekday=blend_hourly(attr, "weekday"),
            saturday=blend_hourly(attr, "saturday"),
            sunday=blend_hourly(attr, "sunday"),
        )

    blended_occupancy = AnnualSchedule(
        base=WeeklySchedule(default=blend_daily("occupancy"))
    )
    blended_lighting = AnnualSchedule(
        base=WeeklySchedule(default=blend_daily("lighting"))
    )
    blended_equipment = AnnualSchedule(
        base=WeeklySchedule(default=blend_daily("equipment"))
    )
    blended_dhw = AnnualSchedule(
        base=WeeklySchedule(default=blend_daily("dhw"))
    )

    # Weighted average of densities
    avg_density = sum(p.occupant_density_m2_person * w for p, w in patterns_weights)
    avg_lighting = sum(p.lighting_power_density_w_m2 * w for p, w in patterns_weights)
    avg_equipment = sum(p.equipment_power_density_w_m2 * w for p, w in patterns_weights)
    avg_dhw = sum(p.dhw_liters_per_person_day * w for p, w in patterns_weights)

    return SwedishOccupancyPattern(
        profile=OccupantProfile.MIXED_RESIDENTIAL,
        name="Blended Pattern",
        description="Custom blend of multiple patterns",
        occupancy=blended_occupancy,
        lighting=blended_lighting,
        equipment=blended_equipment,
        dhw=blended_dhw,
        occupant_density_m2_person=avg_density,
        lighting_power_density_w_m2=avg_lighting,
        equipment_power_density_w_m2=avg_equipment,
        dhw_liters_per_person_day=avg_dhw,
    )


# Create MIXED_RESIDENTIAL pattern now that blend_patterns is defined
RESIDENTIAL_PATTERNS[OccupantProfile.MIXED_RESIDENTIAL] = SwedishOccupancyPattern(
    profile=OccupantProfile.MIXED_RESIDENTIAL,
    name="Mixed Residential (Typical MFH)",
    description="Typical Swedish MFH with mix of families, elderly, singles. Uses weighted average.",
    occupancy=blend_patterns(
        [
            (RESIDENTIAL_PATTERNS[OccupantProfile.FAMILIES], 0.40),
            (RESIDENTIAL_PATTERNS[OccupantProfile.ELDERLY], 0.25),
            (RESIDENTIAL_PATTERNS[OccupantProfile.YOUNG_PROFESSIONALS], 0.20),
            (RESIDENTIAL_PATTERNS[OccupantProfile.STUDENTS], 0.15),
        ]
    ).occupancy,
    lighting=RESIDENTIAL_PATTERNS[OccupantProfile.FAMILIES].lighting,
    equipment=RESIDENTIAL_PATTERNS[OccupantProfile.FAMILIES].equipment,
    dhw=RESIDENTIAL_PATTERNS[OccupantProfile.FAMILIES].dhw,
    occupant_density_m2_person=35.0,
    lighting_power_density_w_m2=8.0,
    equipment_power_density_w_m2=4.0,
    dhw_liters_per_person_day=40.0,
    heating_setpoint_occupied_c=21.0,
    heating_setpoint_unoccupied_c=19.0,
)


def get_pattern_for_building(
    building_type: str = "residential",
    occupant_profile: Optional[str] = None,
    num_apartments: Optional[int] = None,
    commercial_type: Optional[str] = None,
) -> SwedishOccupancyPattern:
    """
    Get appropriate pattern for a building.

    Args:
        building_type: "residential" or "commercial"
        occupant_profile: Specific profile name (e.g., "families", "elderly")
        num_apartments: Number of apartments (for inferring mix)
        commercial_type: Type of commercial use

    Returns:
        SwedishOccupancyPattern for the building
    """
    if building_type.lower() == "commercial":
        if commercial_type:
            # Map string to enum
            type_map = {
                "office": OccupantProfile.OFFICE_STANDARD,
                "retail": OccupantProfile.RETAIL,
                "restaurant": OccupantProfile.RESTAURANT,
                "grocery": OccupantProfile.GROCERY,
                "school": OccupantProfile.SCHOOL,
            }
            profile = type_map.get(commercial_type.lower(), OccupantProfile.OFFICE_STANDARD)
            return COMMERCIAL_PATTERNS.get(profile, COMMERCIAL_PATTERNS[OccupantProfile.OFFICE_STANDARD])
        return COMMERCIAL_PATTERNS[OccupantProfile.OFFICE_STANDARD]

    # Residential
    if occupant_profile:
        # Map string to enum
        profile_map = {
            "families": OccupantProfile.FAMILIES,
            "families_small_children": OccupantProfile.FAMILIES_SMALL_CHILDREN,
            "elderly": OccupantProfile.ELDERLY,
            "students": OccupantProfile.STUDENTS,
            "young_professionals": OccupantProfile.YOUNG_PROFESSIONALS,
            "mixed": OccupantProfile.MIXED_RESIDENTIAL,
        }
        profile = profile_map.get(occupant_profile.lower(), OccupantProfile.MIXED_RESIDENTIAL)
        return RESIDENTIAL_PATTERNS.get(profile, RESIDENTIAL_PATTERNS[OccupantProfile.MIXED_RESIDENTIAL])

    # Default to mixed residential
    return RESIDENTIAL_PATTERNS[OccupantProfile.MIXED_RESIDENTIAL]


def create_custom_pattern(
    base_pattern: SwedishOccupancyPattern,
    occupancy_multiplier: float = 1.0,
    lighting_multiplier: float = 1.0,
    equipment_multiplier: float = 1.0,
    setpoint_adjustment_c: float = 0.0,
) -> SwedishOccupancyPattern:
    """
    Create custom pattern from base with adjustments.

    Useful for calibration or building-specific tuning.

    Args:
        base_pattern: Starting pattern
        occupancy_multiplier: Scale occupancy (e.g., 0.8 for 80%)
        lighting_multiplier: Scale lighting
        equipment_multiplier: Scale equipment
        setpoint_adjustment_c: Adjust heating setpoint

    Returns:
        Adjusted pattern
    """
    def scale_profile(profile: HourlyProfile, multiplier: float) -> HourlyProfile:
        return HourlyProfile([min(1.0, v * multiplier) for v in profile.values])

    def scale_daily(daily: DailySchedule, multiplier: float) -> DailySchedule:
        return DailySchedule(
            weekday=scale_profile(daily.weekday, multiplier),
            saturday=scale_profile(daily.saturday, multiplier),
            sunday=scale_profile(daily.sunday, multiplier),
        )

    scaled_occupancy = AnnualSchedule(
        base=WeeklySchedule(
            default=scale_daily(base_pattern.occupancy.base.default, occupancy_multiplier)
        ),
        seasonal_adjustments=base_pattern.occupancy.seasonal_adjustments,
    )

    scaled_lighting = AnnualSchedule(
        base=WeeklySchedule(
            default=scale_daily(base_pattern.lighting.base.default, lighting_multiplier)
        ),
    )

    scaled_equipment = AnnualSchedule(
        base=WeeklySchedule(
            default=scale_daily(base_pattern.equipment.base.default, equipment_multiplier)
        ),
    )

    return SwedishOccupancyPattern(
        profile=base_pattern.profile,
        name=f"{base_pattern.name} (Custom)",
        description=f"Modified from {base_pattern.name}",
        occupancy=scaled_occupancy,
        lighting=scaled_lighting,
        equipment=scaled_equipment,
        dhw=base_pattern.dhw,
        occupant_density_m2_person=base_pattern.occupant_density_m2_person,
        lighting_power_density_w_m2=base_pattern.lighting_power_density_w_m2 * lighting_multiplier,
        equipment_power_density_w_m2=base_pattern.equipment_power_density_w_m2 * equipment_multiplier,
        dhw_liters_per_person_day=base_pattern.dhw_liters_per_person_day,
        heating_setpoint_occupied_c=base_pattern.heating_setpoint_occupied_c + setpoint_adjustment_c,
        heating_setpoint_unoccupied_c=base_pattern.heating_setpoint_unoccupied_c + setpoint_adjustment_c,
    )


# =============================================================================
# ENERGYPLUS IDF GENERATION
# =============================================================================

def generate_schedule_idf(
    pattern: SwedishOccupancyPattern,
    zone_name: str = "Zone1",
    prefix: str = "",
) -> str:
    """
    Generate EnergyPlus Schedule objects for a pattern.

    Creates:
    - Schedule:Compact for occupancy
    - Schedule:Compact for lighting
    - Schedule:Compact for equipment
    - Schedule:Compact for DHW
    - Schedule:Compact for heating setpoint
    - Schedule:Compact for cooling setpoint

    Args:
        pattern: SwedishOccupancyPattern to convert
        zone_name: Zone name for naming schedules
        prefix: Optional prefix for schedule names

    Returns:
        EnergyPlus IDF snippet as string
    """
    name_prefix = f"{prefix}{zone_name}_" if prefix else f"{zone_name}_"

    def compact_schedule(name: str, hourly_weekday: HourlyProfile, hourly_weekend: HourlyProfile) -> str:
        """Generate Schedule:Compact from hourly profiles."""
        # EnergyPlus Schedule:Compact format
        lines = [
            f"Schedule:Compact,",
            f"    {name_prefix}{name},  !- Name",
            f"    Fraction,              !- Schedule Type Limits Name",
            f"    Through: 12/31,        !- Field 1",
            f"    For: Weekdays,         !- Field 2",
        ]

        # Add weekday hourly values
        for hour in range(24):
            val = hourly_weekday.values[hour]
            lines.append(f"    Until: {hour+1:02d}:00, {val:.2f},  !- Hour {hour}")

        lines.append(f"    For: Weekends Holidays CustomDay1 CustomDay2,  !- Field")

        # Add weekend hourly values
        for hour in range(24):
            val = hourly_weekend.values[hour]
            end = ";" if hour == 23 else ","
            lines.append(f"    Until: {hour+1:02d}:00, {val:.2f}{end}  !- Hour {hour}")

        return "\n".join(lines)

    def temperature_schedule(name: str, occupied_temp: float, unoccupied_temp: float) -> str:
        """Generate temperature setpoint schedule."""
        lines = [
            f"Schedule:Compact,",
            f"    {name_prefix}{name},  !- Name",
            f"    Temperature,           !- Schedule Type Limits Name",
            f"    Through: 12/31,        !- Field 1",
            f"    For: Weekdays,         !- Field 2",
            f"    Until: 06:00, {unoccupied_temp:.1f},  !- Night setback",
            f"    Until: 22:00, {occupied_temp:.1f},   !- Occupied",
            f"    Until: 24:00, {unoccupied_temp:.1f},  !- Night setback",
            f"    For: Weekends Holidays CustomDay1 CustomDay2,",
            f"    Until: 08:00, {unoccupied_temp:.1f},  !- Weekend morning",
            f"    Until: 23:00, {occupied_temp:.1f},   !- Weekend day",
            f"    Until: 24:00, {unoccupied_temp:.1f};  !- Weekend night",
        ]
        return "\n".join(lines)

    # Get profiles
    occ_wd = pattern.occupancy.base.default.weekday
    occ_we = pattern.occupancy.base.default.sunday

    light_wd = pattern.lighting.base.default.weekday
    light_we = pattern.lighting.base.default.sunday

    equip_wd = pattern.equipment.base.default.weekday
    equip_we = pattern.equipment.base.default.sunday

    dhw_wd = pattern.dhw.base.default.weekday
    dhw_we = pattern.dhw.base.default.sunday

    # Build IDF snippet
    snippets = [
        f"! ==========================================",
        f"! Schedules for {zone_name} - {pattern.name}",
        f"! {pattern.description}",
        f"! ==========================================",
        "",
        "! Schedule Type Limits",
        "ScheduleTypeLimits,",
        "    Fraction,              !- Name",
        "    0,                     !- Lower Limit",
        "    1,                     !- Upper Limit",
        "    Continuous;            !- Numeric Type",
        "",
        "ScheduleTypeLimits,",
        "    Temperature,           !- Name",
        "    0,                     !- Lower Limit",
        "    50,                    !- Upper Limit",
        "    Continuous;            !- Numeric Type",
        "",
        compact_schedule("Occupancy", occ_wd, occ_we),
        "",
        compact_schedule("Lighting", light_wd, light_we),
        "",
        compact_schedule("Equipment", equip_wd, equip_we),
        "",
        compact_schedule("DHW", dhw_wd, dhw_we),
        "",
        temperature_schedule(
            "HeatingSetpoint",
            pattern.heating_setpoint_occupied_c,
            pattern.heating_setpoint_unoccupied_c
        ),
        "",
        temperature_schedule(
            "CoolingSetpoint",
            pattern.cooling_setpoint_c,
            pattern.cooling_setpoint_c + 2.0  # Allow higher when unoccupied
        ),
    ]

    return "\n".join(snippets)
