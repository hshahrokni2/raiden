"""
Swedish Building Zone Configurations.

BBR (Boverkets Byggregler) compliant ventilation rates and thermal parameters
for different building use types. Critical for multi-zone energy modeling.

References:
- BBR 6:23 - Ventilation requirements by activity type
- BBR 6:24 - Kitchen exhaust requirements
- Sveby - Swedish energy calculation conventions
"""

from dataclasses import dataclass
from typing import Dict, Any

# Zone configurations based on Swedish BBR and Sveby
# All airflow rates in L/s per m² Atemp
ZONE_CONFIGS: Dict[str, Dict[str, Any]] = {
    # ==========================================================================
    # RESIDENTIAL
    # ==========================================================================
    'residential': {
        'name_sv': 'Bostad',
        'name_en': 'Residential',
        'ventilation_type': 'FTX',  # Typical for modern buildings
        'airflow_l_s_m2': 0.35,  # BBR minimum for dwellings
        'heat_recovery_eff': 0.80,  # Typical FTX efficiency
        'internal_gains_w_m2': 5.0,  # Sveby: 4-6 W/m² for residential
        'hot_water_kwh_m2_year': 25.0,  # Sveby: 20-30 kWh/m²
        'occupancy_hours_per_day': 14,  # Average presence
        'operating_days_per_year': 365,
        'setpoint_heating_c': 21.0,
        'setpoint_cooling_c': 26.0,
        'lighting_w_m2': 8.0,
        'equipment_w_m2': 4.0,
        'notes': 'Standard residential with FTX. Airflow can be 0.35-0.50 L/s·m².',
    },

    'residential_f_only': {
        'name_sv': 'Bostad (F-ventilation)',
        'name_en': 'Residential (exhaust only)',
        'ventilation_type': 'F',
        'airflow_l_s_m2': 0.35,
        'heat_recovery_eff': 0.0,  # No heat recovery!
        'internal_gains_w_m2': 5.0,
        'hot_water_kwh_m2_year': 25.0,
        'occupancy_hours_per_day': 14,
        'operating_days_per_year': 365,
        'setpoint_heating_c': 21.0,
        'setpoint_cooling_c': 26.0,
        'lighting_w_m2': 8.0,
        'equipment_w_m2': 4.0,
        'notes': 'Older residential without heat recovery. Common pre-1990.',
    },

    # ==========================================================================
    # COMMERCIAL - FOOD SERVICE
    # ==========================================================================
    'restaurant': {
        'name_sv': 'Restaurang',
        'name_en': 'Restaurant',
        'ventilation_type': 'F',  # Kitchen exhaust MUST be F-only (grease, odor)
        'airflow_l_s_m2': 10.0,  # BBR 6:24: 5-15 L/s·m² for commercial kitchens
        'heat_recovery_eff': 0.0,  # Cannot recover from greasy exhaust
        'internal_gains_w_m2': 50.0,  # High: cooking equipment
        'hot_water_kwh_m2_year': 50.0,  # High: dishwashing
        'occupancy_hours_per_day': 12,  # Typical restaurant hours
        'operating_days_per_year': 330,  # ~10% closed
        'setpoint_heating_c': 20.0,
        'setpoint_cooling_c': 24.0,
        'lighting_w_m2': 15.0,
        'equipment_w_m2': 100.0,  # Commercial kitchen equipment
        'notes': 'BBR requires separate exhaust for kitchens. No HR possible due to grease.',
    },

    'grocery': {
        'name_sv': 'Livsmedelsbutik',
        'name_en': 'Grocery Store',
        'ventilation_type': 'FTX',  # Can have FTX except food prep areas
        'airflow_l_s_m2': 3.0,  # Higher due to customer traffic + refrigeration
        'heat_recovery_eff': 0.60,  # Partial - some areas need F-only
        'internal_gains_w_m2': 40.0,  # High: refrigeration waste heat
        'hot_water_kwh_m2_year': 10.0,
        'occupancy_hours_per_day': 14,
        'operating_days_per_year': 360,
        'setpoint_heating_c': 18.0,  # Lower due to internal gains
        'setpoint_cooling_c': 22.0,
        'lighting_w_m2': 25.0,  # Bright display lighting
        'equipment_w_m2': 80.0,  # Refrigeration
        'notes': 'Refrigeration creates significant internal gains. Often net cooling load.',
    },

    # ==========================================================================
    # COMMERCIAL - RETAIL
    # ==========================================================================
    'retail': {
        'name_sv': 'Butik',
        'name_en': 'Retail Shop',
        'ventilation_type': 'FTX',  # Usually FTX
        'airflow_l_s_m2': 1.5,  # BBR: 0.35 + activity based
        'heat_recovery_eff': 0.70,
        'internal_gains_w_m2': 20.0,  # Display lighting, equipment
        'hot_water_kwh_m2_year': 5.0,
        'occupancy_hours_per_day': 10,
        'operating_days_per_year': 310,
        'setpoint_heating_c': 20.0,
        'setpoint_cooling_c': 24.0,
        'lighting_w_m2': 20.0,  # Display lighting
        'equipment_w_m2': 10.0,
        'notes': 'Standard retail. Can vary widely depending on type.',
    },

    # ==========================================================================
    # COMMERCIAL - OFFICE
    # ==========================================================================
    'office': {
        'name_sv': 'Kontor',
        'name_en': 'Office',
        'ventilation_type': 'FTX',
        'airflow_l_s_m2': 1.0,  # BBR: 0.35 + 7 L/s per person
        'heat_recovery_eff': 0.75,
        'internal_gains_w_m2': 25.0,  # Computers, lighting
        'hot_water_kwh_m2_year': 3.0,
        'occupancy_hours_per_day': 10,
        'operating_days_per_year': 250,
        'setpoint_heating_c': 21.0,
        'setpoint_cooling_c': 24.0,
        'lighting_w_m2': 12.0,
        'equipment_w_m2': 15.0,  # Computers
        'notes': 'Modern office with FTX. High internal gains from IT equipment.',
    },

    # ==========================================================================
    # HOSPITALITY
    # ==========================================================================
    'hotel': {
        'name_sv': 'Hotell',
        'name_en': 'Hotel',
        'ventilation_type': 'FTX',
        'airflow_l_s_m2': 0.50,  # Slightly higher than residential
        'heat_recovery_eff': 0.75,
        'internal_gains_w_m2': 8.0,
        'hot_water_kwh_m2_year': 40.0,  # High: frequent showers
        'occupancy_hours_per_day': 16,
        'operating_days_per_year': 365,
        'setpoint_heating_c': 21.0,
        'setpoint_cooling_c': 24.0,
        'lighting_w_m2': 10.0,
        'equipment_w_m2': 5.0,
        'notes': 'Similar to residential but higher hot water demand.',
    },

    # ==========================================================================
    # EDUCATION & HEALTHCARE
    # ==========================================================================
    'school': {
        'name_sv': 'Skola',
        'name_en': 'School',
        'ventilation_type': 'FTX',
        'airflow_l_s_m2': 1.2,  # High occupancy density
        'heat_recovery_eff': 0.75,
        'internal_gains_w_m2': 15.0,  # Students + equipment
        'hot_water_kwh_m2_year': 5.0,
        'occupancy_hours_per_day': 8,
        'operating_days_per_year': 200,  # School year
        'setpoint_heating_c': 20.0,
        'setpoint_cooling_c': 25.0,
        'lighting_w_m2': 12.0,
        'equipment_w_m2': 8.0,
        'notes': 'High occupancy density. Intermittent use (school holidays).',
    },

    'healthcare': {
        'name_sv': 'Vård',
        'name_en': 'Healthcare',
        'ventilation_type': 'FTX',
        'airflow_l_s_m2': 2.0,  # High for infection control
        'heat_recovery_eff': 0.70,  # May be lower for isolation rooms
        'internal_gains_w_m2': 20.0,
        'hot_water_kwh_m2_year': 30.0,
        'occupancy_hours_per_day': 24,
        'operating_days_per_year': 365,
        'setpoint_heating_c': 22.0,  # Higher for patient comfort
        'setpoint_cooling_c': 24.0,
        'lighting_w_m2': 15.0,
        'equipment_w_m2': 25.0,  # Medical equipment
        'notes': '24/7 operation. High ventilation for infection control.',
    },

    'daycare': {
        'name_sv': 'Förskola/Daghem',
        'name_en': 'Daycare',
        'ventilation_type': 'FTX',
        'airflow_l_s_m2': 1.5,  # High for child health
        'heat_recovery_eff': 0.75,
        'internal_gains_w_m2': 12.0,
        'hot_water_kwh_m2_year': 15.0,
        'occupancy_hours_per_day': 10,
        'operating_days_per_year': 250,
        'setpoint_heating_c': 21.0,
        'setpoint_cooling_c': 25.0,
        'lighting_w_m2': 10.0,
        'equipment_w_m2': 5.0,
        'notes': 'Similar to school but longer hours, higher ventilation.',
    },

    # ==========================================================================
    # LEISURE & CULTURE
    # ==========================================================================
    'theater': {
        'name_sv': 'Teater/Bio',
        'name_en': 'Theater/Cinema',
        'ventilation_type': 'FTX',
        'airflow_l_s_m2': 2.5,  # Very high occupancy during events
        'heat_recovery_eff': 0.70,
        'internal_gains_w_m2': 10.0,
        'hot_water_kwh_m2_year': 3.0,
        'occupancy_hours_per_day': 6,  # Intermittent
        'operating_days_per_year': 300,
        'setpoint_heating_c': 20.0,
        'setpoint_cooling_c': 23.0,
        'lighting_w_m2': 8.0,
        'equipment_w_m2': 5.0,
        'notes': 'Highly variable occupancy. Peak loads during performances.',
    },

    'pool': {
        'name_sv': 'Badhus',
        'name_en': 'Swimming Pool',
        'ventilation_type': 'F',  # High humidity, often F-only
        'airflow_l_s_m2': 8.0,  # Very high for humidity control
        'heat_recovery_eff': 0.0,  # Humid air, no standard HR
        'internal_gains_w_m2': 5.0,
        'hot_water_kwh_m2_year': 100.0,  # Very high: pool heating + showers
        'occupancy_hours_per_day': 12,
        'operating_days_per_year': 350,
        'setpoint_heating_c': 26.0,  # Pool hall temperature
        'setpoint_cooling_c': 30.0,
        'lighting_w_m2': 15.0,
        'equipment_w_m2': 10.0,
        'notes': 'Extremely high energy use. Pool heating dominates. May use heat pump dehumidification.',
    },

    # ==========================================================================
    # OTHER
    # ==========================================================================
    'other': {
        'name_sv': 'Övrig verksamhet',
        'name_en': 'Other',
        'ventilation_type': 'FTX',
        'airflow_l_s_m2': 1.0,  # Conservative estimate
        'heat_recovery_eff': 0.70,
        'internal_gains_w_m2': 15.0,
        'hot_water_kwh_m2_year': 10.0,
        'occupancy_hours_per_day': 10,
        'operating_days_per_year': 300,
        'setpoint_heating_c': 20.0,
        'setpoint_cooling_c': 24.0,
        'lighting_w_m2': 12.0,
        'equipment_w_m2': 10.0,
        'notes': 'Generic values for unspecified activities.',
    },
}


@dataclass
class ZoneConfig:
    """Structured zone configuration for type safety."""
    name_sv: str
    name_en: str
    ventilation_type: str  # 'F', 'FT', 'FTX'
    airflow_l_s_m2: float
    heat_recovery_eff: float
    internal_gains_w_m2: float
    hot_water_kwh_m2_year: float
    occupancy_hours_per_day: int
    operating_days_per_year: int
    setpoint_heating_c: float
    setpoint_cooling_c: float
    lighting_w_m2: float
    equipment_w_m2: float
    notes: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ZoneConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def get_zone_config(zone_type: str) -> ZoneConfig:
    """Get zone configuration by type."""
    if zone_type not in ZONE_CONFIGS:
        zone_type = 'other'
    return ZoneConfig.from_dict(ZONE_CONFIGS[zone_type])


def calculate_effective_ventilation(
    zones: Dict[str, float],
    has_ftx: bool = True,
) -> Dict[str, float]:
    """
    Calculate effective building-wide ventilation parameters.

    Args:
        zones: Dict of zone_type -> fraction (0.0-1.0)
        has_ftx: Whether the building has FTX (affects residential zones)

    Returns:
        Dict with effective_airflow, effective_hr, and breakdown
    """
    total_airflow_weighted = 0.0
    total_heat_loss_factor = 0.0

    zone_details = []

    for zone_type, fraction in zones.items():
        config = ZONE_CONFIGS.get(zone_type, ZONE_CONFIGS['other'])
        airflow = config['airflow_l_s_m2']

        # Adjust HR based on zone type and building ventilation
        if zone_type == 'residential' and not has_ftx:
            hr = 0.0  # F-only residential
        else:
            hr = config['heat_recovery_eff']

        # Accumulate
        total_airflow_weighted += airflow * fraction
        heat_loss = airflow * (1.0 - hr)
        total_heat_loss_factor += heat_loss * fraction

        zone_details.append({
            'zone': zone_type,
            'fraction': fraction,
            'airflow': airflow,
            'hr': hr,
            'heat_loss_contribution': heat_loss * fraction,
        })

    # Effective HR = 1 - (weighted_heat_loss / weighted_airflow)
    if total_airflow_weighted > 0:
        effective_hr = 1.0 - (total_heat_loss_factor / total_airflow_weighted)
    else:
        effective_hr = 0.0

    return {
        'effective_airflow_l_s_m2': total_airflow_weighted,
        'effective_heat_recovery': max(0.0, min(1.0, effective_hr)),
        'heat_loss_factor': total_heat_loss_factor,
        'zone_details': zone_details,
    }
