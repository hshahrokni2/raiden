"""
Zone Assignment for Multi-Use Buildings.

Assigns building uses to specific floors based on Swedish building patterns:
- Ground floor: Commercial (retail, restaurant, grocery)
- Upper floors: Residential (with FTX where available)

This physical layout is critical because:
1. Commercial zones have separate ventilation (no heat recovery)
2. Thermal coupling between floors affects energy use
3. EnergyPlus needs proper zone geometry for accurate simulation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class FloorZone:
    """A zone assigned to a specific floor."""
    floor: int  # 1-indexed
    zone_type: str  # residential, restaurant, retail, etc.
    area_m2: float
    height_m: float = 2.8

    # Zone-specific properties
    ventilation_type: str = "FTX"  # F, FT, FTX
    airflow_l_s_m2: float = 0.35
    heat_recovery_eff: float = 0.80
    internal_gains_w_m2: float = 5.0

    # Geometry
    is_ground_floor: bool = False
    is_top_floor: bool = False
    has_exterior_walls: bool = True

    @property
    def zone_name(self) -> str:
        """Generate unique zone name."""
        return f"Floor{self.floor}_{self.zone_type.capitalize()}"


@dataclass
class BuildingZoneLayout:
    """Complete zone layout for a building."""
    total_floors: int
    footprint_area_m2: float
    floor_height_m: float = 2.8

    # Zone assignments
    floor_zones: List[FloorZone] = field(default_factory=list)

    # Summary
    residential_floors: int = 0
    commercial_floors: int = 0

    @property
    def total_atemp_m2(self) -> float:
        return self.footprint_area_m2 * self.total_floors

    def get_zones_by_type(self, zone_type: str) -> List[FloorZone]:
        """Get all zones of a specific type."""
        return [z for z in self.floor_zones if z.zone_type == zone_type]

    def get_zone_by_floor(self, floor: int) -> Optional[FloorZone]:
        """Get zone for a specific floor."""
        for z in self.floor_zones:
            if z.floor == floor:
                return z
        return None


def assign_zones_to_floors(
    total_floors: int,
    footprint_area_m2: float,
    zone_breakdown: Dict[str, float],
    floor_height_m: float = 2.8,
    has_ftx: bool = True,
    has_f_only: bool = False,
) -> BuildingZoneLayout:
    """
    Assign zones to floors based on use-type breakdown.

    Swedish pattern:
    - Commercial (retail, restaurant, grocery) → Ground floor(s)
    - Residential → Upper floors
    - Office can be either ground or mid-floors

    Args:
        total_floors: Number of floors above ground
        footprint_area_m2: Area per floor
        zone_breakdown: Dict of zone_type -> fraction (0.0-1.0)
        floor_height_m: Height per floor
        has_ftx: Whether residential has FTX
        has_f_only: Whether F-only ventilation exists (commercial/older)

    Returns:
        BuildingZoneLayout with floor assignments
    """
    from ..ingest.zone_configs import ZONE_CONFIGS

    layout = BuildingZoneLayout(
        total_floors=total_floors,
        footprint_area_m2=footprint_area_m2,
        floor_height_m=floor_height_m,
    )

    # Separate commercial and residential zones
    commercial_types = ['restaurant', 'retail', 'grocery', 'theater', 'pool']
    office_types = ['office']
    residential_types = ['residential', 'hotel']

    commercial_fraction = sum(
        zone_breakdown.get(t, 0) for t in commercial_types
    )
    office_fraction = sum(
        zone_breakdown.get(t, 0) for t in office_types
    )
    residential_fraction = sum(
        zone_breakdown.get(t, 0) for t in residential_types
    )

    # Calculate how many floors each use type gets
    # Commercial goes on ground floor(s), residential on upper
    commercial_floors = max(1, round(commercial_fraction * total_floors)) if commercial_fraction > 0.02 else 0
    office_floors = max(1, round(office_fraction * total_floors)) if office_fraction > 0.02 else 0
    residential_floors = total_floors - commercial_floors - office_floors

    # Ensure at least 1 residential floor if there's any residential
    if residential_fraction > 0 and residential_floors < 1:
        residential_floors = 1
        if commercial_floors > 1:
            commercial_floors -= 1
        elif office_floors > 1:
            office_floors -= 1

    layout.commercial_floors = commercial_floors
    layout.residential_floors = residential_floors

    # Assign zones to floors
    current_floor = 1

    # Ground floor(s): Commercial
    if commercial_floors > 0:
        # Distribute commercial types across floors
        commercial_zones = [(t, f) for t, f in zone_breakdown.items()
                           if t in commercial_types and f > 0.01]

        for floor_idx in range(commercial_floors):
            floor_num = current_floor + floor_idx

            # Assign dominant commercial type for this floor
            if commercial_zones:
                zone_type, _ = commercial_zones[floor_idx % len(commercial_zones)]
            else:
                zone_type = 'retail'  # Default

            config = ZONE_CONFIGS.get(zone_type, ZONE_CONFIGS['retail'])

            layout.floor_zones.append(FloorZone(
                floor=floor_num,
                zone_type=zone_type,
                area_m2=footprint_area_m2,
                height_m=floor_height_m,
                ventilation_type=config.get('ventilation_type', 'F'),
                airflow_l_s_m2=config.get('airflow_l_s_m2', 1.5),
                heat_recovery_eff=config.get('heat_recovery_eff', 0.0),
                internal_gains_w_m2=config.get('internal_gains_w_m2', 20.0),
                is_ground_floor=(floor_num == 1),
                is_top_floor=(floor_num == total_floors),
            ))

        current_floor += commercial_floors

    # Middle floor(s): Office (if present)
    if office_floors > 0:
        config = ZONE_CONFIGS.get('office', ZONE_CONFIGS['other'])

        for floor_idx in range(office_floors):
            floor_num = current_floor + floor_idx

            layout.floor_zones.append(FloorZone(
                floor=floor_num,
                zone_type='office',
                area_m2=footprint_area_m2,
                height_m=floor_height_m,
                ventilation_type='FTX',
                airflow_l_s_m2=config.get('airflow_l_s_m2', 1.0),
                heat_recovery_eff=config.get('heat_recovery_eff', 0.75),
                internal_gains_w_m2=config.get('internal_gains_w_m2', 25.0),
                is_ground_floor=(floor_num == 1),
                is_top_floor=(floor_num == total_floors),
            ))

        current_floor += office_floors

    # Upper floor(s): Residential
    if residential_floors > 0:
        config = ZONE_CONFIGS.get('residential', ZONE_CONFIGS['residential'])

        # Determine ventilation based on building info
        if has_ftx:
            vent_type = 'FTX'
            hr_eff = 0.80
        elif has_f_only:
            vent_type = 'F'
            hr_eff = 0.0
        else:
            vent_type = 'FTX'  # Default for modern buildings
            hr_eff = 0.75

        for floor_idx in range(residential_floors):
            floor_num = current_floor + floor_idx

            layout.floor_zones.append(FloorZone(
                floor=floor_num,
                zone_type='residential',
                area_m2=footprint_area_m2,
                height_m=floor_height_m,
                ventilation_type=vent_type,
                airflow_l_s_m2=config.get('airflow_l_s_m2', 0.35),
                heat_recovery_eff=hr_eff,
                internal_gains_w_m2=config.get('internal_gains_w_m2', 5.0),
                is_ground_floor=(floor_num == 1),
                is_top_floor=(floor_num == total_floors),
            ))

        current_floor += residential_floors

    # Handle pure residential (no commercial)
    if not layout.floor_zones:
        config = ZONE_CONFIGS.get('residential', ZONE_CONFIGS['residential'])
        hr_eff = 0.80 if has_ftx else 0.0

        for floor_num in range(1, total_floors + 1):
            layout.floor_zones.append(FloorZone(
                floor=floor_num,
                zone_type='residential',
                area_m2=footprint_area_m2,
                height_m=floor_height_m,
                ventilation_type='FTX' if has_ftx else 'F',
                airflow_l_s_m2=0.35,
                heat_recovery_eff=hr_eff,
                internal_gains_w_m2=5.0,
                is_ground_floor=(floor_num == 1),
                is_top_floor=(floor_num == total_floors),
            ))

        layout.residential_floors = total_floors

    return layout


def get_zone_layout_summary(layout: BuildingZoneLayout) -> str:
    """Generate human-readable summary of zone layout."""
    lines = [
        f"Building: {layout.total_floors} floors, {layout.footprint_area_m2:.0f} m²/floor",
        f"Total Atemp: {layout.total_atemp_m2:.0f} m²",
        f"Commercial floors: {layout.commercial_floors}",
        f"Residential floors: {layout.residential_floors}",
        "",
        "Floor Layout:",
    ]

    for zone in sorted(layout.floor_zones, key=lambda z: z.floor):
        vent_info = f"{zone.ventilation_type}"
        if zone.heat_recovery_eff > 0:
            vent_info += f" ({zone.heat_recovery_eff:.0%} HR)"

        floor_type = ""
        if zone.is_ground_floor:
            floor_type = " [GROUND]"
        elif zone.is_top_floor:
            floor_type = " [TOP]"

        lines.append(
            f"  Floor {zone.floor}: {zone.zone_type.upper()}{floor_type} "
            f"- {vent_info}, {zone.airflow_l_s_m2:.2f} L/s·m²"
        )

    return "\n".join(lines)
