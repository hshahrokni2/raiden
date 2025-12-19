"""
Thermal Mass Calculator

Calculates building thermal mass from geometry and materials:
- Effective thermal mass for heating/cooling dynamics
- Time constants for temperature response
- Material-specific heat capacity

Important for:
- Peak load calculations
- Night setback recovery
- Solar gain utilization
"""

from dataclasses import dataclass
from typing import Dict
from enum import Enum


class ConstructionMaterial(Enum):
    """Common Swedish building materials."""
    CONCRETE = "concrete"
    BRICK = "brick"
    LIGHTWEIGHT_CONCRETE = "lightweight_concrete"  # Lättbetong
    WOOD_FRAME = "wood_frame"
    STEEL_FRAME = "steel_frame"
    CLT = "clt"  # Cross-laminated timber


@dataclass
class MaterialProperties:
    """Thermal properties of a material."""
    density_kg_m3: float
    specific_heat_j_kg_k: float
    conductivity_w_m_k: float

    @property
    def volumetric_heat_capacity(self) -> float:
        """Heat capacity per volume (J/m³K)."""
        return self.density_kg_m3 * self.specific_heat_j_kg_k


# Swedish building material database
MATERIAL_DATABASE: Dict[ConstructionMaterial, MaterialProperties] = {
    ConstructionMaterial.CONCRETE: MaterialProperties(
        density_kg_m3=2400,
        specific_heat_j_kg_k=880,
        conductivity_w_m_k=1.7
    ),
    ConstructionMaterial.BRICK: MaterialProperties(
        density_kg_m3=1800,
        specific_heat_j_kg_k=800,
        conductivity_w_m_k=0.8
    ),
    ConstructionMaterial.LIGHTWEIGHT_CONCRETE: MaterialProperties(
        density_kg_m3=600,
        specific_heat_j_kg_k=1000,
        conductivity_w_m_k=0.2
    ),
    ConstructionMaterial.WOOD_FRAME: MaterialProperties(
        density_kg_m3=500,
        specific_heat_j_kg_k=1600,
        conductivity_w_m_k=0.13
    ),
    ConstructionMaterial.CLT: MaterialProperties(
        density_kg_m3=480,
        specific_heat_j_kg_k=1600,
        conductivity_w_m_k=0.12
    ),
}


@dataclass
class ThermalMass:
    """Building thermal mass assessment."""
    # Total mass
    total_mass_kg: float
    effective_mass_kg: float  # Mass that participates in diurnal cycling

    # Heat capacity
    total_heat_capacity_mj_k: float  # MJ/K
    effective_heat_capacity_mj_k: float
    heat_capacity_per_floor_area_kj_m2_k: float

    # Time constant
    time_constant_hours: float  # τ = C/UA

    # Classification
    mass_class: str  # 'light', 'medium', 'heavy'


class ThermalMassCalculator:
    """
    Calculate building thermal mass.

    Usage:
        calculator = ThermalMassCalculator()
        mass = calculator.calculate(
            floor_area_m2=2240,
            floors=7,
            structure_material=ConstructionMaterial.CONCRETE,
            floor_thickness_m=0.2,
            internal_walls_area_m2=500
        )
    """

    # Effective mass depth (how much mass participates in daily cycle)
    EFFECTIVE_DEPTH_M = 0.10  # First 10cm of exposed mass

    # Mass classification thresholds (kJ/m²K floor area)
    LIGHT_THRESHOLD = 80
    MEDIUM_THRESHOLD = 200

    def __init__(self):
        pass

    def calculate(
        self,
        floor_area_m2: float,
        floors: int,
        structure_material: ConstructionMaterial,
        floor_thickness_m: float = 0.20,
        internal_walls_area_m2: float = 0.0,
        exposed_ceiling: bool = True,
        u_value_average: float = 0.30
    ) -> ThermalMass:
        """
        Calculate building thermal mass.

        Args:
            floor_area_m2: Total floor area (Atemp)
            floors: Number of floors
            structure_material: Primary structural material
            floor_thickness_m: Concrete floor slab thickness
            internal_walls_area_m2: Additional internal thermal mass
            exposed_ceiling: Whether ceiling concrete is exposed
            u_value_average: Average envelope U-value (for time constant)

        Returns:
            ThermalMass assessment
        """
        # Get material properties
        props = MATERIAL_DATABASE.get(
            structure_material,
            MATERIAL_DATABASE[ConstructionMaterial.CONCRETE]
        )

        floor_area_per_floor = floor_area_m2 / floors if floors > 0 else floor_area_m2

        # Calculate floor slab mass
        # Each floor has top (floor) and bottom (ceiling) exposed surfaces
        # Effective depth is limited to EFFECTIVE_DEPTH_M
        effective_floor_depth = min(floor_thickness_m, self.EFFECTIVE_DEPTH_M)
        floor_slab_volume_m3 = floor_area_per_floor * floor_thickness_m * floors

        # Effective volume (mass participating in diurnal cycling)
        # Floor surface + ceiling surface (if exposed)
        surfaces_per_floor = 2 if exposed_ceiling else 1
        effective_volume_m3 = floor_area_per_floor * effective_floor_depth * surfaces_per_floor * floors

        # Internal walls thermal mass
        internal_wall_thickness = 0.15  # Assume 150mm internal walls
        internal_wall_volume_m3 = internal_walls_area_m2 * internal_wall_thickness
        effective_internal_volume = internal_walls_area_m2 * min(internal_wall_thickness, self.EFFECTIVE_DEPTH_M * 2)

        # Total volumes
        total_volume_m3 = floor_slab_volume_m3 + internal_wall_volume_m3
        total_effective_volume_m3 = effective_volume_m3 + effective_internal_volume

        # Calculate masses
        total_mass_kg = total_volume_m3 * props.density_kg_m3
        effective_mass_kg = total_effective_volume_m3 * props.density_kg_m3

        # Calculate heat capacities
        volumetric_cp = props.volumetric_heat_capacity  # J/m³K
        total_heat_capacity_j_k = total_volume_m3 * volumetric_cp
        effective_heat_capacity_j_k = total_effective_volume_m3 * volumetric_cp

        # Convert to MJ/K
        total_heat_capacity_mj_k = total_heat_capacity_j_k / 1e6
        effective_heat_capacity_mj_k = effective_heat_capacity_j_k / 1e6

        # Heat capacity per floor area (kJ/m²K)
        heat_capacity_per_area = (effective_heat_capacity_j_k / 1000) / floor_area_m2

        # Time constant τ = C / UA
        # Approximate envelope area
        envelope_area_m2 = floor_area_m2 * 0.5  # Rough approximation
        ua_value = envelope_area_m2 * u_value_average  # W/K
        time_constant_seconds = effective_heat_capacity_j_k / ua_value if ua_value > 0 else 0
        time_constant_hours = time_constant_seconds / 3600

        # Classify mass level
        mass_class = self._classify_mass(heat_capacity_per_area)

        return ThermalMass(
            total_mass_kg=total_mass_kg,
            effective_mass_kg=effective_mass_kg,
            total_heat_capacity_mj_k=total_heat_capacity_mj_k,
            effective_heat_capacity_mj_k=effective_heat_capacity_mj_k,
            heat_capacity_per_floor_area_kj_m2_k=heat_capacity_per_area,
            time_constant_hours=time_constant_hours,
            mass_class=mass_class,
        )

    def _classify_mass(self, heat_capacity_per_area: float) -> str:
        """Classify thermal mass as light/medium/heavy."""
        if heat_capacity_per_area < self.LIGHT_THRESHOLD:
            return 'light'
        elif heat_capacity_per_area < self.MEDIUM_THRESHOLD:
            return 'medium'
        else:
            return 'heavy'
