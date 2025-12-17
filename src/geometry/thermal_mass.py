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
        exposed_ceiling: bool = True
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

        Returns:
            ThermalMass assessment
        """
        # TODO: Implement
        # 1. Get material properties
        # 2. Calculate floor slab mass (exposed surfaces)
        # 3. Calculate internal wall mass
        # 4. Calculate effective mass (first 10cm)
        # 5. Calculate heat capacity
        # 6. Estimate time constant
        # 7. Classify mass level
        raise NotImplementedError("Implement thermal mass calculation")

    def _classify_mass(self, heat_capacity_per_area: float) -> str:
        """Classify thermal mass as light/medium/heavy."""
        if heat_capacity_per_area < self.LIGHT_THRESHOLD:
            return 'light'
        elif heat_capacity_per_area < self.MEDIUM_THRESHOLD:
            return 'medium'
        else:
            return 'heavy'
