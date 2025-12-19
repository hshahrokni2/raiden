"""Building analysis modules."""

from .u_value_calculator import (
    UValueEstimate,
    BuildingEnvelope,
    calculate_envelope_areas,
    calculate_heat_loss,
    back_calculate_u_values,
    estimate_from_specific_energy,
    get_era_estimates,
)
from .roof_analyzer import (
    RoofType,
    RoofAnalyzer,
    RoofAnalysis,
    RoofSegment,
    RoofObstruction,
    ObstructionType,
    ExistingSolarInstallation,
    analyze_roof,
)
from .building_analyzer import (
    BuildingAnalyzer,
    BuildingAnalysisResult,
    ECMScenarioResult,
    AnalysisPackage,
    analyze_building,
)
from .package_generator import (
    PackageGenerator,
    ECMPackage,
    ECMPackageItem,
    generate_packages,
    ECM_COSTS_PER_M2,
    ENERGY_PRICE_SEK_KWH,
)
from .package_simulator import (
    PackageSimulator,
    SimulatedPackage,
    PackageECM,
    create_packages_from_ecm_results,
)

__all__ = [
    # U-value calculations
    "UValueEstimate",
    "BuildingEnvelope",
    "calculate_envelope_areas",
    "calculate_heat_loss",
    "back_calculate_u_values",
    "estimate_from_specific_energy",
    "get_era_estimates",
    # Roof analysis
    "RoofType",
    "RoofAnalyzer",
    "RoofAnalysis",
    "RoofSegment",
    "RoofObstruction",
    "ObstructionType",
    "ExistingSolarInstallation",
    "analyze_roof",
    # Building analyzer
    "BuildingAnalyzer",
    "BuildingAnalysisResult",
    "ECMScenarioResult",
    "AnalysisPackage",
    "analyze_building",
    # Package generator (estimated)
    "PackageGenerator",
    "ECMPackage",
    "ECMPackageItem",
    "generate_packages",
    "ECM_COSTS_PER_M2",
    "ENERGY_PRICE_SEK_KWH",
    # Package simulator (physics-based)
    "PackageSimulator",
    "SimulatedPackage",
    "PackageECM",
    "create_packages_from_ecm_results",
]
