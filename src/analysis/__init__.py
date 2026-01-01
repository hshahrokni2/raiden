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
    get_ecm_cost_per_m2,
    ENERGY_PRICE_SEK_KWH,
)
from .package_simulator import (
    PackageSimulator,
    SimulatedPackage,
    PackageECM,
    create_packages_from_ecm_results,
)
from .energy_breakdown import (
    EndUse,
    EnergyBreakdown,
    ECM_END_USE_EFFECTS,
    DHW_DEFAULTS,
    PROPERTY_EL_DEFAULTS,
    estimate_baseline_breakdown,
    calculate_ecm_savings,
    format_breakdown_for_report,
)
from .visual_analyzer import (
    VisualAnalyzer,
    VisualAnalysisResult,
    GroundFloorResult,
    analyze_building_visually,
    analyze_address_visually,
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
    "get_ecm_cost_per_m2",
    "ENERGY_PRICE_SEK_KWH",
    # Package simulator (physics-based)
    "PackageSimulator",
    "SimulatedPackage",
    "PackageECM",
    "create_packages_from_ecm_results",
    # Energy breakdown (multi-end-use tracking)
    "EndUse",
    "EnergyBreakdown",
    "ECM_END_USE_EFFECTS",
    "DHW_DEFAULTS",
    "PROPERTY_EL_DEFAULTS",
    "estimate_baseline_breakdown",
    "calculate_ecm_savings",
    "format_breakdown_for_report",
    # Visual analyzer (standalone facade analysis)
    "VisualAnalyzer",
    "VisualAnalysisResult",
    "GroundFloorResult",
    "analyze_building_visually",
    "analyze_address_visually",
]
