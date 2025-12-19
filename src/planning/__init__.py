"""
BRF Maintenance Plan Builder.

Tools for creating long-term energy investment plans for Swedish BRFs.
Includes cash flow simulation, ECM sequencing, and effektvakt optimization.

Key components:
- MaintenancePlan: Data model for plans
- CashFlowSimulator: Year-by-year projection
- ECMSequencer: Optimal investment ordering
- EffektvaktOptimizer: Peak demand reduction

Usage:
    from src.planning import (
        MaintenancePlan,
        CashFlowSimulator,
        ECMSequencer,
        create_cascade_sequence,
        analyze_effektvakt_potential,
    )
"""

from .models import (
    # Enums
    RenovationType,
    LoanTolerance,
    FundingSource,
    # Data classes
    PlannedRenovation,
    TariffStructure,
    BRFFinancials,
    ECMInvestment,
    YearlyProjection,
    MaintenancePlan,
    PlanScenario,
    # Helpers
    create_sample_financials,
    create_typical_renovation_plan,
)

from .cash_flow import (
    CashFlowSimulator,
    SimulationConfig,
    simulate_baseline_scenario,
    compare_scenarios,
)

from .sequencer import (
    ECMSequencer,
    SequencingStrategy,
    ECMCandidate,
    create_cascade_sequence,
)

from .effektvakt import (
    EffektvaktOptimizer,
    BuildingThermalProperties,
    TariffPeakStructure,
    PeakShavingResult,
    estimate_thermal_properties,
    analyze_effektvakt_potential,
)

__all__ = [
    # Models
    "RenovationType",
    "LoanTolerance",
    "FundingSource",
    "PlannedRenovation",
    "TariffStructure",
    "BRFFinancials",
    "ECMInvestment",
    "YearlyProjection",
    "MaintenancePlan",
    "PlanScenario",
    "create_sample_financials",
    "create_typical_renovation_plan",
    # Cash flow
    "CashFlowSimulator",
    "SimulationConfig",
    "simulate_baseline_scenario",
    "compare_scenarios",
    # Sequencer
    "ECMSequencer",
    "SequencingStrategy",
    "ECMCandidate",
    "create_cascade_sequence",
    # Effektvakt
    "EffektvaktOptimizer",
    "BuildingThermalProperties",
    "TariffPeakStructure",
    "PeakShavingResult",
    "estimate_thermal_properties",
    "analyze_effektvakt_potential",
]
