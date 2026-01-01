"""
Raiden Orchestrator - Portfolio-scale building analysis with agentic QC.

Provides tiered processing for analyzing 1000+ buildings efficiently:
- Tier 1: Fast triage via Sweden Buildings GeoJSON (10 buildings/sec)
- Tier 2: Standard analysis with pre-trained surrogates (50 concurrent)
- Tier 3: Deep analysis with EnergyPlus simulation (10 concurrent)

Agentic QC triggers when confidence is low:
- ImageQCAgent: Re-analyze facades when WWR/material confidence < 60%
- ECMRefinerAgent: Adjust packages based on building-specific context
- AnomalyAgent: Investigate unusual patterns (renovation detection)

Usage:
    from src.orchestrator import RaidenOrchestrator

    orchestrator = RaidenOrchestrator()
    results = await orchestrator.analyze_portfolio(
        addresses=["Bellmansgatan 16", "Aktergatan 11", ...],
        parallel_workers=50,
    )

    print(f"Analyzed: {results.analytics.analyzed} buildings")
    print(f"Total savings: {results.analytics.total_savings_potential_kwh:,.0f} kWh")
"""

from .orchestrator import (
    RaidenOrchestrator,
    BuildingResult,
    PortfolioResult,
    AnalysisTier,
    TierConfig,
    analyze_portfolio,
    analyze_portfolio_hybrid,
)
from .prioritizer import (
    BuildingPrioritizer,
    PrioritizationStrategy,
    TriageResult,
)
from .qc_agent import (
    QCAgent,
    ImageQCAgent,
    ECMRefinerAgent,
    AnomalyAgent,
    QCResult,
    QCTrigger,
    QCTriggerType,
)
from .surrogate_library import (
    SurrogateLibrary,
    ArchetypeSurrogate,
    get_or_train_surrogate,
)
from .portfolio_report import (
    PortfolioAnalytics,
    generate_portfolio_report,
)

__all__ = [
    # Core orchestration
    "RaidenOrchestrator",
    "BuildingResult",
    "PortfolioResult",
    "AnalysisTier",
    "TierConfig",
    "analyze_portfolio",
    "analyze_portfolio_hybrid",
    # Prioritization
    "BuildingPrioritizer",
    "PrioritizationStrategy",
    "TriageResult",
    # QC Agents
    "QCAgent",
    "ImageQCAgent",
    "ECMRefinerAgent",
    "AnomalyAgent",
    "QCResult",
    "QCTrigger",
    "QCTriggerType",
    # Surrogate library
    "SurrogateLibrary",
    "ArchetypeSurrogate",
    "get_or_train_surrogate",
    # Reporting
    "PortfolioAnalytics",
    "generate_portfolio_report",
]
