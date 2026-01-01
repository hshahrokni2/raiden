"""
Building prioritization strategies for portfolio analysis.

Determines which buildings should be analyzed first and at what tier.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .orchestrator import AnalysisTier

logger = logging.getLogger(__name__)


class PrioritizationStrategy(Enum):
    """Strategy for ordering buildings in portfolio analysis."""

    # Energy-based
    HIGHEST_CONSUMPTION_FIRST = "highest_consumption"
    LOWEST_ENERGY_CLASS_FIRST = "lowest_energy_class"

    # ROI-based
    HIGHEST_ROI_POTENTIAL = "highest_roi"
    QUICKEST_PAYBACK = "quickest_payback"

    # Building characteristics
    OLDEST_FIRST = "oldest_first"
    LARGEST_FIRST = "largest_first"

    # Data quality
    HIGHEST_CONFIDENCE_FIRST = "highest_confidence"
    LOWEST_CONFIDENCE_FIRST = "lowest_confidence"  # QC focus


@dataclass
class TriageResult:
    """Result of building triage."""

    address: str
    tier: AnalysisTier
    priority_score: float
    priority_reason: str

    # Building data (if available)
    construction_year: Optional[int] = None
    atemp_m2: Optional[float] = None
    energy_class: Optional[str] = None
    energy_kwh_m2: Optional[float] = None

    # Confidence
    data_confidence: float = 0.0
    data_source: str = "unknown"

    # Flags
    skip_reason: Optional[str] = None
    needs_deep_analysis: bool = False
    needs_qc: bool = False


class BuildingPrioritizer:
    """
    Prioritizes buildings for portfolio analysis.

    Assigns tier and priority score based on strategy and building data.
    """

    # Energy class priority (lower = higher priority for optimization)
    ENERGY_CLASS_PRIORITY = {
        "G": 0,
        "F": 1,
        "E": 2,
        "D": 3,
        "C": 4,
        "B": 5,
        "A": 6,
        None: 3,  # Default to middle
    }

    # Estimated savings potential by energy class (kWh/m²)
    SAVINGS_POTENTIAL = {
        "G": 80,
        "F": 60,
        "E": 40,
        "D": 25,
        "C": 15,
        "B": 8,
        "A": 3,
    }

    def __init__(
        self,
        strategy: PrioritizationStrategy = PrioritizationStrategy.HIGHEST_ROI_POTENTIAL,
        skip_optimized: bool = True,
        skip_energy_classes: Tuple[str, ...] = ("A", "B"),
    ):
        """
        Initialize prioritizer.

        Args:
            strategy: Prioritization strategy
            skip_optimized: Whether to skip already-optimized buildings
            skip_energy_classes: Energy classes to skip if skip_optimized=True
        """
        self.strategy = strategy
        self.skip_optimized = skip_optimized
        self.skip_energy_classes = skip_energy_classes

    def triage_building(
        self,
        address: str,
        building_data: Optional[Dict[str, Any]] = None,
    ) -> TriageResult:
        """
        Triage a single building.

        Args:
            address: Building address
            building_data: Optional data from GeoJSON/API

        Returns:
            TriageResult with tier, priority, and flags
        """
        result = TriageResult(
            address=address,
            tier=AnalysisTier.STANDARD,  # Default
            priority_score=0.5,
            priority_reason="default",
        )

        if building_data:
            result.construction_year = building_data.get("construction_year")
            result.atemp_m2 = building_data.get("atemp_m2")
            result.energy_class = building_data.get("energy_class")
            result.energy_kwh_m2 = building_data.get("energy_performance_kwh_m2")
            result.data_source = building_data.get("source", "geojson")
            result.data_confidence = building_data.get("confidence", 0.8)

            # Check if should skip
            if self.skip_optimized and result.energy_class in self.skip_energy_classes:
                result.tier = AnalysisTier.SKIP
                result.skip_reason = f"Already optimized (energy class {result.energy_class})"
                result.priority_score = 0.0
                return result

            # Determine tier based on data quality
            if result.data_confidence >= 0.8:
                result.tier = AnalysisTier.FAST
            elif result.data_confidence >= 0.5:
                result.tier = AnalysisTier.STANDARD
            else:
                result.tier = AnalysisTier.DEEP
                result.needs_deep_analysis = True

            # Calculate priority score
            result.priority_score = self._calculate_priority(result)
            result.priority_reason = self._get_priority_reason(result)

        else:
            # No data - needs full analysis
            result.tier = AnalysisTier.STANDARD
            result.data_source = "none"
            result.data_confidence = 0.0
            result.needs_qc = True
            result.priority_score = 0.5
            result.priority_reason = "no_data_available"

        return result

    def _calculate_priority(self, result: TriageResult) -> float:
        """Calculate priority score based on strategy."""
        if self.strategy == PrioritizationStrategy.HIGHEST_CONSUMPTION_FIRST:
            # Higher consumption = higher priority
            if result.energy_kwh_m2:
                return min(1.0, result.energy_kwh_m2 / 200.0)
            return 0.5

        elif self.strategy == PrioritizationStrategy.LOWEST_ENERGY_CLASS_FIRST:
            # G=1.0, A=0.0
            class_priority = self.ENERGY_CLASS_PRIORITY.get(result.energy_class, 3)
            return 1.0 - (class_priority / 6.0)

        elif self.strategy == PrioritizationStrategy.HIGHEST_ROI_POTENTIAL:
            # Combine savings potential with building size
            savings = self.SAVINGS_POTENTIAL.get(result.energy_class, 25)
            size_factor = min(1.0, (result.atemp_m2 or 1000) / 5000.0)
            return 0.7 * (savings / 80.0) + 0.3 * size_factor

        elif self.strategy == PrioritizationStrategy.QUICKEST_PAYBACK:
            # Quick payback = low investment, high savings
            # Older buildings often have quick payback for basic measures
            if result.construction_year:
                age_factor = max(0, min(1.0, (2025 - result.construction_year) / 80.0))
                savings = self.SAVINGS_POTENTIAL.get(result.energy_class, 25)
                return 0.5 * age_factor + 0.5 * (savings / 80.0)
            return 0.5

        elif self.strategy == PrioritizationStrategy.OLDEST_FIRST:
            if result.construction_year:
                return max(0, min(1.0, (2025 - result.construction_year) / 100.0))
            return 0.5

        elif self.strategy == PrioritizationStrategy.LARGEST_FIRST:
            if result.atemp_m2:
                return min(1.0, result.atemp_m2 / 10000.0)
            return 0.5

        elif self.strategy == PrioritizationStrategy.HIGHEST_CONFIDENCE_FIRST:
            return result.data_confidence

        elif self.strategy == PrioritizationStrategy.LOWEST_CONFIDENCE_FIRST:
            return 1.0 - result.data_confidence

        return 0.5

    def _get_priority_reason(self, result: TriageResult) -> str:
        """Get human-readable priority reason."""
        if self.strategy == PrioritizationStrategy.HIGHEST_CONSUMPTION_FIRST:
            return f"consumption={result.energy_kwh_m2}kWh/m²"

        elif self.strategy == PrioritizationStrategy.LOWEST_ENERGY_CLASS_FIRST:
            return f"energy_class={result.energy_class}"

        elif self.strategy == PrioritizationStrategy.HIGHEST_ROI_POTENTIAL:
            return f"roi_potential (class={result.energy_class}, size={result.atemp_m2}m²)"

        elif self.strategy == PrioritizationStrategy.QUICKEST_PAYBACK:
            return f"payback_potential (year={result.construction_year})"

        elif self.strategy == PrioritizationStrategy.OLDEST_FIRST:
            return f"age (year={result.construction_year})"

        elif self.strategy == PrioritizationStrategy.LARGEST_FIRST:
            return f"size={result.atemp_m2}m²"

        elif self.strategy == PrioritizationStrategy.HIGHEST_CONFIDENCE_FIRST:
            return f"confidence={result.data_confidence:.0%}"

        elif self.strategy == PrioritizationStrategy.LOWEST_CONFIDENCE_FIRST:
            return f"low_confidence={result.data_confidence:.0%}"

        return "unknown"

    def prioritize_portfolio(
        self,
        buildings: List[Tuple[str, Optional[Dict[str, Any]]]],
    ) -> List[TriageResult]:
        """
        Triage and prioritize a portfolio of buildings.

        Args:
            buildings: List of (address, building_data) tuples

        Returns:
            List of TriageResults sorted by priority (highest first)
        """
        results = [
            self.triage_building(addr, data)
            for addr, data in buildings
        ]

        # Sort by priority (highest first), skipped last
        results.sort(
            key=lambda r: (
                0 if r.tier == AnalysisTier.SKIP else 1,
                -r.priority_score,
            )
        )

        return results


def quick_triage(
    addresses: List[str],
    strategy: PrioritizationStrategy = PrioritizationStrategy.HIGHEST_ROI_POTENTIAL,
) -> List[TriageResult]:
    """
    Quick triage of addresses using Sweden Buildings GeoJSON.

    Args:
        addresses: List of addresses
        strategy: Prioritization strategy

    Returns:
        Prioritized list of TriageResults
    """
    # Load GeoJSON
    try:
        from src.ingest import load_sweden_buildings
        loader = load_sweden_buildings()
    except Exception:
        loader = None

    prioritizer = BuildingPrioritizer(strategy=strategy)

    buildings = []
    for addr in addresses:
        building_data = None
        if loader:
            matches = loader.find_by_address(addr)
            if matches:
                b = matches[0]
                building_data = {
                    "construction_year": b.construction_year,
                    "atemp_m2": b.atemp_m2,
                    "energy_class": b.energy_class,
                    "energy_performance_kwh_m2": b.energy_performance_kwh_m2,
                    "source": "geojson",
                    "confidence": 0.85,
                }
        buildings.append((addr, building_data))

    return prioritizer.prioritize_portfolio(buildings)
