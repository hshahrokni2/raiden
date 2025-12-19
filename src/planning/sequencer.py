"""
ECM Sequencer - Optimal ordering of energy investments.

Determines the best sequence and timing of ECM investments considering:
- Cash flow constraints (fund balance, loan capacity)
- Renovation synergies (combine insulation with facade work)
- Payback cascade (quick wins fund larger investments)
- Energy price escalation (earlier = more cumulative savings)
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import date
import logging
from copy import deepcopy

from .models import (
    MaintenancePlan,
    ECMInvestment,
    PlannedRenovation,
    BRFFinancials,
    FundingSource,
)
from .cash_flow import CashFlowSimulator, SimulationConfig

logger = logging.getLogger(__name__)


@dataclass
class SequencingStrategy:
    """Strategy parameters for ECM sequencing."""
    # Timing preferences
    start_with_zero_cost: bool = True    # Always start with Steg 0
    years_between_investments: int = 1    # Minimum gap between major investments
    coordinate_with_renovations: bool = True  # Align ECMs with planned renovations

    # Financial constraints
    max_investment_per_year: float = 500_000  # Unless coordinated with renovation
    maintain_fund_cushion_sek: float = 200_000  # Keep this much in fund

    # Optimization goals
    optimize_for: str = "npv"  # "npv", "payback", "cash_flow", "avgift"


@dataclass
class ECMCandidate:
    """An ECM candidate for sequencing."""
    ecm_id: str
    name: str
    investment_sek: float
    annual_savings_sek: float
    payback_years: float
    priority_score: float = 0.0

    # Coordination
    synergy_with_renovation: Optional[str] = None
    cost_if_coordinated: float = 0
    savings_if_coordinated: float = 0

    # Category
    is_zero_cost: bool = False
    steg: int = 0  # 0, 1, 2, or 3


class ECMSequencer:
    """
    Determine optimal ECM investment sequence.

    Usage:
        sequencer = ECMSequencer()
        candidates = [ECMCandidate(...), ...]
        plan = sequencer.create_optimal_plan(
            candidates=candidates,
            financials=brf_financials,
            renovations=planned_renovations,
        )
    """

    def __init__(self, strategy: SequencingStrategy = None):
        self.strategy = strategy or SequencingStrategy()

    def create_optimal_plan(
        self,
        candidates: List[ECMCandidate],
        financials: BRFFinancials,
        renovations: List[PlannedRenovation],
        start_year: int = None,
        plan_horizon_years: int = 30,
    ) -> MaintenancePlan:
        """
        Create an optimally sequenced maintenance plan.

        Args:
            candidates: Available ECM options
            financials: BRF financial parameters
            renovations: Planned renovations (for coordination)
            start_year: When to start (default: current year)
            plan_horizon_years: How far to plan

        Returns:
            MaintenancePlan with optimally sequenced ECMs
        """
        start_year = start_year or date.today().year

        # Step 1: Score and rank candidates
        scored_candidates = self._score_candidates(candidates, renovations)

        # Step 2: Identify renovation synergies
        synergy_map = self._map_synergies(scored_candidates, renovations)

        # Step 3: Create initial sequence
        sequence = self._create_sequence(
            scored_candidates,
            synergy_map,
            financials,
            start_year,
            plan_horizon_years,
        )

        # Step 4: Build the plan
        plan = self._build_plan(
            sequence,
            financials,
            renovations,
            start_year,
            plan_horizon_years,
        )

        # Step 5: Simulate to verify feasibility
        simulator = CashFlowSimulator()
        plan = simulator.simulate(plan, start_year)

        # Step 6: Adjust if needed (fund warnings, etc.)
        plan = self._adjust_for_feasibility(plan, start_year)

        return plan

    def _score_candidates(
        self,
        candidates: List[ECMCandidate],
        renovations: List[PlannedRenovation],
    ) -> List[ECMCandidate]:
        """Score and rank ECM candidates."""
        scored = []

        for candidate in candidates:
            # Base score from ROI
            if candidate.payback_years > 0:
                roi_score = 100 / candidate.payback_years  # Higher = better
            else:
                roi_score = 100  # Zero-cost = top score

            # Bonus for zero-cost
            if candidate.is_zero_cost:
                roi_score += 50

            # Bonus for renovation synergy
            if candidate.synergy_with_renovation:
                roi_score += 20

            # Penalty for very high investment
            if candidate.investment_sek > 1_000_000:
                roi_score -= 10

            candidate.priority_score = roi_score
            scored.append(candidate)

        # Sort by priority (highest first)
        return sorted(scored, key=lambda x: x.priority_score, reverse=True)

    def _map_synergies(
        self,
        candidates: List[ECMCandidate],
        renovations: List[PlannedRenovation],
    ) -> Dict[str, Tuple[str, int]]:
        """Map ECMs to renovation synergies."""
        synergy_map = {}

        # Build renovation lookup
        renovation_ecms = {}
        for reno in renovations:
            for ecm_id in reno.ecm_synergy:
                renovation_ecms[ecm_id] = (reno.id, reno.planned_year)

        # Map candidates to renovations
        for candidate in candidates:
            if candidate.ecm_id in renovation_ecms:
                reno_id, reno_year = renovation_ecms[candidate.ecm_id]
                synergy_map[candidate.ecm_id] = (reno_id, reno_year)
                candidate.synergy_with_renovation = reno_id

        return synergy_map

    def _create_sequence(
        self,
        candidates: List[ECMCandidate],
        synergy_map: Dict[str, Tuple[str, int]],
        financials: BRFFinancials,
        start_year: int,
        horizon_years: int,
    ) -> List[Tuple[int, ECMCandidate]]:
        """Create year-by-year sequence of ECM investments."""
        sequence = []
        current_year = start_year
        fund_projection = financials.current_fund_sek

        # Separate by category
        zero_cost = [c for c in candidates if c.is_zero_cost]
        capital = [c for c in candidates if not c.is_zero_cost]

        # Step 1: All zero-cost measures in year 1
        if self.strategy.start_with_zero_cost and zero_cost:
            for candidate in zero_cost:
                sequence.append((start_year, candidate))

        # Step 2: Capital measures by priority, respecting constraints
        years_used = {start_year}  # Track which years have investments
        annual_savings = sum(c.annual_savings_sek for c in zero_cost)

        for candidate in capital:
            # Check for renovation synergy
            if candidate.ecm_id in synergy_map:
                reno_id, reno_year = synergy_map[candidate.ecm_id]
                # Schedule with renovation
                sequence.append((reno_year, candidate))
                years_used.add(reno_year)
                continue

            # Find next available year
            year = current_year + 1
            while year in years_used:
                year += 1

            # Check if we can afford it
            years_of_savings = year - start_year
            projected_fund = (
                financials.current_fund_sek
                + financials.annual_fund_contribution_sek * years_of_savings
                + annual_savings * years_of_savings
            )

            if projected_fund >= candidate.investment_sek + self.strategy.maintain_fund_cushion_sek:
                sequence.append((year, candidate))
                years_used.add(year)
                annual_savings += candidate.annual_savings_sek

                # Respect gap between investments
                current_year = year + self.strategy.years_between_investments - 1

            elif year < start_year + horizon_years:
                # Postpone until affordable
                while projected_fund < candidate.investment_sek + self.strategy.maintain_fund_cushion_sek:
                    year += 1
                    years_of_savings = year - start_year
                    projected_fund = (
                        financials.current_fund_sek
                        + financials.annual_fund_contribution_sek * years_of_savings
                        + annual_savings * years_of_savings
                    )
                    if year > start_year + horizon_years:
                        break

                if year <= start_year + horizon_years:
                    sequence.append((year, candidate))
                    years_used.add(year)
                    annual_savings += candidate.annual_savings_sek

        # Sort by year
        return sorted(sequence, key=lambda x: x[0])

    def _build_plan(
        self,
        sequence: List[Tuple[int, ECMCandidate]],
        financials: BRFFinancials,
        renovations: List[PlannedRenovation],
        start_year: int,
        horizon_years: int,
    ) -> MaintenancePlan:
        """Build MaintenancePlan from sequence."""
        ecm_investments = []

        for year, candidate in sequence:
            # Check if coordinated with renovation
            cost = candidate.investment_sek
            if candidate.synergy_with_renovation:
                cost = candidate.cost_if_coordinated or cost * 0.8  # 20% discount

            investment = ECMInvestment(
                ecm_id=candidate.ecm_id,
                name=candidate.name,
                name_sv=candidate.name,
                planned_year=year,
                investment_sek=cost,
                annual_savings_sek=candidate.annual_savings_sek,
                funding_source=FundingSource.FUND if candidate.is_zero_cost else FundingSource.MIXED,
                coordinated_with=candidate.synergy_with_renovation,
                cost_if_standalone=candidate.investment_sek,
            )
            ecm_investments.append(investment)

        plan = MaintenancePlan(
            brf_name="",  # Will be filled by caller
            plan_horizon_years=horizon_years,
            financials=financials,
            renovations=renovations,
            ecm_investments=ecm_investments,
        )

        return plan

    def _adjust_for_feasibility(
        self,
        plan: MaintenancePlan,
        start_year: int,
    ) -> MaintenancePlan:
        """Adjust plan if simulation shows problems."""
        if not plan.projections:
            return plan

        # Check for fund warnings
        fund_warnings = [p for p in plan.projections if p.fund_warning]

        if fund_warnings:
            logger.warning(
                f"Plan has {len(fund_warnings)} years with low fund balance. "
                "Consider postponing some investments."
            )

            # Simple adjustment: postpone non-coordinated ECMs by 1 year
            # until feasible (would need iterative simulation for full fix)

        return plan


def create_cascade_sequence(
    ecm_results: List[Dict],
    financials: BRFFinancials,
    renovations: List[PlannedRenovation] = None,
    start_year: int = None,
) -> MaintenancePlan:
    """
    Convenience function to create a cash-flow cascade plan.

    Args:
        ecm_results: Results from ECM simulation (id, savings_percent, cost, etc.)
        financials: BRF financial parameters
        renovations: Optional planned renovations
        start_year: Start year for plan

    Returns:
        Optimally sequenced MaintenancePlan
    """
    from ..roi.costs_sweden import ECM_COSTS, CostCategory

    start_year = start_year or date.today().year

    # Convert ECM results to candidates
    candidates = []
    for ecm in ecm_results:
        ecm_id = ecm.get('id', ecm.get('ecm_id'))
        cost_data = ECM_COSTS.get(ecm_id)

        is_zero_cost = (
            cost_data and cost_data.category == CostCategory.ZERO_COST
        )

        # Determine steg
        if is_zero_cost:
            steg = 0
        elif ecm.get('cost_sek', 0) < 500_000:
            steg = 1
        elif ecm.get('cost_sek', 0) < 2_000_000:
            steg = 2
        else:
            steg = 3

        payback = ecm.get('payback_years', 999)
        if ecm.get('annual_savings_sek', 0) > 0:
            payback = ecm.get('cost_sek', 0) / ecm.get('annual_savings_sek', 1)

        candidate = ECMCandidate(
            ecm_id=ecm_id,
            name=ecm.get('name', ecm_id),
            investment_sek=ecm.get('cost_sek', 0),
            annual_savings_sek=ecm.get('annual_savings_sek', 0),
            payback_years=payback,
            is_zero_cost=is_zero_cost,
            steg=steg,
        )
        candidates.append(candidate)

    # Create sequencer and build plan
    sequencer = ECMSequencer()
    plan = sequencer.create_optimal_plan(
        candidates=candidates,
        financials=financials,
        renovations=renovations or [],
        start_year=start_year,
    )

    return plan
