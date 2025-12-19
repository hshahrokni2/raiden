"""
Cash Flow Simulator for BRF Maintenance Planning.

Simulates year-by-year cash flow for a BRF considering:
- Planned renovations
- ECM investments and savings
- Loan amortization
- Energy cost escalation
- Fund balance management
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import date
import logging

from .models import (
    MaintenancePlan,
    YearlyProjection,
    BRFFinancials,
    TariffStructure,
    PlannedRenovation,
    ECMInvestment,
    FundingSource,
    LoanTolerance,
)

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for cash flow simulation."""
    discount_rate: float = 0.03          # For NPV calculations
    min_fund_balance_sek: float = 100_000  # Warning threshold
    inflation_rate: float = 0.02         # General cost inflation
    include_loan_interest: bool = True
    max_simulation_years: int = 30


class CashFlowSimulator:
    """
    Simulate BRF cash flow over time.

    Usage:
        simulator = CashFlowSimulator()
        plan = MaintenancePlan(...)
        results = simulator.simulate(plan)
    """

    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()

    def simulate(
        self,
        plan: MaintenancePlan,
        start_year: int = None,
    ) -> MaintenancePlan:
        """
        Run cash flow simulation and populate projections.

        Args:
            plan: MaintenancePlan with renovations and ECMs
            start_year: Override start year (default: current year)

        Returns:
            MaintenancePlan with projections filled in
        """
        start_year = start_year or date.today().year
        end_year = start_year + plan.plan_horizon_years

        # Initialize tracking variables
        fund_balance = plan.financials.current_fund_sek
        loan_balance = plan.financials.current_loans_sek
        cumulative_energy_savings = 0.0
        cumulative_total_savings = 0.0
        baseline_energy_cost = plan.financials.annual_energy_cost_sek

        # Build lookup for investments by year
        renovations_by_year = self._group_by_year(plan.renovations)
        ecms_by_year = self._group_by_year(plan.ecm_investments)

        projections = []

        for year in range(start_year, end_year + 1):
            # Get items for this year
            year_renovations = renovations_by_year.get(year, [])
            year_ecms = ecms_by_year.get(year, [])

            # Calculate costs
            renovation_cost = sum(r.estimated_cost_sek for r in year_renovations)
            ecm_cost = sum(e.investment_sek for e in year_ecms)

            # Calculate new energy savings from this year's ECMs
            new_annual_savings = sum(e.annual_savings_sek for e in year_ecms)
            cumulative_energy_savings += new_annual_savings

            # Energy cost with escalation
            years_elapsed = year - start_year
            escalation_factor = (1 + plan.tariffs.el_annual_increase) ** years_elapsed
            baseline_escalated = baseline_energy_cost * escalation_factor
            actual_energy_cost = baseline_escalated - cumulative_energy_savings

            # Determine funding strategy
            total_investment = renovation_cost + ecm_cost
            funding_result = self._determine_funding(
                investment=total_investment,
                fund_balance=fund_balance,
                loan_balance=loan_balance,
                financials=plan.financials,
            )

            # Apply funding
            fund_spend = funding_result['fund_spend']
            new_loan = funding_result['new_loan']

            # Loan payment (simplified: interest only during construction, then amortization)
            loan_balance += new_loan
            loan_payment = self._calculate_loan_payment(
                loan_balance,
                plan.financials.loan_interest_rate,
                plan.financials.loan_term_years,
            )

            # Update fund balance
            fund_start = fund_balance
            fund_balance = (
                fund_balance
                + plan.financials.annual_fund_contribution_sek
                - fund_spend
                + cumulative_energy_savings  # Savings go back to fund
                - loan_payment
            )

            # Cumulative tracking
            cumulative_total_savings += cumulative_energy_savings

            # Check warnings
            fund_warning = fund_balance < self.config.min_fund_balance_sek
            loan_warning = loan_balance > plan.financials.max_loan_sek * 0.9
            avgift_change = self._calculate_avgift_impact(
                fund_balance, loan_balance, plan.financials
            )
            avgift_warning = avgift_change > plan.financials.avgift_increase_tolerance_pct

            # Create projection
            projection = YearlyProjection(
                year=year,
                renovations=[r.id for r in year_renovations],
                ecm_investments=[e.ecm_id for e in year_ecms],
                fund_start_sek=fund_start,
                fund_contribution_sek=plan.financials.annual_fund_contribution_sek,
                renovation_spend_sek=renovation_cost,
                ecm_investment_sek=ecm_cost,
                energy_savings_sek=cumulative_energy_savings,
                loan_payment_sek=loan_payment,
                fund_end_sek=fund_balance,
                new_loan_sek=new_loan,
                loan_balance_sek=loan_balance,
                energy_cost_sek=actual_energy_cost,
                baseline_energy_cost_sek=baseline_escalated,
                cumulative_savings_sek=cumulative_total_savings,
                required_avgift_change_pct=avgift_change,
                fund_warning=fund_warning,
                loan_warning=loan_warning,
                avgift_warning=avgift_warning,
            )

            projections.append(projection)

            # Amortize loan
            if loan_balance > 0:
                loan_balance = max(0, loan_balance - loan_payment * 0.5)  # Simplified

        # Update plan with results
        plan.projections = projections

        # Calculate summary metrics
        self._calculate_summary_metrics(plan, start_year)

        return plan

    def _group_by_year(self, items: List) -> Dict[int, List]:
        """Group items by their planned year."""
        result = {}
        for item in items:
            year = item.planned_year if hasattr(item, 'planned_year') else item.planned_year
            if year not in result:
                result[year] = []
            result[year].append(item)
        return result

    def _determine_funding(
        self,
        investment: float,
        fund_balance: float,
        loan_balance: float,
        financials: BRFFinancials,
    ) -> Dict[str, float]:
        """Determine how to fund an investment."""
        if investment == 0:
            return {'fund_spend': 0, 'new_loan': 0}

        # Strategy: Use fund first, then loan if allowed
        available_fund = max(0, fund_balance - self.config.min_fund_balance_sek)

        if investment <= available_fund:
            # Can cover entirely from fund
            return {'fund_spend': investment, 'new_loan': 0}

        # Need additional funding
        fund_spend = available_fund
        remaining = investment - fund_spend

        # Check loan tolerance
        loan_headroom = financials.max_loan_sek - loan_balance

        if financials.loan_tolerance == LoanTolerance.NONE:
            # No loans allowed - will show warning
            return {'fund_spend': fund_balance, 'new_loan': 0}

        if remaining <= loan_headroom:
            return {'fund_spend': fund_spend, 'new_loan': remaining}

        # Over loan limit - take what we can
        return {'fund_spend': fund_spend, 'new_loan': loan_headroom}

    def _calculate_loan_payment(
        self,
        balance: float,
        interest_rate: float,
        term_years: int,
    ) -> float:
        """Calculate annual loan payment (interest + principal)."""
        if balance <= 0:
            return 0

        # Simple annuity calculation
        r = interest_rate
        n = term_years
        if r == 0:
            return balance / n

        payment = balance * (r * (1 + r) ** n) / ((1 + r) ** n - 1)
        return payment

    def _calculate_avgift_impact(
        self,
        fund_balance: float,
        loan_balance: float,
        financials: BRFFinancials,
    ) -> float:
        """Calculate required avgift change percentage."""
        # Simplified: if fund is low and loans are high, avgift needs to increase
        target_fund = financials.target_fund_sek or financials.annual_fund_contribution_sek * 3

        if fund_balance >= target_fund and loan_balance <= financials.current_loans_sek:
            return 0.0

        # Calculate shortfall
        fund_shortfall = max(0, target_fund - fund_balance)
        extra_loan_cost = max(0, loan_balance - financials.current_loans_sek) * 0.05

        annual_shortfall = fund_shortfall / 5 + extra_loan_cost  # Spread over 5 years

        current_annual_avgift = (
            financials.current_avgift_sek_month * 12 * financials.num_apartments
        )

        if current_annual_avgift > 0:
            return (annual_shortfall / current_annual_avgift) * 100

        return 0.0

    def _calculate_summary_metrics(
        self,
        plan: MaintenancePlan,
        start_year: int,
    ):
        """Calculate summary metrics from projections."""
        if not plan.projections:
            return

        # Total investment
        plan.total_investment_sek = sum(
            p.renovation_spend_sek + p.ecm_investment_sek
            for p in plan.projections
        )

        # Total savings
        plan.total_savings_30yr_sek = plan.projections[-1].cumulative_savings_sek

        # NPV calculation
        npv = 0
        for i, proj in enumerate(plan.projections):
            year_savings = proj.energy_savings_sek
            year_cost = proj.ecm_investment_sek
            net = year_savings - year_cost
            npv += net / ((1 + self.config.discount_rate) ** i)
        plan.net_present_value_sek = npv

        # Break-even year
        cumulative_investment = 0
        cumulative_savings = 0
        for proj in plan.projections:
            cumulative_investment += proj.ecm_investment_sek
            cumulative_savings += proj.energy_savings_sek
            if cumulative_savings >= cumulative_investment and cumulative_investment > 0:
                plan.break_even_year = proj.year
                break

        # Final fund balance
        plan.final_fund_balance_sek = plan.projections[-1].fund_end_sek

        # Max loan
        plan.max_loan_used_sek = max(p.loan_balance_sek for p in plan.projections)

        # Max avgift change
        plan.avgift_change_required_pct = max(
            p.required_avgift_change_pct for p in plan.projections
        )


def simulate_baseline_scenario(
    plan: MaintenancePlan,
) -> MaintenancePlan:
    """
    Simulate baseline scenario (no ECMs, just renovations).

    Creates a copy of the plan without ECM investments.
    """
    from copy import deepcopy

    baseline = deepcopy(plan)
    baseline.ecm_investments = []

    simulator = CashFlowSimulator()
    return simulator.simulate(baseline)


def compare_scenarios(
    with_ecms: MaintenancePlan,
    without_ecms: MaintenancePlan,
) -> Dict[str, float]:
    """Compare two scenarios and return key differences."""
    return {
        'total_savings_difference_sek': (
            with_ecms.total_savings_30yr_sek - without_ecms.total_savings_30yr_sek
        ),
        'npv_difference_sek': (
            with_ecms.net_present_value_sek - without_ecms.net_present_value_sek
        ),
        'final_fund_difference_sek': (
            with_ecms.final_fund_balance_sek - without_ecms.final_fund_balance_sek
        ),
        'max_loan_difference_sek': (
            with_ecms.max_loan_used_sek - without_ecms.max_loan_used_sek
        ),
        'avgift_difference_pct': (
            with_ecms.avgift_change_required_pct - without_ecms.avgift_change_required_pct
        ),
    }
