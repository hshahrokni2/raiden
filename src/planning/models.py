"""
Maintenance Plan Data Models.

Data structures for BRF maintenance planning with energy integration.
Supports cash flow simulation, ECM sequencing, and long-term forecasting.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import date
from enum import Enum


class RenovationType(Enum):
    """Types of planned renovations."""
    FACADE = "facade"           # Fasadrenovering
    ROOF = "roof"               # Takrenovering
    WINDOWS = "windows"         # Fönsterbyte
    PLUMBING = "plumbing"       # Stambyte
    ELECTRICAL = "electrical"   # Elrenovering
    BALCONIES = "balconies"     # Balkongrenoverig
    ELEVATOR = "elevator"       # Hissrenovering
    VENTILATION = "ventilation" # Ventilationsrenovering
    HEATING = "heating"         # Värmesystem
    OTHER = "other"


class LoanTolerance(Enum):
    """BRF loan tolerance level."""
    NONE = "none"               # Inga lån, endast underhållsfond
    SMALL = "small"             # Små lån ok (< 1M SEK)
    MODERATE = "moderate"       # Måttliga lån (1-5M SEK)
    FLEXIBLE = "flexible"       # Flexibel, kan ta större lån


class FundingSource(Enum):
    """How an investment is funded."""
    FUND = "fund"               # Underhållsfond
    LOAN = "loan"               # Lån
    AVGIFT = "avgift"           # Avgiftshöjning
    MIXED = "mixed"             # Kombination
    SAVINGS = "savings"         # Ackumulerade energibesparingar


@dataclass
class PlannedRenovation:
    """A planned renovation in the maintenance plan."""
    id: str
    name: str
    name_sv: str
    type: RenovationType
    planned_year: int
    estimated_cost_sek: float
    description: str = ""

    # ECM integration opportunity
    ecm_synergy: List[str] = field(default_factory=list)  # ECMs that can be combined
    ecm_cost_reduction: float = 0.0  # Cost reduction if combined with ECM

    # Flexibility
    can_postpone_years: int = 0  # How many years can this be postponed?
    can_advance_years: int = 0   # How many years can this be advanced?

    # Status
    is_mandatory: bool = False   # Legal/safety requirement
    notes: str = ""


@dataclass
class TariffStructure:
    """Energy tariff structure for cost calculations."""
    # Electricity
    el_energy_sek_kwh: float = 1.50      # Energy price
    el_grid_fee_sek_kwh: float = 0.50    # Nätavgift
    el_peak_sek_kw_month: float = 59.0   # Effektavgift (Ellevio)
    el_fuse_sek_month: float = 500.0     # Säkringsavgift

    # District heating (fjärrvärme)
    fv_energy_sek_kwh: float = 0.80      # Energy price
    fv_peak_sek_kw_year: float = 400.0   # Effektavgift (annual)
    fv_fixed_sek_year: float = 5000.0    # Fast avgift

    # Price escalation assumptions
    el_annual_increase: float = 0.03     # 3% per year
    fv_annual_increase: float = 0.02     # 2% per year

    # Provider info
    el_provider: str = "Ellevio"
    fv_provider: str = "Stockholm Exergi"


@dataclass
class BRFFinancials:
    """BRF financial parameters."""
    # Required fields (no defaults) first
    current_fund_sek: float              # Current balance
    annual_fund_contribution_sek: float  # Yearly avsättning
    current_avgift_sek_month: float      # Per apartment average
    num_apartments: int

    # Optional fields with defaults
    target_fund_sek: float = 0           # Target balance (optional)
    avgift_increase_tolerance_pct: float = 5.0  # Max acceptable increase

    # Loan parameters
    loan_tolerance: LoanTolerance = LoanTolerance.SMALL
    current_loans_sek: float = 0
    max_loan_sek: float = 0              # Board-approved maximum
    loan_interest_rate: float = 0.045    # 4.5%
    loan_term_years: int = 20

    # Energy costs (current)
    annual_energy_cost_sek: float = 0    # Total energy bill
    annual_el_cost_sek: float = 0
    annual_fv_cost_sek: float = 0
    peak_el_kw: float = 0                # Current peak demand
    peak_fv_kw: float = 0


@dataclass
class ECMInvestment:
    """An ECM investment in the plan."""
    ecm_id: str
    name: str
    name_sv: str

    # Timing
    planned_year: int

    # Costs
    investment_sek: float
    annual_savings_sek: float
    annual_maintenance_sek: float = 0

    # Energy impact
    energy_savings_kwh: float = 0
    peak_reduction_kw: float = 0         # For effektvakt calculation

    # Funding
    funding_source: FundingSource = FundingSource.FUND

    # Coordination
    coordinated_with: Optional[str] = None  # Renovation ID if combined
    cost_if_standalone: float = 0        # Higher cost if not combined

    # Results (filled after simulation)
    actual_year: int = 0                 # May differ from planned
    cumulative_savings_sek: float = 0
    payback_achieved_year: int = 0


@dataclass
class YearlyProjection:
    """Projection for a single year."""
    year: int

    # Actions this year
    renovations: List[str] = field(default_factory=list)  # Renovation IDs
    ecm_investments: List[str] = field(default_factory=list)  # ECM IDs

    # Cash flow
    fund_start_sek: float = 0
    fund_contribution_sek: float = 0
    renovation_spend_sek: float = 0
    ecm_investment_sek: float = 0
    energy_savings_sek: float = 0        # Cumulative annual savings
    loan_payment_sek: float = 0
    fund_end_sek: float = 0

    # Loans
    new_loan_sek: float = 0
    loan_balance_sek: float = 0

    # Energy
    energy_cost_sek: float = 0           # After savings
    baseline_energy_cost_sek: float = 0  # Without ECMs
    cumulative_savings_sek: float = 0

    # Avgift impact
    required_avgift_change_pct: float = 0

    # Status flags
    fund_warning: bool = False           # Fund below safe level
    loan_warning: bool = False           # Approaching loan limit
    avgift_warning: bool = False         # Avgift increase needed


@dataclass
class MaintenancePlan:
    """Complete BRF maintenance plan with energy integration."""
    # Metadata
    brf_name: str
    org_number: str = ""
    address: str = ""
    created_date: date = field(default_factory=date.today)
    plan_horizon_years: int = 30

    # Building data
    atemp_m2: float = 0
    num_apartments: int = 0
    construction_year: int = 0

    # Financial parameters
    financials: BRFFinancials = None
    tariffs: TariffStructure = field(default_factory=TariffStructure)

    # Planned items
    renovations: List[PlannedRenovation] = field(default_factory=list)
    ecm_investments: List[ECMInvestment] = field(default_factory=list)

    # Simulation results
    projections: List[YearlyProjection] = field(default_factory=list)

    # Summary metrics (filled after simulation)
    total_investment_sek: float = 0
    total_savings_30yr_sek: float = 0
    net_present_value_sek: float = 0
    break_even_year: int = 0
    final_fund_balance_sek: float = 0
    max_loan_used_sek: float = 0
    avgift_change_required_pct: float = 0


@dataclass
class PlanScenario:
    """A scenario for comparison (with/without ECMs, different timing, etc.)."""
    id: str
    name: str
    description: str

    # The plan
    plan: MaintenancePlan

    # Key metrics for comparison
    total_cost_30yr_sek: float = 0       # Total spend over 30 years
    total_energy_cost_30yr_sek: float = 0
    avgift_impact_pct: float = 0
    fund_minimum_sek: float = 0          # Lowest fund balance
    loan_maximum_sek: float = 0

    # Comparison flags
    is_baseline: bool = False            # "Do nothing" scenario
    is_recommended: bool = False


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_sample_financials(
    fund_balance: float = 500_000,
    annual_contribution: float = 200_000,
    num_apartments: int = 50,
    avgift_per_month: float = 4500,
) -> BRFFinancials:
    """Create sample BRF financials for testing."""
    return BRFFinancials(
        current_fund_sek=fund_balance,
        annual_fund_contribution_sek=annual_contribution,
        current_avgift_sek_month=avgift_per_month,
        num_apartments=num_apartments,
        avgift_increase_tolerance_pct=5.0,
        loan_tolerance=LoanTolerance.SMALL,
        max_loan_sek=2_000_000,
    )


def create_typical_renovation_plan(start_year: int = 2025) -> List[PlannedRenovation]:
    """Create typical Swedish BRF renovation plan."""
    return [
        PlannedRenovation(
            id="facade_2030",
            name="Facade Renovation",
            name_sv="Fasadrenovering",
            type=RenovationType.FACADE,
            planned_year=start_year + 5,
            estimated_cost_sek=5_000_000,
            ecm_synergy=["wall_external_insulation"],
            ecm_cost_reduction=500_000,  # Scaffolding already there
            can_postpone_years=2,
            description="Putsrenovering med möjlighet till tilläggsisolering"
        ),
        PlannedRenovation(
            id="roof_2035",
            name="Roof Renovation",
            name_sv="Takrenovering",
            type=RenovationType.ROOF,
            planned_year=start_year + 10,
            estimated_cost_sek=2_000_000,
            ecm_synergy=["roof_insulation", "solar_pv"],
            ecm_cost_reduction=200_000,
            description="Taktäckning med möjlighet till isolering och solceller"
        ),
        PlannedRenovation(
            id="windows_2032",
            name="Window Replacement",
            name_sv="Fönsterbyte",
            type=RenovationType.WINDOWS,
            planned_year=start_year + 7,
            estimated_cost_sek=3_000_000,
            ecm_synergy=["window_replacement"],
            ecm_cost_reduction=0,  # Windows ARE the ECM
            is_mandatory=False,
            description="Byte till energieffektiva fönster"
        ),
    ]
