"""
ROI Calculator - Calculate returns for ECM investments.

Metrics:
- Simple payback period
- Net Present Value (NPV)
- Internal Rate of Return (IRR)
- Levelized Cost of Energy Saved (LCES)
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
import math

from .costs_sweden import SwedishCosts
from ..ecm.combinations import ECMCombination, ECMVariant
from ..simulation.results import AnnualResults


@dataclass
class ROIResult:
    """ROI calculation results for an ECM combination."""
    combination_id: str
    combination_name: str

    # Costs
    total_investment_sek: float
    annual_maintenance_sek: float

    # Savings
    annual_energy_savings_kwh: float
    annual_cost_savings_sek: float
    lifetime_savings_sek: float

    # ROI Metrics
    simple_payback_years: float
    npv_sek: float
    irr_percent: Optional[float]  # None if can't calculate
    lces_sek_per_kwh: float  # Levelized cost of energy saved

    # Environmental
    annual_co2_reduction_kg: float
    lifetime_co2_reduction_tonnes: float

    # Ranking score (higher = better)
    score: float

    # Details
    ecm_costs_breakdown: Dict[str, float]
    energy_type: str
    analysis_period_years: int


class ROICalculator:
    """
    Calculate ROI for ECM combinations.

    Usage:
        calculator = ROICalculator()
        roi = calculator.calculate(
            combination=ecm_combo,
            baseline_results=baseline,
            ecm_results=ecm_scenario,
            building_context=context
        )
    """

    DEFAULT_DISCOUNT_RATE = 0.04  # 4% real discount rate
    DEFAULT_ANALYSIS_PERIOD = 25  # years

    def __init__(
        self,
        costs: SwedishCosts = None,
        discount_rate: float = None,
        analysis_period: int = None
    ):
        self.costs = costs or SwedishCosts()
        self.discount_rate = discount_rate or self.DEFAULT_DISCOUNT_RATE
        self.analysis_period = analysis_period or self.DEFAULT_ANALYSIS_PERIOD

    def calculate(
        self,
        combination: ECMCombination,
        baseline_results: AnnualResults,
        ecm_results: AnnualResults,
        building_areas: Dict[str, float],  # 'floor', 'wall', 'window', 'roof'
        energy_type: str = 'district_heating'
    ) -> ROIResult:
        """
        Calculate ROI for an ECM combination.

        Args:
            combination: ECM combination that was simulated
            baseline_results: Results from baseline simulation
            ecm_results: Results from ECM scenario simulation
            building_areas: Areas for cost calculation
            energy_type: Primary heating energy type

        Returns:
            ROIResult with all financial metrics
        """
        # Calculate investment cost
        investment, cost_breakdown = self._calculate_investment(
            combination, building_areas
        )

        # Calculate energy savings
        heating_savings = baseline_results.heating_kwh - ecm_results.heating_kwh
        electricity_savings = (
            baseline_results.lighting_kwh + baseline_results.equipment_kwh + baseline_results.fan_kwh
        ) - (
            ecm_results.lighting_kwh + ecm_results.equipment_kwh + ecm_results.fan_kwh
        )

        # Determine which savings to use based on ECMs
        total_savings_kwh = heating_savings + electricity_savings

        # Calculate cost savings
        heating_cost_savings = self.costs.annual_savings(
            energy_type, baseline_results.heating_kwh, ecm_results.heating_kwh
        )
        electricity_cost_savings = self.costs.annual_savings(
            'electricity',
            baseline_results.lighting_kwh + baseline_results.equipment_kwh,
            ecm_results.lighting_kwh + ecm_results.equipment_kwh
        )
        annual_savings = heating_cost_savings + electricity_cost_savings

        # Annual maintenance cost
        annual_maintenance = investment * 0.01  # 1% of investment

        # Net annual savings
        net_annual_savings = annual_savings - annual_maintenance

        # Simple payback
        if net_annual_savings > 0:
            payback = investment / net_annual_savings
        else:
            payback = float('inf')

        # NPV calculation
        npv = self._calculate_npv(
            investment, net_annual_savings, self.analysis_period
        )

        # IRR calculation
        irr = self._calculate_irr(investment, net_annual_savings, self.analysis_period)

        # Levelized cost of energy saved
        if total_savings_kwh > 0:
            lces = investment / (total_savings_kwh * self.analysis_period)
        else:
            lces = float('inf')

        # CO2 reduction
        energy_cost = self.costs.energy_price(energy_type)
        annual_co2 = heating_savings * energy_cost.carbon_intensity_kg_per_kwh
        annual_co2 += electricity_savings * self.costs.energy_price('electricity').carbon_intensity_kg_per_kwh

        # Lifetime values
        lifetime_savings = net_annual_savings * self.analysis_period
        lifetime_co2 = annual_co2 * self.analysis_period / 1000  # tonnes

        # Calculate score (higher = better investment)
        # Weighted combination of payback, NPV, and CO2
        score = self._calculate_score(payback, npv, investment, annual_co2)

        return ROIResult(
            combination_id=combination.id,
            combination_name=combination.name,
            total_investment_sek=investment,
            annual_maintenance_sek=annual_maintenance,
            annual_energy_savings_kwh=total_savings_kwh,
            annual_cost_savings_sek=annual_savings,
            lifetime_savings_sek=lifetime_savings,
            simple_payback_years=payback,
            npv_sek=npv,
            irr_percent=irr,
            lces_sek_per_kwh=lces,
            annual_co2_reduction_kg=annual_co2,
            lifetime_co2_reduction_tonnes=lifetime_co2,
            score=score,
            ecm_costs_breakdown=cost_breakdown,
            energy_type=energy_type,
            analysis_period_years=self.analysis_period
        )

    def _calculate_investment(
        self,
        combination: ECMCombination,
        areas: Dict[str, float]
    ) -> tuple[float, Dict[str, float]]:
        """Calculate total investment and breakdown by ECM."""
        total = 0.0
        breakdown = {}

        for variant in combination.variants:
            ecm_id = variant.ecm.id

            # Determine quantity based on ECM type
            if 'wall' in ecm_id:
                quantity = areas.get('wall', 0)
            elif 'window' in ecm_id:
                quantity = areas.get('window', 0)
            elif 'roof' in ecm_id:
                quantity = areas.get('roof', 0)
            elif 'pv' in ecm_id or 'solar' in ecm_id:
                # PV sized by roof area, estimate kWp
                pv_area = areas.get('pv', areas.get('roof', 0) * 0.5)
                quantity = pv_area * 0.2  # ~200 W/m² = 0.2 kWp/m²
            elif 'heat_pump' in ecm_id:
                # Size by heating load (rough estimate)
                quantity = areas.get('floor', 0) * 0.03  # ~30 W/m²
            else:
                # Default to floor area
                quantity = areas.get('floor', 0)

            cost = self.costs.ecm_cost(ecm_id, quantity)
            breakdown[ecm_id] = cost
            total += cost

        return total, breakdown

    def _calculate_npv(
        self,
        investment: float,
        annual_savings: float,
        years: int
    ) -> float:
        """Calculate Net Present Value."""
        npv = -investment
        for year in range(1, years + 1):
            npv += annual_savings / ((1 + self.discount_rate) ** year)
        return npv

    def _calculate_irr(
        self,
        investment: float,
        annual_savings: float,
        years: int
    ) -> Optional[float]:
        """Calculate Internal Rate of Return using iterative method."""
        if annual_savings <= 0 or investment <= 0:
            return None

        # Newton-Raphson iteration
        irr = 0.10  # Initial guess
        for _ in range(50):
            npv = -investment
            dnpv = 0
            for year in range(1, years + 1):
                factor = (1 + irr) ** year
                npv += annual_savings / factor
                dnpv -= year * annual_savings / (factor * (1 + irr))

            if abs(npv) < 0.01:
                return irr * 100

            if dnpv == 0:
                break

            irr = irr - npv / dnpv

            if irr < -0.99 or irr > 10:
                return None

        return irr * 100 if -0.99 < irr < 10 else None

    def _calculate_score(
        self,
        payback: float,
        npv: float,
        investment: float,
        annual_co2: float
    ) -> float:
        """
        Calculate composite score for ranking.

        Higher score = better investment.
        """
        # Normalize components
        payback_score = max(0, 20 - payback) / 20  # 0-20 year payback -> 0-1
        npv_score = max(0, min(1, npv / (investment + 1)))  # NPV relative to investment
        co2_score = min(1, annual_co2 / 10000)  # 10 tonnes/year -> score of 1

        # Weighted combination
        score = (
            0.4 * payback_score +
            0.4 * npv_score +
            0.2 * co2_score
        ) * 100

        return max(0, score)

    def rank_results(self, results: List[ROIResult]) -> List[ROIResult]:
        """Sort results by score (best first)."""
        return sorted(results, key=lambda r: r.score, reverse=True)

    def generate_summary(self, results: List[ROIResult]) -> str:
        """Generate human-readable summary of ROI results."""
        lines = []
        lines.append("ECM Investment Analysis Summary")
        lines.append("=" * 60)
        lines.append("")

        ranked = self.rank_results(results)

        lines.append(f"{'ECM':<30} {'Invest (kSEK)':<15} {'Payback':<10} {'NPV (kSEK)':<12} {'Score':<8}")
        lines.append("-" * 75)

        for r in ranked[:10]:  # Top 10
            invest_k = r.total_investment_sek / 1000
            npv_k = r.npv_sek / 1000
            payback_str = f"{r.simple_payback_years:.1f} yr" if r.simple_payback_years < 100 else "N/A"
            lines.append(
                f"{r.combination_name[:28]:<30} "
                f"{invest_k:>12,.0f}   "
                f"{payback_str:<10} "
                f"{npv_k:>10,.0f}   "
                f"{r.score:>6.1f}"
            )

        lines.append("")
        lines.append(f"Analysis period: {self.analysis_period} years")
        lines.append(f"Discount rate: {self.discount_rate*100:.1f}%")

        return "\n".join(lines)
