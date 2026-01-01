"""
Portfolio analytics and reporting for building portfolios.

Aggregates individual building results into portfolio-level insights.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PortfolioAnalytics:
    """Aggregate analytics for a building portfolio."""

    # Counts
    total_buildings: int = 0
    analyzed: int = 0
    skipped_already_optimized: int = 0
    failed: int = 0
    flagged_for_qc: int = 0
    qc_completed: int = 0

    # Energy metrics
    total_current_consumption_kwh: float = 0.0
    total_savings_potential_kwh: float = 0.0
    average_current_kwh_m2: float = 0.0
    average_savings_kwh_m2: float = 0.0

    # Financial metrics
    total_investment_sek: float = 0.0
    total_annual_savings_sek: float = 0.0
    portfolio_npv_sek: float = 0.0
    portfolio_payback_years: float = 0.0
    portfolio_irr: float = 0.0

    # Uncertainty
    savings_uncertainty_kwh: float = 0.0
    savings_ci_90_low: float = 0.0
    savings_ci_90_high: float = 0.0

    # Top buildings
    top_10_roi: List[Dict[str, Any]] = field(default_factory=list)
    top_10_savings: List[Dict[str, Any]] = field(default_factory=list)
    worst_10_consumption: List[Dict[str, Any]] = field(default_factory=list)

    # ECM frequency
    ecm_frequency: Dict[str, int] = field(default_factory=dict)
    ecm_total_savings: Dict[str, float] = field(default_factory=dict)

    # Archetype distribution
    archetype_distribution: Dict[str, int] = field(default_factory=dict)

    # Energy class distribution
    energy_class_distribution: Dict[str, int] = field(default_factory=dict)

    # Processing stats
    total_processing_time_sec: float = 0.0
    tier_distribution: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_results(cls, results: List[Any]) -> "PortfolioAnalytics":
        """
        Create analytics from a list of BuildingResults.

        Args:
            results: List of BuildingResult objects

        Returns:
            PortfolioAnalytics instance
        """
        analytics = cls()

        analytics.total_buildings = len(results)

        # Energy price for calculations
        energy_price_sek_kwh = 1.20

        # Process each result
        successful_results = []
        for r in results:
            # Counts
            if r.tier.value == "skip":
                analytics.skipped_already_optimized += 1
            elif not r.success:
                analytics.failed += 1
            else:
                analytics.analyzed += 1
                successful_results.append(r)

            if r.needs_qc:
                analytics.flagged_for_qc += 1
            if r.qc_completed:
                analytics.qc_completed += 1

            # Tier distribution
            tier_name = r.tier.value
            analytics.tier_distribution[tier_name] = analytics.tier_distribution.get(tier_name, 0) + 1

            # Energy class distribution
            if r.energy_class:
                analytics.energy_class_distribution[r.energy_class] = \
                    analytics.energy_class_distribution.get(r.energy_class, 0) + 1

            # Archetype distribution
            if r.archetype_id:
                analytics.archetype_distribution[r.archetype_id] = \
                    analytics.archetype_distribution.get(r.archetype_id, 0) + 1

            # Energy metrics
            if r.current_kwh_m2 and r.atemp_m2:
                analytics.total_current_consumption_kwh += r.current_kwh_m2 * r.atemp_m2

            if r.total_savings_kwh_m2 and r.atemp_m2:
                savings_kwh = r.total_savings_kwh_m2 * r.atemp_m2
                analytics.total_savings_potential_kwh += savings_kwh

                # Uncertainty
                if r.savings_uncertainty_kwh_m2:
                    analytics.savings_uncertainty_kwh += r.savings_uncertainty_kwh_m2 * r.atemp_m2

            # Financial metrics
            if r.total_investment_sek:
                analytics.total_investment_sek += r.total_investment_sek

            # ECM frequency
            for ecm in r.recommended_ecms:
                ecm_id = ecm.get("ecm_id", "unknown")
                analytics.ecm_frequency[ecm_id] = analytics.ecm_frequency.get(ecm_id, 0) + 1

                if "savings_kwh_m2" in ecm and r.atemp_m2:
                    ecm_savings = ecm["savings_kwh_m2"] * r.atemp_m2
                    analytics.ecm_total_savings[ecm_id] = \
                        analytics.ecm_total_savings.get(ecm_id, 0) + ecm_savings

            # Processing time
            analytics.total_processing_time_sec += r.processing_time_sec

        # Calculate averages
        if analytics.analyzed > 0:
            total_atemp = sum(r.atemp_m2 for r in successful_results if r.atemp_m2) or 1

            analytics.average_current_kwh_m2 = \
                analytics.total_current_consumption_kwh / total_atemp

            analytics.average_savings_kwh_m2 = \
                analytics.total_savings_potential_kwh / total_atemp

        # Calculate financial totals
        analytics.total_annual_savings_sek = \
            analytics.total_savings_potential_kwh * energy_price_sek_kwh

        if analytics.total_annual_savings_sek > 0:
            analytics.portfolio_payback_years = \
                analytics.total_investment_sek / analytics.total_annual_savings_sek

        # NPV calculation (30 year horizon, 4% discount)
        discount_rate = 0.04
        horizon_years = 30
        if analytics.total_annual_savings_sek > 0:
            npv_factor = sum(1 / (1 + discount_rate) ** t for t in range(1, horizon_years + 1))
            analytics.portfolio_npv_sek = \
                analytics.total_annual_savings_sek * npv_factor - analytics.total_investment_sek

        # Top 10 lists
        analytics.top_10_roi = cls._get_top_roi(successful_results)
        analytics.top_10_savings = cls._get_top_savings(successful_results)
        analytics.worst_10_consumption = cls._get_worst_consumption(successful_results)

        # Uncertainty CI
        if analytics.savings_uncertainty_kwh > 0:
            # Assume normal distribution for portfolio
            z_90 = 1.645
            analytics.savings_ci_90_low = max(0, analytics.total_savings_potential_kwh - z_90 * analytics.savings_uncertainty_kwh)
            analytics.savings_ci_90_high = analytics.total_savings_potential_kwh + z_90 * analytics.savings_uncertainty_kwh

        return analytics

    @staticmethod
    def _get_top_roi(results: List[Any], n: int = 10) -> List[Dict[str, Any]]:
        """Get top N buildings by ROI."""
        with_roi = [
            r for r in results
            if r.simple_payback_years and r.simple_payback_years > 0
        ]
        sorted_results = sorted(with_roi, key=lambda r: r.simple_payback_years)

        return [
            {
                "address": r.address,
                "payback_years": r.simple_payback_years,
                "savings_kwh_m2": r.total_savings_kwh_m2,
                "investment_sek": r.total_investment_sek,
            }
            for r in sorted_results[:n]
        ]

    @staticmethod
    def _get_top_savings(results: List[Any], n: int = 10) -> List[Dict[str, Any]]:
        """Get top N buildings by absolute savings."""
        with_savings = [
            r for r in results
            if r.total_savings_kwh_m2 and r.atemp_m2
        ]
        sorted_results = sorted(
            with_savings,
            key=lambda r: r.total_savings_kwh_m2 * r.atemp_m2,
            reverse=True,
        )

        return [
            {
                "address": r.address,
                "savings_kwh": r.total_savings_kwh_m2 * r.atemp_m2,
                "savings_kwh_m2": r.total_savings_kwh_m2,
                "atemp_m2": r.atemp_m2,
            }
            for r in sorted_results[:n]
        ]

    @staticmethod
    def _get_worst_consumption(results: List[Any], n: int = 10) -> List[Dict[str, Any]]:
        """Get top N buildings by current consumption."""
        with_consumption = [
            r for r in results
            if r.current_kwh_m2 and r.current_kwh_m2 > 0
        ]
        sorted_results = sorted(
            with_consumption,
            key=lambda r: r.current_kwh_m2,
            reverse=True,
        )

        return [
            {
                "address": r.address,
                "current_kwh_m2": r.current_kwh_m2,
                "energy_class": r.energy_class,
                "construction_year": r.construction_year,
            }
            for r in sorted_results[:n]
        ]


def generate_portfolio_report(
    analytics: PortfolioAnalytics,
    output_path: Optional[Path] = None,
    format: str = "html",
) -> str:
    """
    Generate a portfolio report from analytics.

    Args:
        analytics: PortfolioAnalytics instance
        output_path: Optional path to save report
        format: Report format ("html", "markdown", "json")

    Returns:
        Report content as string
    """
    if format == "html":
        content = _generate_html_report(analytics)
    elif format == "markdown":
        content = _generate_markdown_report(analytics)
    elif format == "json":
        import json
        content = json.dumps(_analytics_to_dict(analytics), indent=2)
    else:
        raise ValueError(f"Unknown format: {format}")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(content)
        logger.info(f"Report saved to {output_path}")

    return content


def _analytics_to_dict(analytics: PortfolioAnalytics) -> Dict[str, Any]:
    """Convert analytics to dict for serialization."""
    return {
        "summary": {
            "total_buildings": analytics.total_buildings,
            "analyzed": analytics.analyzed,
            "skipped": analytics.skipped_already_optimized,
            "failed": analytics.failed,
            "flagged_for_qc": analytics.flagged_for_qc,
        },
        "energy": {
            "total_current_consumption_kwh": analytics.total_current_consumption_kwh,
            "total_savings_potential_kwh": analytics.total_savings_potential_kwh,
            "average_current_kwh_m2": analytics.average_current_kwh_m2,
            "average_savings_kwh_m2": analytics.average_savings_kwh_m2,
            "savings_ci_90": [analytics.savings_ci_90_low, analytics.savings_ci_90_high],
        },
        "financial": {
            "total_investment_sek": analytics.total_investment_sek,
            "total_annual_savings_sek": analytics.total_annual_savings_sek,
            "portfolio_npv_sek": analytics.portfolio_npv_sek,
            "portfolio_payback_years": analytics.portfolio_payback_years,
        },
        "top_buildings": {
            "top_10_roi": analytics.top_10_roi,
            "top_10_savings": analytics.top_10_savings,
            "worst_10_consumption": analytics.worst_10_consumption,
        },
        "distributions": {
            "energy_class": analytics.energy_class_distribution,
            "archetype": analytics.archetype_distribution,
            "tier": analytics.tier_distribution,
        },
        "ecm_analysis": {
            "frequency": analytics.ecm_frequency,
            "total_savings": analytics.ecm_total_savings,
        },
    }


def _generate_markdown_report(analytics: PortfolioAnalytics) -> str:
    """Generate markdown report."""
    lines = [
        "# Portfolio Energy Analysis Report",
        "",
        "## Summary",
        "",
        f"- **Total Buildings**: {analytics.total_buildings}",
        f"- **Analyzed**: {analytics.analyzed}",
        f"- **Skipped (Already Optimized)**: {analytics.skipped_already_optimized}",
        f"- **Failed**: {analytics.failed}",
        f"- **Flagged for QC**: {analytics.flagged_for_qc}",
        "",
        "## Energy Metrics",
        "",
        f"- **Current Consumption**: {analytics.total_current_consumption_kwh:,.0f} kWh",
        f"- **Savings Potential**: {analytics.total_savings_potential_kwh:,.0f} kWh",
        f"- **Average Current**: {analytics.average_current_kwh_m2:.1f} kWh/m²",
        f"- **Average Savings**: {analytics.average_savings_kwh_m2:.1f} kWh/m²",
        "",
        "## Financial Metrics",
        "",
        f"- **Total Investment**: {analytics.total_investment_sek:,.0f} SEK",
        f"- **Annual Savings**: {analytics.total_annual_savings_sek:,.0f} SEK",
        f"- **Portfolio NPV**: {analytics.portfolio_npv_sek:,.0f} SEK",
        f"- **Portfolio Payback**: {analytics.portfolio_payback_years:.1f} years",
        "",
        "## Top 10 Buildings by ROI",
        "",
        "| Address | Payback (years) | Savings (kWh/m²) |",
        "|---------|-----------------|------------------|",
    ]

    for b in analytics.top_10_roi:
        lines.append(f"| {b['address']} | {b['payback_years']:.1f} | {b['savings_kwh_m2']:.1f} |")

    lines.extend([
        "",
        "## Most Common ECMs",
        "",
        "| ECM | Buildings | Total Savings (kWh) |",
        "|-----|-----------|---------------------|",
    ])

    sorted_ecms = sorted(
        analytics.ecm_frequency.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    for ecm_id, count in sorted_ecms[:10]:
        savings = analytics.ecm_total_savings.get(ecm_id, 0)
        lines.append(f"| {ecm_id} | {count} | {savings:,.0f} |")

    lines.extend([
        "",
        "## Energy Class Distribution",
        "",
    ])

    for cls, count in sorted(analytics.energy_class_distribution.items()):
        lines.append(f"- **Class {cls}**: {count} buildings")

    return "\n".join(lines)


def _generate_html_report(analytics: PortfolioAnalytics) -> str:
    """Generate HTML report."""
    md_content = _generate_markdown_report(analytics)

    # Simple markdown to HTML conversion
    html_content = md_content
    html_content = html_content.replace("# ", "<h1>").replace("\n## ", "</h1>\n<h2>")
    html_content = html_content.replace("\n\n", "</p>\n<p>")

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>Portfolio Energy Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .metric {{ background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .metric strong {{ color: #2c3e50; }}
    </style>
</head>
<body>
<pre>{md_content}</pre>
</body>
</html>"""
