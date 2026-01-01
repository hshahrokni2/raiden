"""
HTML Report Generator for BRF Building Analysis.

Generates professional reports showing:
- Building summary and characteristics
- Existing energy measures
- Applicable ECMs with savings potential
- Simulation results comparison
- Financial analysis and recommendations
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

from src.analysis.package_generator import (
    PackageGenerator,
    ECMPackage,
    ECM_COSTS_PER_M2_FALLBACK,
    get_ecm_cost_v2,
    get_energy_price,
)


@dataclass
class ECMResult:
    """Result for a single ECM."""
    id: str
    name: str
    category: str
    baseline_kwh_m2: float
    result_kwh_m2: float
    savings_kwh_m2: float
    savings_percent: float
    estimated_cost_sek: float
    simple_payback_years: float
    # Multi-end-use energy tracking
    total_kwh_m2: float = 0  # Total energy (heating + DHW + property_el)
    total_savings_percent: float = 0  # Savings including all end-uses
    heating_kwh_m2: float = 0  # Heating component
    dhw_kwh_m2: float = 0  # Domestic hot water component
    property_el_kwh_m2: float = 0  # Property electricity (lighting, fans, pumps)
    savings_by_end_use: Optional[Dict[str, float]] = None  # Per-end-use breakdown


@dataclass
class MaintenancePlanData:
    """Maintenance plan data for report."""
    total_investment_sek: float = 0
    total_savings_30yr_sek: float = 0
    net_present_value_sek: float = 0
    break_even_year: int = 0
    final_fund_balance_sek: float = 0
    max_loan_used_sek: float = 0
    projections: List[Dict] = field(default_factory=list)  # Year-by-year projections
    zero_cost_annual_savings: float = 0


@dataclass
class EffektvaktData:
    """Effektvakt (peak shaving) analysis data."""
    current_el_peak_kw: float = 0
    current_fv_peak_kw: float = 0
    optimized_el_peak_kw: float = 0
    optimized_fv_peak_kw: float = 0
    el_peak_reduction_kw: float = 0
    fv_peak_reduction_kw: float = 0
    annual_el_savings_sek: float = 0
    annual_fv_savings_sek: float = 0
    total_annual_savings_sek: float = 0
    pre_heat_hours: float = 0
    pre_heat_temp_c: float = 0
    coast_duration_hours: float = 0
    requires_bms: bool = True
    manual_possible: bool = False
    notes: List[str] = field(default_factory=list)


@dataclass
class RenovationHistoryData:
    """Renovation history from Gripen energy declarations."""
    has_history: bool = False
    is_renovated: bool = False
    original_year: int = 0
    original_energy_class: str = ""
    original_kwh_m2: float = 0
    current_year: int = 0
    current_energy_class: str = ""
    current_kwh_m2: float = 0
    energy_class_improvement: int = 0  # Positive = improved (e.g., F→C = 3)
    kwh_reduction_percent: float = 0
    declarations: List[Dict] = field(default_factory=list)  # All historical declarations


@dataclass
class ReportData:
    """Data for generating a report."""
    # Building info
    building_name: str
    address: str
    construction_year: int
    building_type: str
    facade_material: str
    atemp_m2: float
    floors: int
    energy_class: str
    declared_heating_kwh_m2: float

    # Analysis results
    baseline_heating_kwh_m2: float
    existing_measures: List[str]
    applicable_ecms: List[str]
    excluded_ecms: List[Dict[str, str]]
    ecm_results: List[ECMResult]

    # Multi-end-use energy breakdown (Swedish energideklaration)
    baseline_dhw_kwh_m2: float = 0  # Tappvarmvatten (domestic hot water)
    baseline_property_el_kwh_m2: float = 0  # Fastighetsel (property electricity)
    baseline_cooling_kwh_m2: float = 0  # Komfortkyla (space cooling)
    baseline_total_kwh_m2: float = 0  # Total energy (all end-uses)

    # Solar potential
    existing_pv_m2: float = 0
    remaining_pv_m2: float = 0
    additional_pv_kwp: float = 0

    # ECM Packages
    packages: List[ECMPackage] = None

    # Maintenance Plan (Long-term financial projection)
    maintenance_plan: MaintenancePlanData = None

    # Effektvakt (Peak demand optimization)
    effektvakt: EffektvaktData = None

    # Renovation History (from Gripen energy declarations over time)
    renovation_history: RenovationHistoryData = None

    # BRF Financials (for context)
    num_apartments: int = 0
    current_fund_sek: float = 0
    annual_energy_cost_sek: float = 0

    # Metadata
    analysis_date: str = ""
    analysis_duration_seconds: float = 0

    # Calibration quality (Bayesian calibration results)
    calibration_method: str = ""  # "bayesian_abc_smc" or "simple"
    calibrated_kwh_m2: float = 0
    calibration_std: float = 0  # Uncertainty (std dev)
    ashrae_nmbe: float = 0  # Normalized Mean Bias Error (%)
    ashrae_cvrmse: float = 0  # Coefficient of Variation of RMSE (%)
    ashrae_passes: bool = False  # Passes ASHRAE Guideline 14
    surrogate_r2: float = 0  # Training R²
    surrogate_test_r2: float = 0  # Test R² (generalization)
    surrogate_is_overfit: bool = False  # Overfitting warning
    morris_ranking: Optional[Dict[str, int]] = None  # Parameter importance ranking
    calibrated_params: Optional[List[str]] = None  # Which params were calibrated
    # Data quality assessment
    calibration_data_resolution: str = "annual"  # "hourly", "monthly", "annual"
    calibration_confidence: float = 0.5  # 0-1 confidence based on data quality
    calibration_data_warning: str = ""  # Warning about data quality limitations

    # Building complexity (for model accuracy assessment)
    building_complexity_score: float = 0  # 0-100
    building_complexity_warning: str = ""  # Warning if complex geometry
    single_zone_adequate: bool = True  # Whether single-zone model is adequate


HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="sv">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Energianalys - {building_name}</title>
    <style>
        :root {{
            --primary: #2563eb;
            --success: #16a34a;
            --warning: #d97706;
            --danger: #dc2626;
            --gray-100: #f3f4f6;
            --gray-200: #e5e7eb;
            --gray-700: #374151;
            --gray-900: #111827;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: var(--gray-700);
            background: var(--gray-100);
            padding: 2rem;
        }}

        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            overflow: hidden;
        }}

        header {{
            background: linear-gradient(135deg, var(--primary), #1e40af);
            color: white;
            padding: 2rem;
        }}

        header h1 {{
            font-size: 1.75rem;
            margin-bottom: 0.5rem;
        }}

        header p {{
            opacity: 0.9;
            font-size: 1rem;
        }}

        .content {{
            padding: 2rem;
        }}

        section {{
            margin-bottom: 2rem;
        }}

        h2 {{
            color: var(--gray-900);
            font-size: 1.25rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--gray-200);
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }}

        .stat-card {{
            background: var(--gray-100);
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }}

        .stat-card .value {{
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--primary);
        }}

        .stat-card .label {{
            font-size: 0.875rem;
            color: var(--gray-700);
        }}

        .measure-list {{
            list-style: none;
        }}

        .measure-list li {{
            padding: 0.5rem 0;
            border-bottom: 1px solid var(--gray-200);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .measure-list li:last-child {{
            border-bottom: none;
        }}

        .badge {{
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }}

        .badge-success {{
            background: #dcfce7;
            color: var(--success);
        }}

        .badge-warning {{
            background: #fef3c7;
            color: var(--warning);
        }}

        .badge-danger {{
            background: #fee2e2;
            color: var(--danger);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}

        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--gray-200);
        }}

        th {{
            background: var(--gray-100);
            font-weight: 600;
            color: var(--gray-900);
        }}

        tr:hover {{
            background: var(--gray-100);
        }}

        .savings-positive {{
            color: var(--success);
            font-weight: 600;
        }}

        .savings-neutral {{
            color: var(--gray-700);
        }}

        .chart-bar {{
            height: 20px;
            background: var(--gray-200);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 0.25rem;
        }}

        .chart-fill {{
            height: 100%;
            background: linear-gradient(90deg, var(--success), #22c55e);
            border-radius: 4px;
        }}

        .recommendation {{
            background: linear-gradient(135deg, #dcfce7, #bbf7d0);
            padding: 1.5rem;
            border-radius: 8px;
            margin-top: 1rem;
        }}

        .recommendation h3 {{
            color: var(--success);
            margin-bottom: 0.5rem;
        }}

        footer {{
            background: var(--gray-100);
            padding: 1rem 2rem;
            text-align: center;
            font-size: 0.875rem;
            color: var(--gray-700);
        }}

        /* Package cards */
        .package-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
            margin-top: 1rem;
        }}

        .package-card {{
            border: 2px solid var(--gray-200);
            border-radius: 12px;
            overflow: hidden;
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        .package-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        }}

        .package-card.basic {{
            border-color: #93c5fd;
        }}

        .package-card.standard {{
            border-color: #86efac;
        }}

        .package-card.premium {{
            border-color: #fcd34d;
        }}

        .package-header {{
            padding: 1rem;
            text-align: center;
        }}

        .package-card.basic .package-header {{
            background: linear-gradient(135deg, #dbeafe, #bfdbfe);
        }}

        .package-card.standard .package-header {{
            background: linear-gradient(135deg, #dcfce7, #bbf7d0);
        }}

        .package-card.premium .package-header {{
            background: linear-gradient(135deg, #fef3c7, #fde68a);
        }}

        .package-header h3 {{
            font-size: 1.25rem;
            margin-bottom: 0.25rem;
            color: var(--gray-900);
        }}

        .package-header .package-savings {{
            font-size: 2rem;
            font-weight: 700;
            color: var(--success);
        }}

        .package-body {{
            padding: 1rem;
        }}

        .package-body ul {{
            list-style: none;
            margin-bottom: 1rem;
        }}

        .package-body li {{
            padding: 0.375rem 0;
            font-size: 0.875rem;
            border-bottom: 1px solid var(--gray-200);
        }}

        .package-body li:last-child {{
            border-bottom: none;
        }}

        .package-footer {{
            background: var(--gray-100);
            padding: 1rem;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
            font-size: 0.875rem;
        }}

        .package-footer .label {{
            color: var(--gray-700);
        }}

        .package-footer .value {{
            font-weight: 600;
            text-align: right;
        }}

        /* Maintenance Plan styles */
        .cash-flow-table {{
            width: 100%;
            font-size: 0.85rem;
        }}

        .cash-flow-table th {{
            background: var(--primary);
            color: white;
            padding: 0.5rem;
            text-align: right;
        }}

        .cash-flow-table th:first-child {{
            text-align: left;
        }}

        .cash-flow-table td {{
            text-align: right;
            padding: 0.4rem 0.5rem;
        }}

        .cash-flow-table td:first-child {{
            text-align: left;
            font-weight: 600;
        }}

        .cash-flow-table tr.investment-row {{
            background: #fef3c7;
        }}

        .cash-flow-table tr.warning-row {{
            background: #fee2e2;
        }}

        .plan-summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }}

        .plan-metric {{
            background: linear-gradient(135deg, var(--gray-100), #e5e7eb);
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }}

        .plan-metric.positive {{
            background: linear-gradient(135deg, #dcfce7, #bbf7d0);
        }}

        .plan-metric .metric-value {{
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--primary);
        }}

        .plan-metric.positive .metric-value {{
            color: var(--success);
        }}

        .plan-metric .metric-label {{
            font-size: 0.75rem;
            color: var(--gray-700);
        }}

        /* Effektvakt styles */
        .effektvakt-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            margin: 1rem 0;
        }}

        .effektvakt-card {{
            background: var(--gray-100);
            border-radius: 8px;
            padding: 1rem;
        }}

        .effektvakt-card h4 {{
            font-size: 0.875rem;
            color: var(--gray-700);
            margin-bottom: 0.5rem;
        }}

        .peak-comparison {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .peak-value {{
            font-size: 1.5rem;
            font-weight: 700;
        }}

        .peak-value.before {{
            color: var(--danger);
        }}

        .peak-value.after {{
            color: var(--success);
        }}

        .peak-arrow {{
            font-size: 1.25rem;
            color: var(--gray-700);
        }}

        .peak-reduction {{
            background: #dcfce7;
            color: var(--success);
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-weight: 600;
        }}

        .strategy-list {{
            list-style: none;
            margin-top: 1rem;
        }}

        .strategy-list li {{
            padding: 0.5rem 0;
            border-bottom: 1px solid var(--gray-200);
            display: flex;
            justify-content: space-between;
        }}

        .strategy-list li:last-child {{
            border-bottom: none;
        }}

        .cascade-visual {{
            background: linear-gradient(90deg, #3b82f6, #10b981, #22c55e);
            height: 8px;
            border-radius: 4px;
            margin: 1rem 0;
        }}

        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            .container {{
                box-shadow: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{building_name}</h1>
            <p>{address}</p>
        </header>

        <div class="content">
            <section>
                <h2>Byggnadsöversikt</h2>
                <div class="grid">
                    <div class="stat-card">
                        <div class="value">{construction_year}</div>
                        <div class="label">Byggnadsår</div>
                    </div>
                    <div class="stat-card">
                        <div class="value">{atemp_m2:,.0f} m²</div>
                        <div class="label">Atemp</div>
                    </div>
                    <div class="stat-card">
                        <div class="value">{floors}</div>
                        <div class="label">Våningar</div>
                    </div>
                    <div class="stat-card">
                        <div class="value">{energy_class}</div>
                        <div class="label">Energiklass</div>
                    </div>
                    <div class="stat-card">
                        <div class="value">{declared_heating_kwh_m2:.0f} kWh/m²</div>
                        <div class="label">Deklarerad energi</div>
                    </div>
                    <div class="stat-card">
                        <div class="value">{facade_material}</div>
                        <div class="label">Fasadmaterial</div>
                    </div>
                </div>
            </section>

            <section>
                <h2>Befintliga åtgärder ({num_existing})</h2>
                <ul class="measure-list">
                    {existing_measures_html}
                </ul>
            </section>

            <section>
                <h2>ECM-analys</h2>
                <div class="grid" style="grid-template-columns: repeat(3, 1fr); margin-bottom: 1rem;">
                    <div class="stat-card">
                        <div class="value" style="color: var(--success);">{num_applicable}</div>
                        <div class="label">Tillämpliga åtgärder</div>
                    </div>
                    <div class="stat-card">
                        <div class="value" style="color: var(--warning);">{num_already_done}</div>
                        <div class="label">Redan genomförda</div>
                    </div>
                    <div class="stat-card">
                        <div class="value" style="color: var(--danger);">{num_not_applicable}</div>
                        <div class="label">Ej tillämpliga</div>
                    </div>
                </div>

                {ecm_results_table}
            </section>

            {calibration_section}

            {complexity_section}

            {solar_section}

            {packages_section}

            {effektvakt_section}

            {renovation_history_section}

            {maintenance_plan_section}

            <section>
                <h2>Sammanfattning</h2>
                <div class="recommendation">
                    <h3>Rekommendation</h3>
                    <p>{recommendation_text}</p>
                </div>
            </section>
        </div>

        <footer>
            <p>Genererad av Raiden - Swedish Building Energy Analysis | {analysis_date}</p>
        </footer>
    </div>
</body>
</html>'''


class HTMLReportGenerator:
    """Generate HTML reports for building energy analysis."""

    def __init__(self):
        pass

    def generate(self, data: ReportData, output_path: Optional[Path] = None) -> str:
        """
        Generate HTML report from analysis data.

        Args:
            data: Report data
            output_path: Optional path to save HTML file

        Returns:
            HTML string
        """
        # Format existing measures
        existing_html = ""
        for measure in data.existing_measures:
            existing_html += f'<li><span class="badge badge-success">✓</span> {self._format_measure_name(measure)}</li>\n'

        if not data.existing_measures:
            existing_html = '<li><em>Inga befintliga åtgärder identifierade</em></li>'

        # Format ECM results table
        ecm_table = self._format_ecm_table(data)

        # Format solar section
        solar_section = ""
        if data.remaining_pv_m2 > 0:
            solar_section = f'''
            <section>
                <h2>Solpotential</h2>
                <div class="grid">
                    <div class="stat-card">
                        <div class="value">{data.existing_pv_m2:.0f} m²</div>
                        <div class="label">Befintlig solcellsyta</div>
                    </div>
                    <div class="stat-card">
                        <div class="value">{data.remaining_pv_m2:.0f} m²</div>
                        <div class="label">Tillgänglig takyta</div>
                    </div>
                    <div class="stat-card">
                        <div class="value">{data.additional_pv_kwp:.0f} kWp</div>
                        <div class="label">Potentiell kapacitet</div>
                    </div>
                </div>
            </section>
            '''

        # Generate recommendation
        recommendation = self._generate_recommendation(data)

        # Count ECM categories
        num_already_done = len([e for e in data.excluded_ecms if 'already' in e.get('reason', '').lower()])
        num_not_applicable = len(data.excluded_ecms) - num_already_done

        # Generate packages section
        packages_section = self._format_packages_section(data)

        # Generate effektvakt section
        effektvakt_section = self._format_effektvakt_section(data)

        # Generate renovation history section (from Gripen energy declarations)
        renovation_history_section = self._format_renovation_history_section(data)

        # Generate maintenance plan section
        maintenance_plan_section = self._format_maintenance_plan_section(data)

        # Generate calibration quality section
        calibration_section = self._format_calibration_section(data)

        # Generate complexity section
        complexity_section = self._format_complexity_section(data)

        # Format HTML
        html = HTML_TEMPLATE.format(
            building_name=data.building_name,
            address=data.address,
            construction_year=data.construction_year,
            atemp_m2=data.atemp_m2,
            floors=data.floors,
            energy_class=data.energy_class,
            declared_heating_kwh_m2=data.declared_heating_kwh_m2,
            facade_material=data.facade_material.title(),
            num_existing=len(data.existing_measures),
            existing_measures_html=existing_html,
            num_applicable=len(data.applicable_ecms),
            num_already_done=num_already_done,
            num_not_applicable=num_not_applicable,
            ecm_results_table=ecm_table,
            calibration_section=calibration_section,
            complexity_section=complexity_section,
            solar_section=solar_section,
            packages_section=packages_section,
            effektvakt_section=effektvakt_section,
            renovation_history_section=renovation_history_section,
            maintenance_plan_section=maintenance_plan_section,
            recommendation_text=recommendation,
            analysis_date=data.analysis_date or datetime.now().strftime("%Y-%m-%d %H:%M"),
        )

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)

        return html

    def _format_measure_name(self, measure: str) -> str:
        """Format measure ID to readable name."""
        names = {
            'heat_pump_ground': 'Bergvärmepump',
            'heat_pump_exhaust': 'Frånluftsvärmepump',
            'solar_pv': 'Solceller',
            'ftx_system': 'FTX-ventilation',
            'heat_recovery': 'Värmeåtervinning',
            'window_replacement': 'Moderna fönster',
            'wall_insulation': 'Tilläggsisolering väggar',
            'roof_insulation': 'Tilläggsisolering tak',
        }
        return names.get(measure, measure.replace('_', ' ').title())

    def _format_packages_section(self, data: ReportData) -> str:
        """Format ECM packages as HTML cards with Steg 0 (zero-cost) first."""
        if not data.packages:
            return ''

        # Separate zero-cost (Steg 0) from capital packages
        zero_cost_packages = []
        capital_packages = []

        for pkg in data.packages:
            if 'zero' in pkg.id.lower() or 'steg0' in pkg.id.lower():
                zero_cost_packages.append(pkg)
            else:
                capital_packages.append(pkg)

        result_html = ''

        # === STEG 0: Zero-Cost Section (DO THIS FIRST) ===
        if zero_cost_packages:
            zero_cost_cards = ''
            for pkg in zero_cost_packages:
                ecm_list = ''
                for ecm in pkg.ecms:
                    # Zero-cost ECMs may have 0% thermal savings but estimated cost savings
                    savings_note = f'{ecm.individual_savings_percent:.0f}%' if ecm.individual_savings_percent > 0 else 'kostnadsbesparing'
                    ecm_list += f'<li>{ecm.name} ({savings_note})</li>\n'

                pkg_name = pkg.name_sv if hasattr(pkg, 'name_sv') else pkg.name
                description = pkg.description_sv if hasattr(pkg, 'description_sv') else pkg.description

                zero_cost_cards += f'''
                <div class="package-card zero-cost" style="border-color: #10b981; background: linear-gradient(135deg, #ecfdf5, #d1fae5);">
                    <div class="package-header" style="background: linear-gradient(135deg, #10b981, #059669); color: white;">
                        <h3>{pkg_name} <span class="badge" style="background: white; color: #059669;">GÖR DETTA FÖRST!</span></h3>
                        <div class="package-savings" style="color: white;">~5-15%</div>
                        <div style="opacity: 0.9;">{description}</div>
                    </div>
                    <div class="package-body">
                        <ul>
                            {ecm_list}
                        </ul>
                    </div>
                    <div class="package-footer" style="background: #ecfdf5;">
                        <span class="label">Investering:</span>
                        <span class="value">{pkg.total_cost_sek:,.0f} SEK</span>
                        <span class="label">Typisk återbetalning:</span>
                        <span class="value">{"< 6 månader" if pkg.total_cost_sek < 20000 else "< 1 år"}</span>
                        <span class="label">Risk:</span>
                        <span class="value">Ingen</span>
                        <span class="label">Störning:</span>
                        <span class="value">Ingen</span>
                    </div>
                </div>
                '''

            result_html += f'''
            <section style="margin-bottom: 2rem;">
                <h2 style="color: #059669;">🎯 Steg 0: Nollkostnadsåtgärder</h2>
                <p style="margin-bottom: 1rem; color: var(--gray-700); background: #ecfdf5; padding: 1rem; border-radius: 8px; border-left: 4px solid #10b981;">
                    <strong>Börja här!</strong> Dessa åtgärder kräver minimal eller ingen investering och ger omedelbar besparing.
                    Typiskt 5-15% reduktion av energikostnader genom optimering av befintliga system.
                </p>
                <div class="package-cards">
                    {zero_cost_cards}
                </div>
            </section>
            '''

        # === CAPITAL PACKAGES (Steg 1-3) ===
        if capital_packages:
            cards_html = ''
            is_simulated = False

            for pkg in capital_packages:
                # Build ECM list
                ecm_list = ''
                for ecm in pkg.ecms:
                    ecm_list += f'<li>{ecm.name} ({ecm.individual_savings_percent:.0f}%)</li>\n'

                # Handle both SimulatedPackage (new) and ECMPackage (old) formats
                if hasattr(pkg, 'simulated_savings_percent'):
                    is_simulated = True
                    savings_pct = pkg.simulated_savings_percent
                    interaction = pkg.interaction_factor
                    sum_savings = pkg.sum_individual_savings_percent
                    annual_savings = pkg.annual_savings_sek
                    description = pkg.description_sv if hasattr(pkg, 'description_sv') else pkg.description
                    pkg_name = pkg.name_sv if hasattr(pkg, 'name_sv') else pkg.name
                    status_badge = '<span class="badge badge-success">SIMULERAD</span>' if pkg.simulation_success else '<span class="badge badge-warning">ESTIMERAD</span>'

                    interaction_row = f'''
                        <span class="label">Summa enskilda:</span>
                        <span class="value">{sum_savings:.0f}%</span>
                        <span class="label">Samverkanseffekt:</span>
                        <span class="value">{interaction:.0%}</span>
                    '''
                else:
                    savings_pct = pkg.combined_savings_percent
                    annual_savings = pkg.annual_cost_savings_sek
                    description = pkg.description
                    pkg_name = pkg.name
                    status_badge = ''
                    interaction_row = ''

                # Determine card style based on package type
                card_class = 'basic' if 'steg1' in pkg.id or 'basic' in pkg.id else \
                            'standard' if 'steg2' in pkg.id or 'standard' in pkg.id else \
                            'premium' if 'steg3' in pkg.id or 'premium' in pkg.id else ''

                cards_html += f'''
                <div class="package-card {card_class}">
                    <div class="package-header">
                        <h3>{pkg_name} {status_badge}</h3>
                        <div class="package-savings">-{savings_pct:.0f}%</div>
                        <div>{description}</div>
                    </div>
                    <div class="package-body">
                        <ul>
                            {ecm_list}
                        </ul>
                    </div>
                    <div class="package-footer">
                        <span class="label">Investering:</span>
                        <span class="value">{pkg.total_cost_sek:,.0f} SEK</span>
                        <span class="label">Årlig besparing:</span>
                        <span class="value">{annual_savings:,.0f} SEK</span>
                        <span class="label">Återbetalningstid:</span>
                        <span class="value">{pkg.simple_payback_years:.1f} år</span>
                        <span class="label">CO₂-reduktion:</span>
                        <span class="value">{pkg.co2_reduction_kg_m2:.1f} kg/m²</span>
                        {interaction_row}
                    </div>
                </div>
                '''

            explanation = "Paket simuleras med EnergyPlus för faktiska samverkanseffekter." if is_simulated else \
                         "Kombinerade besparingar uppskattas med 70% samverkanseffekt."

            result_html += f'''
            <section>
                <h2>💰 Investeringspaket (Steg 1-3)</h2>
                <p style="margin-bottom: 1rem; color: var(--gray-700);">
                    {explanation} Välj paket baserat på budget och ambitionsnivå.
                </p>
                <div class="package-cards">
                    {cards_html}
                </div>
            </section>
            '''

        return result_html

    def _format_effektvakt_section(self, data: ReportData) -> str:
        """Format effektvakt (peak demand optimization) section."""
        if not data.effektvakt:
            return ''

        eff = data.effektvakt

        # Build notes list
        notes_html = ''
        for note in eff.notes:
            notes_html += f'<li>{note}</li>\n'

        return f'''
        <section>
            <h2>⚡ Effektvakt - Toppeffektoptimering</h2>
            <p style="margin-bottom: 1rem; color: var(--gray-700);">
                Använd byggnadens termiska massa för att jämna ut effekttoppar och sänka effektavgiften.
            </p>

            <div class="effektvakt-grid">
                <div class="effektvakt-card">
                    <h4>Eleffekt</h4>
                    <div class="peak-comparison">
                        <span class="peak-value before">{eff.current_el_peak_kw:.0f} kW</span>
                        <span class="peak-arrow">→</span>
                        <span class="peak-value after">{eff.optimized_el_peak_kw:.0f} kW</span>
                        <span class="peak-reduction">-{eff.el_peak_reduction_kw:.0f} kW</span>
                    </div>
                    <p style="font-size: 0.875rem; margin-top: 0.5rem;">
                        Besparing: <strong>{eff.annual_el_savings_sek:,.0f} SEK/år</strong>
                    </p>
                </div>

                <div class="effektvakt-card">
                    <h4>Fjärrvärmeeffekt</h4>
                    <div class="peak-comparison">
                        <span class="peak-value before">{eff.current_fv_peak_kw:.0f} kW</span>
                        <span class="peak-arrow">→</span>
                        <span class="peak-value after">{eff.optimized_fv_peak_kw:.0f} kW</span>
                        <span class="peak-reduction">-{eff.fv_peak_reduction_kw:.0f} kW</span>
                    </div>
                    <p style="font-size: 0.875rem; margin-top: 0.5rem;">
                        Besparing: <strong>{eff.annual_fv_savings_sek:,.0f} SEK/år</strong>
                    </p>
                </div>
            </div>

            <div class="stat-card" style="background: linear-gradient(135deg, #dcfce7, #bbf7d0); margin: 1rem 0;">
                <div class="value" style="color: var(--success);">{eff.total_annual_savings_sek:,.0f} SEK/år</div>
                <div class="label">Total effektvaktsbesparing</div>
            </div>

            <h3 style="font-size: 1rem; margin-top: 1.5rem;">Strategi</h3>
            <ul class="strategy-list">
                <li>
                    <span>Förvärmningstid</span>
                    <span><strong>{eff.pre_heat_hours:.1f} timmar</strong> före höglast</span>
                </li>
                <li>
                    <span>Temperaturhöjning</span>
                    <span><strong>+{eff.pre_heat_temp_c:.1f}°C</strong> vid förvärmning</span>
                </li>
                <li>
                    <span>Seglingstid</span>
                    <span><strong>{eff.coast_duration_hours:.1f} timmar</strong> utan aktiv värme</span>
                </li>
                <li>
                    <span>Kräver BMS</span>
                    <span>{'Ja' if eff.requires_bms else 'Nej (manuellt möjligt)'}</span>
                </li>
            </ul>

            <div style="margin-top: 1rem; padding: 1rem; background: var(--gray-100); border-radius: 8px;">
                <h4 style="font-size: 0.875rem; margin-bottom: 0.5rem;">Noteringar</h4>
                <ul style="font-size: 0.875rem; padding-left: 1.5rem; color: var(--gray-700);">
                    {notes_html}
                </ul>
            </div>
        </section>
        '''

    def _format_renovation_history_section(self, data: ReportData) -> str:
        """Format renovation history section (from Gripen energy declarations)."""
        if not data.renovation_history or not data.renovation_history.has_history:
            return ''

        rh = data.renovation_history

        # Build declarations timeline
        timeline_html = ''
        for decl in sorted(rh.declarations, key=lambda x: x.get('year', 0)):
            year = decl.get('year', '')
            energy_class = decl.get('energy_class', '')
            kwh = decl.get('kwh_m2', 0)
            timeline_html += f'''
            <div class="timeline-item">
                <div class="timeline-year">{year}</div>
                <div class="timeline-content">
                    <span class="energy-class {energy_class.lower()}">{energy_class}</span>
                    <span class="kwh-value">{kwh:.0f} kWh/m²</span>
                </div>
            </div>
            '''

        # Determine status badge
        if rh.is_renovated:
            status_badge = '<span class="renovation-badge success">✓ Renoverad</span>'
            status_text = f"Byggnaden har genomgått energirenovering med {rh.energy_class_improvement} klassförbättring och {rh.kwh_reduction_percent:.0f}% energiminskning."
        else:
            status_badge = '<span class="renovation-badge neutral">Ej renoverad</span>'
            status_text = "Byggnaden har historiska energideklarationer men ingen betydande renovering har identifierats."

        return f'''
        <section>
            <h2>📈 Energiutveckling</h2>
            <p style="margin-bottom: 1rem; color: var(--gray-700);">
                Historik från Boverkets energideklarationsregister (Gripen).
            </p>

            <div style="display: flex; gap: 1rem; align-items: center; margin-bottom: 1.5rem;">
                {status_badge}
                <span style="color: var(--gray-600); font-size: 0.875rem;">{status_text}</span>
            </div>

            <div class="renovation-comparison" style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.5rem;">
                <div class="stat-card" style="text-align: center;">
                    <div style="font-size: 0.875rem; color: var(--gray-600);">Ursprunglig ({rh.original_year})</div>
                    <div class="value" style="font-size: 2rem;">
                        <span class="energy-class {rh.original_energy_class.lower()}" style="padding: 0.25rem 0.75rem; border-radius: 4px; font-weight: bold;">{rh.original_energy_class}</span>
                    </div>
                    <div style="font-size: 1.25rem; color: var(--gray-700);">{rh.original_kwh_m2:.0f} kWh/m²</div>
                </div>

                <div class="stat-card" style="text-align: center;">
                    <div style="font-size: 0.875rem; color: var(--gray-600);">Nuvarande ({rh.current_year})</div>
                    <div class="value" style="font-size: 2rem;">
                        <span class="energy-class {rh.current_energy_class.lower()}" style="padding: 0.25rem 0.75rem; border-radius: 4px; font-weight: bold;">{rh.current_energy_class}</span>
                    </div>
                    <div style="font-size: 1.25rem; color: var(--gray-700);">{rh.current_kwh_m2:.0f} kWh/m²</div>
                </div>
            </div>

            <h3 style="font-size: 1rem; margin-bottom: 0.5rem;">Tidslinje</h3>
            <div class="timeline" style="display: flex; flex-direction: column; gap: 0.5rem; padding-left: 1rem; border-left: 2px solid var(--gray-300);">
                {timeline_html}
            </div>

            <style>
                .renovation-badge {{
                    padding: 0.25rem 0.75rem;
                    border-radius: 999px;
                    font-size: 0.875rem;
                    font-weight: 600;
                }}
                .renovation-badge.success {{
                    background: #dcfce7;
                    color: #166534;
                }}
                .renovation-badge.neutral {{
                    background: #f3f4f6;
                    color: #6b7280;
                }}
                .timeline-item {{
                    display: flex;
                    gap: 1rem;
                    align-items: center;
                    padding: 0.5rem 0;
                }}
                .timeline-year {{
                    font-weight: 600;
                    min-width: 3rem;
                }}
                .timeline-content {{
                    display: flex;
                    gap: 0.5rem;
                    align-items: center;
                }}
                .energy-class {{
                    padding: 0.125rem 0.5rem;
                    border-radius: 4px;
                    font-weight: 600;
                    font-size: 0.875rem;
                }}
                .energy-class.a {{ background: #22c55e; color: white; }}
                .energy-class.b {{ background: #84cc16; color: white; }}
                .energy-class.c {{ background: #eab308; color: white; }}
                .energy-class.d {{ background: #f97316; color: white; }}
                .energy-class.e {{ background: #ef4444; color: white; }}
                .energy-class.f {{ background: #dc2626; color: white; }}
                .energy-class.g {{ background: #991b1b; color: white; }}
                .kwh-value {{
                    color: var(--gray-600);
                    font-size: 0.875rem;
                }}
            </style>
        </section>
        '''

    def _format_maintenance_plan_section(self, data: ReportData) -> str:
        """Format maintenance plan (long-term cash flow) section."""
        if not data.maintenance_plan:
            return ''

        plan = data.maintenance_plan

        # Build cash flow table rows (first 10 years)
        rows_html = ''
        for proj in plan.projections[:10]:
            year = proj.get('year', '')
            fund_start = proj.get('fund_start_sek', 0)
            investment = proj.get('investment_sek', 0)
            savings = proj.get('energy_savings_sek', 0)
            fund_end = proj.get('fund_end_sek', 0)
            loan = proj.get('loan_balance_sek', 0)
            ecms = proj.get('ecm_investments', [])

            row_class = 'investment-row' if investment > 0 else ''
            if proj.get('fund_warning', False):
                row_class = 'warning-row'

            ecm_note = f" ({', '.join(ecms[:2])})" if ecms else ''

            rows_html += f'''
            <tr class="{row_class}">
                <td>{year}</td>
                <td>{fund_start:,.0f}</td>
                <td>{investment:,.0f}</td>
                <td style="color: var(--success);">{savings:,.0f}</td>
                <td>{fund_end:,.0f}</td>
                <td>{loan:,.0f}</td>
                <td style="font-size: 0.75rem;">{ecm_note}</td>
            </tr>
            '''

        return f'''
        <section>
            <h2>📊 Underhållsplan - Kassaflödesanalys</h2>
            <p style="margin-bottom: 1rem; color: var(--gray-700);">
                Långsiktig kassaflödesprojektion med energiåtgärder integrerade i underhållsplanen.
                <strong>Strategi: Kassaflödeskaskad</strong> - använd snabba besparingar för att finansiera större investeringar.
            </p>

            <div class="cascade-visual"></div>

            <div class="plan-summary">
                <div class="plan-metric positive">
                    <div class="metric-value">{plan.zero_cost_annual_savings:,.0f} SEK</div>
                    <div class="metric-label">Steg 0 besparing/år</div>
                </div>
                <div class="plan-metric">
                    <div class="metric-value">{plan.total_investment_sek:,.0f} SEK</div>
                    <div class="metric-label">Total investering</div>
                </div>
                <div class="plan-metric positive">
                    <div class="metric-value">{plan.total_savings_30yr_sek:,.0f} SEK</div>
                    <div class="metric-label">Total besparing (30 år)</div>
                </div>
                <div class="plan-metric positive">
                    <div class="metric-value">{plan.net_present_value_sek:,.0f} SEK</div>
                    <div class="metric-label">Nuvärde (NPV)</div>
                </div>
                <div class="plan-metric">
                    <div class="metric-value">{plan.break_even_year}</div>
                    <div class="metric-label">Break-even år</div>
                </div>
                <div class="plan-metric">
                    <div class="metric-value">{plan.max_loan_used_sek:,.0f} SEK</div>
                    <div class="metric-label">Max lån använt</div>
                </div>
            </div>

            <h3 style="font-size: 1rem; margin: 1.5rem 0 0.5rem;">År-för-år kassaflöde</h3>
            <div style="overflow-x: auto;">
                <table class="cash-flow-table">
                    <thead>
                        <tr>
                            <th>År</th>
                            <th>Fond start</th>
                            <th>Investering</th>
                            <th>Besparing/år</th>
                            <th>Fond slut</th>
                            <th>Lån</th>
                            <th>Åtgärder</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
            </div>

            <div class="recommendation" style="margin-top: 1rem;">
                <h3>💡 Nyckelinsikt</h3>
                <p>
                    Genom att börja med Steg 0 (nollkostnadsåtgärder) skapas <strong>{plan.zero_cost_annual_savings:,.0f} SEK/år</strong>
                    i omedelbar besparing. Dessa pengar kan sedan användas för att finansiera större åtgärder
                    utan att belasta avgifterna. Planen visar att break-even uppnås {plan.break_even_year} och
                    att slutbalansen efter 30 år blir <strong>{plan.final_fund_balance_sek:,.0f} SEK</strong>.
                </p>
            </div>
        </section>
        '''

    def _format_calibration_section(self, data: ReportData) -> str:
        """Format calibration quality section with ASHRAE metrics and Morris ranking."""
        if not data.calibration_method:
            return ''

        # Determine ASHRAE pass/fail styling
        nmbe_class = "success" if abs(data.ashrae_nmbe) <= 5 else ("warning" if abs(data.ashrae_nmbe) <= 10 else "danger")
        cvrmse_class = "success" if data.ashrae_cvrmse <= 15 else ("warning" if data.ashrae_cvrmse <= 30 else "danger")
        ashrae_status = "✓ Godkänd" if data.ashrae_passes else "✗ Ej godkänd"
        ashrae_status_class = "success" if data.ashrae_passes else "danger"

        # Surrogate quality indicator
        surrogate_class = "success" if data.surrogate_test_r2 >= 0.90 else ("warning" if data.surrogate_test_r2 >= 0.80 else "danger")
        overfit_warning = ""
        if data.surrogate_is_overfit:
            overfit_warning = '''
            <div class="alert" style="background: var(--warning); color: #000; padding: 0.5rem; border-radius: 4px; margin-top: 0.5rem;">
                ⚠️ Varning: Modellen kan vara överanpassad (train R² >> test R²)
            </div>
            '''

        # Morris ranking visualization
        morris_html = ""
        if data.morris_ranking:
            sorted_params = sorted(data.morris_ranking.items(), key=lambda x: x[1])
            param_names_sv = {
                "heat_recovery_eff": "Värmeåtervinning",
                "infiltration_ach": "Luftläckage",
                "window_u_value": "Fönster U-värde",
                "wall_u_value": "Vägg U-värde",
                "roof_u_value": "Tak U-värde",
                "floor_u_value": "Golv U-värde",
                "heating_setpoint": "Börvärde",
            }
            bars_html = ""
            for i, (param, rank) in enumerate(sorted_params):
                name = param_names_sv.get(param, param)
                is_calibrated = data.calibrated_params and param in data.calibrated_params
                bar_class = "primary" if is_calibrated else "secondary"
                importance = 100 - (rank - 1) * 14  # Scale 1-7 to 100-15%
                bars_html += f'''
                    <div style="margin-bottom: 0.5rem;">
                        <div style="display: flex; justify-content: space-between; font-size: 0.85rem;">
                            <span>{name}</span>
                            <span style="color: var(--{bar_class});">{'Kalibrerad' if is_calibrated else 'Fixerad'}</span>
                        </div>
                        <div style="background: var(--bg-secondary); border-radius: 4px; height: 8px; overflow: hidden;">
                            <div style="background: var(--{bar_class}); width: {importance}%; height: 100%;"></div>
                        </div>
                    </div>
                '''

            morris_html = f'''
            <div class="stat-card" style="grid-column: span 2;">
                <h4 style="margin-bottom: 1rem;">Parameterviktighet (Morris Screening)</h4>
                {bars_html}
                <p style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.5rem;">
                    Endast de viktigaste parametrarna kalibreras. Övriga fixeras vid arkettypvärden.
                </p>
            </div>
            '''

        method_name = "Bayesian ABC-SMC" if "bayesian" in data.calibration_method.lower() else "Enkel"

        # Data quality warning display
        data_quality_html = ""
        if data.calibration_data_warning or data.calibration_data_resolution == "annual":
            # Confidence color
            if data.calibration_confidence >= 0.8:
                conf_class = "success"
                conf_label = "Hög"
            elif data.calibration_confidence >= 0.6:
                conf_class = "success"
                conf_label = "Medium"
            elif data.calibration_confidence >= 0.4:
                conf_class = "warning"
                conf_label = "Låg"
            else:
                conf_class = "danger"
                conf_label = "Mycket låg"

            resolution_sv = {
                "hourly": "Timdata (8760 punkter)",
                "monthly": "Månadsdata (12 punkter)",
                "annual": "Årsdata (1 punkt)",
            }.get(data.calibration_data_resolution, "Okänd")

            # Warning message in Swedish
            warning_sv = ""
            if data.calibration_data_resolution == "annual":
                warning_sv = '''
                <div class="alert" style="background: var(--warning); color: #000; padding: 0.75rem; border-radius: 4px; margin-top: 1rem;">
                    <strong>⚠️ Begränsad dataupplösning:</strong> Kalibreringen baseras endast på årsenergianvändning.
                    ASHRAE Guideline 14 rekommenderar månads- eller timdata för tillförlitlig kalibrering.
                    <br>
                    <span style="font-size: 0.85rem;">
                        Med endast årsdata: (1) R² kan inte beräknas, (2) Månadsfel kan ta ut varandra,
                        (3) Modellens noggrannhet är osäker. För produktion, begär månadsvis energistatistik.
                    </span>
                </div>
                '''

            data_quality_html = f'''
            <div class="stat-card" style="margin-top: 1rem;">
                <h4 style="margin-bottom: 0.5rem;">Datakvalitet</h4>
                <div style="display: flex; gap: 2rem; align-items: center;">
                    <div>
                        <div class="value" style="font-size: 1.5rem; color: var(--{conf_class});">{data.calibration_confidence:.0%}</div>
                        <div class="label">Konfidens ({conf_label})</div>
                    </div>
                    <div>
                        <div style="font-weight: 600;">{resolution_sv}</div>
                        <div style="font-size: 0.85rem; color: var(--text-muted);">Dataupplösning</div>
                    </div>
                </div>
                {warning_sv}
            </div>
            '''

        return f'''
        <section>
            <h2>📊 Kalibreringskvalitet</h2>
            <p style="color: var(--text-muted); margin-bottom: 1rem;">
                Baseline-modellen har kalibrerats mot deklarerad energianvändning med {method_name}-metod.
            </p>

            <div class="grid" style="grid-template-columns: repeat(4, 1fr); gap: 1rem;">
                <div class="stat-card">
                    <div class="value">{data.calibrated_kwh_m2:.1f}</div>
                    <div class="label">Kalibrerad (kWh/m²)</div>
                    <div style="font-size: 0.8rem; color: var(--text-muted);">± {data.calibration_std:.1f}</div>
                </div>

                <div class="stat-card">
                    <div class="value" style="color: var(--{nmbe_class});">{data.ashrae_nmbe:+.1f}%</div>
                    <div class="label">NMBE</div>
                    <div style="font-size: 0.8rem; color: var(--text-muted);">Mål: ±5%</div>
                </div>

                <div class="stat-card">
                    <div class="value" style="color: var(--{cvrmse_class});">{data.ashrae_cvrmse:.1f}%</div>
                    <div class="label">CV(RMSE)</div>
                    <div style="font-size: 0.8rem; color: var(--text-muted);">Mål: &lt;15%</div>
                </div>

                <div class="stat-card">
                    <div class="value" style="color: var(--{ashrae_status_class});">{ashrae_status}</div>
                    <div class="label">ASHRAE Guideline 14</div>
                </div>
            </div>

            {data_quality_html}

            <div class="grid" style="grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-top: 1rem;">
                <div class="stat-card">
                    <div class="value" style="color: var(--{surrogate_class});">{data.surrogate_test_r2:.2f}</div>
                    <div class="label">Surrogatmodell R² (test)</div>
                    <div style="font-size: 0.8rem; color: var(--text-muted);">
                        Training: {data.surrogate_r2:.2f}
                    </div>
                    {overfit_warning}
                </div>

                {morris_html}
            </div>
        </section>
        '''

    def _format_complexity_section(self, data: ReportData) -> str:
        """Format building complexity assessment section."""
        # Only show if there's a warning or notable complexity
        if not data.building_complexity_warning and data.building_complexity_score <= 30:
            return ''

        # Determine complexity level and color
        score = data.building_complexity_score
        if score <= 30:
            level = "Enkel"
            level_class = "success"
            icon = "✓"
        elif score <= 60:
            level = "Måttlig"
            level_class = "warning"
            icon = "⚡"
        else:
            level = "Komplex"
            level_class = "danger"
            icon = "⚠️"

        zone_text = "Enzon" if data.single_zone_adequate else "Flerzon rekommenderas"
        zone_class = "success" if data.single_zone_adequate else "warning"

        warning_html = ""
        if data.building_complexity_warning:
            warning_html = f'''
            <div class="alert" style="background: var(--{level_class}); color: {'#000' if level_class == 'warning' else '#fff'}; padding: 0.75rem; border-radius: 4px; margin-top: 1rem;">
                <strong>{icon} {level} byggnadskomplexitet</strong><br>
                <span style="font-size: 0.9rem;">{data.building_complexity_warning}</span>
            </div>
            '''

        return f'''
        <section>
            <h2>🏗️ Modellkomplexitet</h2>
            <p style="color: var(--text-muted); margin-bottom: 1rem;">
                Bedömning av byggnadsgeometri för termisk zonindelning.
            </p>

            <div class="grid" style="grid-template-columns: repeat(3, 1fr); gap: 1rem;">
                <div class="stat-card">
                    <div class="value" style="color: var(--{level_class});">{score:.0f}</div>
                    <div class="label">Komplexitetspoäng (0-100)</div>
                    <div style="font-size: 0.85rem; color: var(--text-muted);">{level}</div>
                </div>

                <div class="stat-card">
                    <div class="value" style="color: var(--{zone_class});">{zone_text}</div>
                    <div class="label">Modelltyp</div>
                </div>

                <div class="stat-card">
                    <div class="value">{100 - score:.0f}%</div>
                    <div class="label">Modellförtroende</div>
                    <div style="font-size: 0.85rem; color: var(--text-muted);">
                        {'Hög' if score <= 30 else ('Måttlig' if score <= 60 else 'Begränsad')} noggrannhet
                    </div>
                </div>
            </div>

            {warning_html}
        </section>
        '''

    def _format_ecm_table(self, data: ReportData) -> str:
        """Format ECM results as HTML table with multi-end-use energy tracking."""
        if not data.ecm_results:
            return '<p><em>Inga simuleringsresultat tillgängliga</em></p>'

        rows = ""
        has_led_note = False
        has_dhw_note = False

        # Sort by total savings if available, else heating savings
        def get_savings(ecm):
            return ecm.total_savings_percent if ecm.total_savings_percent else ecm.savings_percent

        for ecm in sorted(data.ecm_results, key=get_savings, reverse=True):
            ecm_id = ecm.id.lower()
            note = ""

            # Use total savings (all end-uses) if available, else heating-only
            savings_pct = ecm.total_savings_percent if ecm.total_savings_percent else ecm.savings_percent
            result_total = ecm.total_kwh_m2 if ecm.total_kwh_m2 else ecm.result_kwh_m2

            # Build end-use breakdown tooltip
            end_use_detail = ""
            if ecm.savings_by_end_use:
                details = []
                if ecm.savings_by_end_use.get("heating", 0) > 0:
                    details.append(f"värme: -{ecm.savings_by_end_use['heating']:.1f} kWh/m²")
                if ecm.savings_by_end_use.get("dhw", 0) > 0:
                    details.append(f"VV: -{ecm.savings_by_end_use['dhw']:.1f} kWh/m²")
                if ecm.savings_by_end_use.get("property_el", 0) > 0:
                    details.append(f"el: -{ecm.savings_by_end_use['property_el']:.1f} kWh/m²")
                if details:
                    end_use_detail = f' title="{", ".join(details)}"'

            if savings_pct > 0:
                savings_class = "savings-positive"
                savings_text = f"-{savings_pct:.0f}%"
                bar_width = min(savings_pct, 100)
            elif savings_pct < -1:  # Heating increased
                savings_class = "savings-neutral"
                savings_text = f"+{abs(savings_pct):.0f}%"
                bar_width = 0
                # Special note for LED lighting (heating may increase but electricity decreases)
                if 'led' in ecm_id or 'lighting' in ecm_id:
                    note = " *"
                    has_led_note = True
            else:
                savings_class = "savings-neutral"
                savings_text = "0%"
                bar_width = 0

            # Check for DHW/property_el only savings (no heating savings)
            if ecm.savings_by_end_use and ecm.savings_by_end_use.get("heating", 0) == 0:
                if ecm.savings_by_end_use.get("dhw", 0) > 0 or ecm.savings_by_end_use.get("property_el", 0) > 0:
                    note = " †"
                    has_dhw_note = True

            rows += f'''
            <tr{end_use_detail}>
                <td>{ecm.name}{note}</td>
                <td>{result_total:.1f} kWh/m²</td>
                <td class="{savings_class}">{savings_text}</td>
                <td>
                    <div class="chart-bar">
                        <div class="chart-fill" style="width: {bar_width}%;"></div>
                    </div>
                </td>
            </tr>
            '''

        # LED explanation note
        led_note = ""
        if has_led_note:
            led_note = '''
            <p style="margin-top: 0.75rem; font-size: 0.8rem; color: var(--gray-700); font-style: italic;">
                * LED-belysning minskar elförbrukningen med 40-60%, men mindre interna värmetillskott
                ökar värmebehovet i nordiskt klimat. Nettoeffekten är oftast positiv då elbesparingen
                överstiger den ökade värmekostnaden.
            </p>
            '''

        # DHW/property_el note
        dhw_note = ""
        if has_dhw_note:
            dhw_note = '''
            <p style="margin-top: 0.5rem; font-size: 0.8rem; color: var(--gray-700); font-style: italic;">
                † Denna åtgärd påverkar varmvatten eller fastighetsel, inte uppvärmning.
            </p>
            '''

        # Build baseline breakdown text
        baseline_text = f"Baseline: {data.baseline_heating_kwh_m2:.1f} kWh/m²/år (uppvärmning)"
        if data.baseline_total_kwh_m2 > 0:
            baseline_text = f"Baseline total: {data.baseline_total_kwh_m2:.1f} kWh/m²/år"
            breakdown_parts = [f"uppvärmning {data.baseline_heating_kwh_m2:.1f}"]
            if data.baseline_dhw_kwh_m2 > 0:
                breakdown_parts.append(f"VV {data.baseline_dhw_kwh_m2:.1f}")
            if data.baseline_property_el_kwh_m2 > 0:
                breakdown_parts.append(f"fastighetsel {data.baseline_property_el_kwh_m2:.1f}")
            baseline_text += f" ({', '.join(breakdown_parts)})"

        return f'''
        <table>
            <thead>
                <tr>
                    <th>Åtgärd</th>
                    <th>Resultat</th>
                    <th>Besparing</th>
                    <th>Visualisering</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
        <p style="margin-top: 0.5rem; font-size: 0.875rem; color: var(--gray-700);">
            {baseline_text}
        </p>
        {led_note}
        {dhw_note}
        '''

    def _generate_recommendation(self, data: ReportData) -> str:
        """Generate recommendation text based on analysis."""
        if not data.ecm_results:
            return f"Denna byggnad har redan genomfört {len(data.existing_measures)} energiåtgärder. Ytterligare {len(data.applicable_ecms)} åtgärder är möjliga att implementera."

        # Find best ECMs
        positive_ecms = [e for e in data.ecm_results if e.savings_percent > 0]

        if not positive_ecms:
            return f"Denna byggnad är redan mycket energieffektiv med {len(data.existing_measures)} genomförda åtgärder. Ytterligare förbättringar ger marginell effekt."

        best_ecms = sorted(positive_ecms, key=lambda x: x.savings_percent, reverse=True)[:3]
        best_names = [e.name for e in best_ecms]
        total_potential = sum(e.savings_percent for e in best_ecms) * 0.7  # 70% interaction factor

        return f"Rekommenderade åtgärder: {', '.join(best_names)}. Kombinerat kan dessa åtgärder minska energianvändningen med upp till {total_potential:.0f}% (med hänsyn till samverkanseffekter). Byggnaden har redan {len(data.existing_measures)} genomförda energiåtgärder."


def generate_report(
    building_data: dict,
    simulation_results: dict,
    filter_result: dict,
    output_path: Optional[Path] = None,
    baseline_heating: Optional[float] = None,
    packages: Optional[List] = None,
) -> str:
    """
    Convenience function to generate report from analysis results.

    Args:
        building_data: Enriched building JSON data
        simulation_results: Dict of ECM ID -> SimulationResult
        filter_result: Result from SmartECMFilter
        output_path: Optional path to save HTML file
        baseline_heating: Baseline heating from simulation (kWh/m2)
        packages: Optional list of SimulatedPackage from PackageSimulator

    Returns:
        HTML string
    """
    summary = building_data.get('original_summary', {})
    building = building_data.get('buildings', [{}])[0]
    pdf_data = building_data.get('pdf_extracted_data', {})
    solar = building.get('envelope', {}).get('solar_potential', {})

    # Build ECM results
    ecm_results = []
    baseline = baseline_heating or summary.get('energy_performance_kwh_per_sqm', 100)
    atemp_m2 = summary.get('total_heated_area_sqm', 0) or 1000  # Default 1000 m² if not specified

    # Get energy price for payback calculation (default Stockholm district heating)
    energy_price = get_energy_price(region="stockholm", heating_type="district_heating")

    for ecm_id, result in simulation_results.items():
        if result.success and result.parsed_results:
            # Find ECM object - handle both dict and object formats
            ecm_obj = None
            for e in filter_result.get('applicable', []):
                if hasattr(e, 'id') and e.id == ecm_id:
                    ecm_obj = e
                    break
                elif isinstance(e, dict) and e.get('id') == ecm_id:
                    ecm_obj = e.get('ecm', e)
                    break

            result_kwh = result.parsed_results.heating_kwh_m2
            savings = baseline - result_kwh
            savings_pct = (savings / baseline) * 100 if baseline > 0 else 0

            # Calculate cost using V2 database with regional multipliers
            estimated_cost = get_ecm_cost_v2(
                ecm_id=ecm_id,
                atemp_m2=atemp_m2,
                region="stockholm",
                owner_type="brf",
            )

            # Calculate simple payback: cost / annual savings
            annual_savings_sek = savings * atemp_m2 * energy_price
            payback_years = estimated_cost / annual_savings_sek if annual_savings_sek > 0 else 999

            ecm_results.append(ECMResult(
                id=ecm_id,
                name=ecm_obj.name if ecm_obj else ecm_id,
                category=str(ecm_obj.category.value) if ecm_obj and hasattr(ecm_obj.category, 'value') else 'unknown',
                baseline_kwh_m2=baseline,
                result_kwh_m2=result_kwh,
                savings_kwh_m2=savings,
                savings_percent=savings_pct,
                estimated_cost_sek=estimated_cost,
                simple_payback_years=payback_years,
            ))

    # Build existing measures list
    existing = []
    for item in filter_result.get('already_done', []):
        if 'reason' in item:
            # Extract measure from reason
            reason = item['reason']
            if 'implemented:' in reason:
                measure = reason.split('implemented:')[-1].strip()
                existing.append(measure)

    # Build excluded list
    excluded = []
    for item in filter_result.get('already_done', []):
        excluded.append({'ecm': item['ecm'].name, 'reason': item.get('reason', '')})
    for item in filter_result.get('not_applicable', []):
        reasons = item.get('reasons', [])
        reason = reasons[0][1] if reasons else 'Technical constraint'
        excluded.append({'ecm': item['ecm'].name, 'reason': reason})

    # Use provided packages or generate estimated ones
    atemp_m2 = summary.get('total_heated_area_sqm', 0)
    final_packages = packages  # Use simulated packages if provided
    if not final_packages and ecm_results and atemp_m2 > 0:
        # Fall back to estimated packages if no simulated ones provided
        ecm_data_for_packages = [
            {
                'id': ecm.id,
                'name': ecm.name,
                'savings_percent': ecm.savings_percent,
            }
            for ecm in ecm_results
        ]
        package_generator = PackageGenerator()
        final_packages = package_generator.generate_packages(
            ecm_results=ecm_data_for_packages,
            baseline_kwh_m2=baseline,
            atemp_m2=atemp_m2,
        )

    data = ReportData(
        building_name=building_data.get('brf_name', 'Unknown Building'),
        address=f"{building.get('address', 'Unknown')}, {summary.get('location', 'Sweden')}",
        construction_year=summary.get('construction_year', 0),
        building_type=summary.get('building_type', 'Multi-family'),
        facade_material=building.get('envelope', {}).get('facade_material', 'Unknown'),
        atemp_m2=atemp_m2,
        floors=summary.get('floors', 0) or 4,
        energy_class=summary.get('energy_class', 'Unknown'),
        declared_heating_kwh_m2=pdf_data.get('specific_energy_kwh_sqm', summary.get('energy_performance_kwh_per_sqm', 0)),
        baseline_heating_kwh_m2=baseline,
        existing_measures=existing,
        applicable_ecms=[e.id for e in filter_result.get('applicable', [])],
        excluded_ecms=excluded,
        ecm_results=ecm_results,
        packages=final_packages,
        existing_pv_m2=solar.get('existing_pv_sqm', 0),
        remaining_pv_m2=solar.get('remaining_suitable_area_sqm', 0),
        additional_pv_kwp=solar.get('remaining_capacity_kwp', 0),
        analysis_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )

    generator = HTMLReportGenerator()
    return generator.generate(data, output_path)
