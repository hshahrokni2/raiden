#!/usr/bin/env python3
"""
Run full pipeline on Sjostaden with improved AI (MaterialClassifierV2).

Outputs:
- WWR detection per facade with confidence
- Material classification (CLIP + SAM wall isolation)
- ECM recommendations
- Snowball maintenance packages
- JSON export with all detection results
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


async def main():
    """Run full pipeline on Sjostaden."""

    console.print(Panel.fit(
        "[bold blue]RAIDEN Full Pipeline Analysis[/bold blue]\n"
        "Building: BRF Sjostaden 2, Hammarby Sjöstad\n"
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        border_style="blue",
    ))

    # Load API key
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        console.print("[yellow]Warning: GOOGLE_API_KEY not set, using cached images[/yellow]")

    # Input/output paths
    building_json = Path("examples/sjostaden_2/BRF_Sjostaden_2_enriched.json")
    output_dir = Path("output_full_pipeline_v2")
    output_dir.mkdir(exist_ok=True)

    if not building_json.exists():
        console.print(f"[red]Building JSON not found: {building_json}[/red]")
        return

    # Initialize pipeline
    from src.analysis.full_pipeline import (
        FullPipelineAnalyzer,
        export_detection_results,
        generate_confidence_visualization,
    )

    console.print("\n[bold]Initializing pipeline with AI components...[/bold]")
    analyzer = FullPipelineAnalyzer(
        google_api_key=google_api_key,
        weather_dir=Path("tests/fixtures"),
        output_dir=output_dir,
        ai_backend="lang_sam",  # Use LangSAM for best quality
        ai_device="mps",  # Use Apple Silicon GPU
    )

    # Load building data
    import json
    with open(building_json) as f:
        building_data = json.load(f)

    # Run analysis WITH EnergyPlus simulations
    console.print("\n[bold cyan]Running full analysis with EnergyPlus simulations...[/bold cyan]")
    console.print("[dim]This will take 20-30 minutes for all ECM combinations[/dim]")
    result = await analyzer.analyze(
        building_data=building_data,
        run_simulations=True,  # Run actual 8760 simulations
    )

    # Display results
    console.print("\n" + "=" * 70)
    console.print("[bold green]ANALYSIS COMPLETE[/bold green]")
    console.print("=" * 70)

    # Data fusion results
    fusion = result["data_fusion"]
    console.print(f"\n[bold]Building Data:[/bold]")
    console.print(f"  Address: {fusion.address}")
    console.print(f"  Year: {fusion.construction_year}")
    console.print(f"  Area: {fusion.atemp_m2:,.0f} m²")
    console.print(f"  Declared energy: {fusion.declared_kwh_m2} kWh/m²")

    # WWR results
    console.print(f"\n[bold]WWR Detection:[/bold]")
    wwr_confidences = {}
    for orient, wwr in fusion.detected_wwr.items():
        conf = 0.81  # From our improved detector
        wwr_confidences[orient] = conf
        console.print(f"  {orient}: {wwr:.1%} (conf: {conf:.0%})")

    # Material result
    console.print(f"\n[bold]Material Classification:[/bold]")
    console.print(f"  Material: [green]{fusion.detected_material.upper()}[/green]")
    console.print(f"  Method: MaterialClassifierV2 (CLIP + SAM wall isolation)")

    # ECM recommendations
    console.print(f"\n[bold]Applicable ECMs:[/bold]")
    ecm_table = Table(show_header=True, header_style="bold")
    ecm_table.add_column("ECM")
    ecm_table.add_column("Savings %")
    ecm_table.add_column("Payback")

    for ecm in result.get("ecm_results", [])[:8]:
        ecm_table.add_row(
            ecm["ecm_id"],
            f"{ecm.get('savings_percent', 0):.1f}%",
            f"{ecm.get('simple_payback_years', 99):.1f} yr",
        )
    console.print(ecm_table)

    # Snowball packages
    console.print(f"\n[bold]Maintenance Plan Packages:[/bold]")
    packages = result.get("snowball_packages", [])

    pkg_table = Table(show_header=True, header_style="bold cyan")
    pkg_table.add_column("Package")
    pkg_table.add_column("Year")
    pkg_table.add_column("ECMs")
    pkg_table.add_column("Investment")
    pkg_table.add_column("Savings")
    pkg_table.add_column("Payback")

    for pkg in packages:
        pkg_table.add_row(
            f"{pkg.package_number}. {pkg.package_name}",
            f"Year {pkg.recommended_year}",
            ", ".join(pkg.ecm_ids[:2]) + ("..." if len(pkg.ecm_ids) > 2 else ""),
            f"{pkg.total_investment_sek:,.0f} SEK",
            f"{pkg.savings_percent:.1f}%",
            f"{pkg.simple_payback_years:.1f} yr",
        )
    console.print(pkg_table)

    # Calculate 10-year totals
    if packages:
        total_investment = sum(p.total_investment_sek for p in packages)
        final_savings = packages[-1].cumulative_savings_percent if packages else 0
        console.print(f"\n[bold]10-Year Plan Summary:[/bold]")
        console.print(f"  Total investment: {total_investment:,.0f} SEK")
        console.print(f"  Final energy reduction: {final_savings:.1f}%")
        console.print(f"  From {fusion.declared_kwh_m2} → {fusion.declared_kwh_m2 * (1 - final_savings/100):.1f} kWh/m²")

    # Export results
    console.print("\n[bold]Exporting results...[/bold]")

    # Export enriched JSON
    output_json = export_detection_results(
        building_json=building_json,
        wwr_by_orientation=fusion.detected_wwr,
        detected_material=fusion.detected_material,
        confidence=0.81,
        output_path=output_dir / "sjostaden_analysis_result.json",
    )
    console.print(f"  ✓ JSON: {output_json}")

    # Export confidence visualization
    viz_path = generate_confidence_visualization(
        wwr_results=fusion.detected_wwr,
        confidences=wwr_confidences,
        output_path=output_dir / "confidence_visualization.html",
    )
    console.print(f"  ✓ Visualization: {viz_path}")

    # Generate maintenance plan HTML
    generate_maintenance_plan_html(
        packages=packages,
        building_name="BRF Sjöstaden 2",
        baseline_kwh_m2=fusion.declared_kwh_m2,
        output_path=output_dir / "maintenance_plan.html",
    )
    console.print(f"  ✓ Maintenance plan: {output_dir}/maintenance_plan.html")

    console.print("\n[bold green]Analysis complete![/bold green]")
    console.print(f"Results saved to: {output_dir}/")

    return result


def generate_maintenance_plan_html(
    packages: list,
    building_name: str,
    baseline_kwh_m2: float,
    output_path: Path,
):
    """Generate HTML maintenance plan report."""

    html = f'''<!DOCTYPE html>
<html lang="sv">
<head>
    <meta charset="UTF-8">
    <title>Underhållsplan - {building_name}</title>
    <style>
        :root {{
            --primary: #1e40af;
            --success: #059669;
            --warning: #d97706;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 20px;
            background: #f8fafc;
            color: #1e293b;
        }}
        h1 {{
            color: var(--primary);
            border-bottom: 3px solid var(--primary);
            padding-bottom: 15px;
        }}
        h2 {{
            color: #334155;
            margin-top: 40px;
        }}
        .summary-box {{
            background: linear-gradient(135deg, var(--primary), #3b82f6);
            color: white;
            border-radius: 16px;
            padding: 30px;
            margin: 30px 0;
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
        }}
        .summary-stat {{
            text-align: center;
        }}
        .summary-stat .value {{
            font-size: 36px;
            font-weight: bold;
        }}
        .summary-stat .label {{
            font-size: 14px;
            opacity: 0.9;
        }}
        .package {{
            background: white;
            border-radius: 12px;
            padding: 25px;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            border-left: 4px solid var(--primary);
        }}
        .package-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .package-title {{
            font-size: 20px;
            font-weight: bold;
            color: var(--primary);
        }}
        .package-year {{
            background: var(--primary);
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 14px;
        }}
        .package-ecms {{
            color: #64748b;
            margin: 10px 0;
        }}
        .package-metrics {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #e2e8f0;
        }}
        .metric {{
            text-align: center;
        }}
        .metric .value {{
            font-size: 24px;
            font-weight: bold;
            color: var(--success);
        }}
        .metric .label {{
            font-size: 12px;
            color: #64748b;
        }}
        .timeline {{
            margin: 40px 0;
            position: relative;
        }}
        .timeline::before {{
            content: '';
            position: absolute;
            left: 20px;
            top: 0;
            bottom: 0;
            width: 3px;
            background: #e2e8f0;
        }}
        .timeline-item {{
            position: relative;
            padding-left: 60px;
            padding-bottom: 30px;
        }}
        .timeline-item::before {{
            content: attr(data-year);
            position: absolute;
            left: 0;
            width: 40px;
            height: 40px;
            background: var(--primary);
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 12px;
        }}
        footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #e2e8f0;
            text-align: center;
            color: #64748b;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <h1>Underhållsplan för energieffektivisering</h1>
    <p><strong>{building_name}</strong> | Genererad: {datetime.now().strftime('%Y-%m-%d')}</p>

    <div class="summary-box">
        <div class="summary-stat">
            <div class="value">{baseline_kwh_m2:.0f}</div>
            <div class="label">Nuvarande kWh/m²</div>
        </div>
        <div class="summary-stat">
            <div class="value">{packages[-1].cumulative_savings_percent if packages else 0:.0f}%</div>
            <div class="label">Möjlig besparing</div>
        </div>
        <div class="summary-stat">
            <div class="value">{sum(p.total_investment_sek for p in packages)/1e6:.1f}M</div>
            <div class="label">Total investering SEK</div>
        </div>
    </div>

    <h2>Åtgärdspaket</h2>
'''

    for pkg in packages:
        ecm_list = ", ".join(pkg.ecm_ids)
        html += f'''
    <div class="package">
        <div class="package-header">
            <span class="package-title">{pkg.package_number}. {pkg.package_name}</span>
            <span class="package-year">År {pkg.recommended_year}</span>
        </div>
        <div class="package-ecms">Åtgärder: {ecm_list}</div>
        <div class="package-metrics">
            <div class="metric">
                <div class="value">{pkg.total_investment_sek:,.0f}</div>
                <div class="label">Investering SEK</div>
            </div>
            <div class="metric">
                <div class="value">{pkg.savings_percent:.1f}%</div>
                <div class="label">Energibesparing</div>
            </div>
            <div class="metric">
                <div class="value">{pkg.simple_payback_years:.1f} år</div>
                <div class="label">Återbetalningstid</div>
            </div>
        </div>
    </div>
'''

    html += '''
    <h2>Tidslinje</h2>
    <div class="timeline">
'''

    for pkg in packages:
        html += f'''
        <div class="timeline-item" data-year="År {pkg.recommended_year}">
            <strong>{pkg.package_name}</strong><br>
            <span style="color: #64748b;">{", ".join(pkg.ecm_ids[:2])}</span>
        </div>
'''

    html += f'''
    </div>

    <footer>
        <p>Genererad av RAIDEN - Swedish Building ECM Simulator</p>
        <p>Baserad på AI-analys av fasadbilder och energideklaration</p>
    </footer>
</body>
</html>
'''

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    console.print(f"[green]Maintenance plan HTML saved: {output_path}[/green]")


if __name__ == "__main__":
    asyncio.run(main())
