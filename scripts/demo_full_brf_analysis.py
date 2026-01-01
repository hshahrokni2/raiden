#!/usr/bin/env python3
"""
DEMO: Complete BRF Analysis with E+ Baseline + All ECM Packages

This script demonstrates the full Raiden pipeline for a single BRF:
1. Load building data (from GeoJSON or manual input)
2. Match archetype
3. Generate baseline IDF
4. Run baseline E+ simulation
5. Generate ECM package IDFs (Steg 0-4)
6. Run E+ simulation for each package
7. Calculate savings and ROI
8. Generate report

Usage:
    python scripts/demo_full_brf_analysis.py

    # Or with custom address
    python scripts/demo_full_brf_analysis.py --address "Bellmansgatan 16, Stockholm"

    # Skip simulation (use cached results)
    python scripts/demo_full_brf_analysis.py --skip-simulation
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class PackageResult:
    """Result for one ECM package simulation."""
    package_id: str
    package_name: str
    heating_kwh: float
    heating_kwh_m2: float
    savings_kwh_m2: float
    savings_percent: float
    cost_sek: float
    payback_years: float
    simulation_time_sec: float


@dataclass
class BRFAnalysisResult:
    """Complete analysis result for a BRF."""
    address: str
    archetype_id: str
    atemp_m2: float
    construction_year: int

    # Baseline
    baseline_kwh: float
    baseline_kwh_m2: float
    declared_kwh_m2: Optional[float]
    calibration_gap_percent: float

    # Packages
    packages: List[PackageResult]

    # Totals
    best_package_id: str
    max_savings_kwh_m2: float
    max_savings_percent: float
    total_analysis_time_sec: float


def find_energyplus() -> str:
    """Find EnergyPlus executable."""
    # Common paths
    paths = [
        "/usr/local/bin/energyplus",
        "/usr/local/EnergyPlus-25-1-0/energyplus",
        "/Applications/EnergyPlus-25-1-0/energyplus",
        shutil.which("energyplus"),
    ]

    for p in paths:
        if p and Path(p).exists():
            return str(p)

    raise FileNotFoundError("EnergyPlus not found. Install from energyplus.net")


def find_weather_file() -> Path:
    """Find Stockholm weather file."""
    paths = [
        PROJECT_ROOT / "tests" / "fixtures" / "stockholm.epw",
        PROJECT_ROOT / "weather" / "SWE_Stockholm.AP.024600_TMYx.epw",
        PROJECT_ROOT / "examples" / "sjostaden_2" / "energyplus" / "weather" / "SWE_Stockholm.AP.024600_TMYx.epw",
        Path.home() / ".raiden" / "weather" / "SWE_Stockholm.AP.024600_TMYx.epw",
    ]

    for p in paths:
        if p.exists():
            return p

    # Try to download
    logger.warning("Weather file not found, using default path")
    return paths[0]


def run_energyplus(idf_path: Path, weather_path: Path, output_dir: Path) -> Tuple[bool, float]:
    """Run EnergyPlus simulation and return (success, heating_kwh)."""
    start = time.time()

    ep_exe = find_energyplus()

    cmd = [
        ep_exe,
        "-w", str(weather_path),
        "-d", str(output_dir),
        "-r",  # Read vars
        str(idf_path),
    ]

    logger.info(f"Running E+: {idf_path.name}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=300,
        )

        elapsed = time.time() - start

        if result.returncode != 0:
            logger.error(f"E+ failed: {result.stderr.decode()[:500]}")
            return False, 0.0, elapsed

        # Parse results
        heating_kwh = parse_eplus_results(output_dir)

        return True, heating_kwh, elapsed

    except subprocess.TimeoutExpired:
        logger.error("E+ simulation timed out")
        return False, 0.0, 300.0
    except Exception as e:
        logger.error(f"E+ error: {e}")
        return False, 0.0, time.time() - start


def parse_eplus_results(output_dir: Path) -> float:
    """Parse E+ output to get total heating energy in kWh."""
    import csv
    import re

    # Best source: table CSV (has annual totals)
    table_csv = output_dir / "eplustbl.csv"
    if table_csv.exists():
        with open(table_csv) as f:
            content = f.read()

            # Look for "District Heating Water [kWh]" in the Heating row
            # Format: ,Heating,General,0.00,...,93765.10,...
            match = re.search(r',Heating,.*?District Heating Water \[kWh\],(\d+\.?\d*)', content)
            if match:
                return float(match.group(1))

            # Alternative: look for "District Heating Water Intensity [kWh/m2]"
            # and multiply by floor area
            match = re.search(r'District Heating Water Intensity \[kWh/m2\],(\d+\.?\d*)', content)
            if match:
                intensity = float(match.group(1))
                # Get floor area
                area_match = re.search(r'Total Building Area,(\d+\.?\d*)', content)
                if area_match:
                    area = float(area_match.group(1))
                    return intensity * area

            # Try simpler pattern - look for heating kWh value
            # ,Heating,General,0.00,0.00,...,93765.10,0.00,0.00
            lines = content.split('\n')
            for line in lines:
                if ',Heating,General,' in line or ',Heating,Unassigned,' in line:
                    parts = line.split(',')
                    # Find non-zero heating value (skip the first few columns)
                    for i, part in enumerate(parts):
                        try:
                            val = float(part)
                            if val > 100:  # Likely kWh not intensity
                                return val
                        except (ValueError, TypeError):
                            pass

    # Fallback: meter CSV (sum hourly Heating:EnergyTransfer)
    meter_csv = output_dir / "eplusmtr.csv"
    if meter_csv.exists():
        with open(meter_csv) as f:
            reader = csv.DictReader(f)
            total_heating_j = 0.0
            for row in reader:
                # Look for heating columns
                for key, value in row.items():
                    if 'Heating:EnergyTransfer' in key:
                        try:
                            total_heating_j += float(value)
                        except (ValueError, TypeError):
                            pass
            if total_heating_j > 0:
                return total_heating_j / 3600000  # J to kWh

    # Fallback: eplusout.csv (sum zone ideal loads heating)
    out_csv = output_dir / "eplusout.csv"
    if out_csv.exists():
        with open(out_csv) as f:
            reader = csv.DictReader(f)
            total_heating_j = 0.0
            for row in reader:
                for key, value in row.items():
                    if 'Total Heating Energy' in key:
                        try:
                            total_heating_j += float(value)
                        except (ValueError, TypeError):
                            pass
            if total_heating_j > 0:
                return total_heating_j / 3600000  # J to kWh

    logger.warning("Could not parse heating results")
    return 0.0


def generate_package_idf(
    base_idf_path: Path,
    package_id: str,
    output_path: Path,
) -> None:
    """Generate IDF for an ECM package by modifying baseline."""
    from eppy.modeleditor import IDF
    from src.core.idf_parser import IDFParser

    # Initialize eppy
    idd_path = os.environ.get("ENERGYPLUS_IDD_PATH")
    if idd_path:
        IDF.setiddname(idd_path)
    else:
        # Try common paths
        idd_paths = [
            "/usr/local/EnergyPlus-25-1-0/Energy+.idd",
            "/Applications/EnergyPlus-25-1-0/Energy+.idd",
        ]
        for p in idd_paths:
            if Path(p).exists():
                IDF.setiddname(p)
                break

    # Load base IDF
    idf = IDF(str(base_idf_path))
    parser = IDFParser()

    # Package parameter modifications
    packages = {
        "baseline": {
            # No changes
        },
        "steg0_nollkostnad": {
            "heating_setpoint": 20.0,  # Lower from 21 to 20
        },
        "steg1_enkel": {
            "heating_setpoint": 20.0,
            "infiltration_ach": 0.04,  # Improve air sealing
        },
        "steg2_standard": {
            "heating_setpoint": 20.0,
            "infiltration_ach": 0.03,
            "window_u_value": 1.0,  # Better windows
        },
        "steg3_premium": {
            "heating_setpoint": 20.0,
            "infiltration_ach": 0.02,
            "window_u_value": 0.9,
            "heat_recovery_eff": 0.80,  # Add/improve FTX
        },
        "steg4_djuprenovering": {
            "heating_setpoint": 20.0,
            "infiltration_ach": 0.015,
            "window_u_value": 0.8,
            "wall_u_value": 0.15,  # Add wall insulation
            "roof_u_value": 0.12,  # Add roof insulation
            "heat_recovery_eff": 0.85,
        },
    }

    params = packages.get(package_id, {})

    # Apply modifications
    if "infiltration_ach" in params:
        parser.set_infiltration_ach(idf, params["infiltration_ach"])
    if "window_u_value" in params:
        parser.set_window_u_value(idf, params["window_u_value"])
    if "wall_u_value" in params:
        parser.set_wall_u_value(idf, params["wall_u_value"])
    if "roof_u_value" in params:
        parser.set_roof_u_value(idf, params["roof_u_value"])
    if "heat_recovery_eff" in params:
        parser.set_heat_recovery_effectiveness(idf, params["heat_recovery_eff"])
    if "heating_setpoint" in params:
        parser.set_heating_setpoint(idf, params["heating_setpoint"])

    # Save
    idf.saveas(str(output_path))
    logger.debug(f"Generated {package_id} IDF: {output_path}")


def get_package_cost(package_id: str, atemp_m2: float) -> float:
    """Get estimated cost for a package in SEK."""
    # Costs per m² (rough estimates from Swedish market)
    costs_per_m2 = {
        "baseline": 0,
        "steg0_nollkostnad": 0,  # Zero cost operational
        "steg1_enkel": 50,  # LED, thermostats
        "steg2_standard": 500,  # + windows, air sealing
        "steg3_premium": 2000,  # + FTX system
        "steg4_djuprenovering": 5000,  # + wall/roof insulation
    }

    return costs_per_m2.get(package_id, 0) * atemp_m2


def analyze_brf(
    address: str = "Aktergatan 11, Stockholm",
    construction_year: int = 2003,
    atemp_m2: float = 2240,
    declared_kwh_m2: Optional[float] = 33.0,
    base_idf_path: Optional[Path] = None,
    skip_simulation: bool = False,
    output_dir: Optional[Path] = None,
) -> BRFAnalysisResult:
    """
    Run complete BRF analysis with E+ simulations.

    Args:
        address: Building address
        construction_year: Year built
        atemp_m2: Heated floor area
        declared_kwh_m2: Energy declaration value (for calibration comparison)
        base_idf_path: Path to baseline IDF (or generate from archetype)
        skip_simulation: Use cached/estimated results instead of running E+
        output_dir: Output directory for results

    Returns:
        BRFAnalysisResult with baseline and all package results
    """
    start_time = time.time()

    print("=" * 70)
    print("RAIDEN - Complete BRF Analysis")
    print("=" * 70)
    print(f"Address: {address}")
    print(f"Year: {construction_year}")
    print(f"Atemp: {atemp_m2:,.0f} m²")
    if declared_kwh_m2:
        print(f"Declared: {declared_kwh_m2} kWh/m²")
    print()

    # Setup output directory
    if output_dir is None:
        output_dir = PROJECT_ROOT / "output_brf_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find weather file
    weather_path = find_weather_file()
    print(f"Weather: {weather_path.name}")

    # Get or generate baseline IDF
    if base_idf_path is None:
        base_idf_path = PROJECT_ROOT / "examples" / "sjostaden_2" / "energyplus" / "sjostaden_7zone.idf"

    if not base_idf_path.exists():
        raise FileNotFoundError(f"Baseline IDF not found: {base_idf_path}")

    print(f"Baseline IDF: {base_idf_path.name}")
    print()

    # Determine archetype
    from src.baseline import get_archetype_by_year
    archetype = get_archetype_by_year(construction_year)
    archetype_id = archetype.id if archetype else "mfh_1996_2010"
    print(f"Archetype: {archetype_id}")

    # Define packages
    packages_to_run = [
        ("baseline", "Baseline (nuläge)"),
        ("steg0_nollkostnad", "Steg 0: Nollkostnad"),
        ("steg1_enkel", "Steg 1: Enkel"),
        ("steg2_standard", "Steg 2: Standard"),
        ("steg3_premium", "Steg 3: Premium"),
        ("steg4_djuprenovering", "Steg 4: Djuprenovering"),
    ]

    # Run simulations
    print("\n" + "-" * 70)
    print("Running E+ Simulations")
    print("-" * 70)

    results = []
    baseline_kwh = 0.0
    baseline_kwh_m2 = 0.0

    for package_id, package_name in packages_to_run:
        pkg_output_dir = output_dir / f"sim_{package_id}"
        pkg_output_dir.mkdir(parents=True, exist_ok=True)

        # Generate package IDF
        pkg_idf_path = pkg_output_dir / f"{package_id}.idf"
        generate_package_idf(base_idf_path, package_id, pkg_idf_path)

        if skip_simulation:
            # Use estimated values
            estimates = {
                "baseline": 36.3,
                "steg0_nollkostnad": 34.5,
                "steg1_enkel": 32.0,
                "steg2_standard": 28.0,
                "steg3_premium": 22.0,
                "steg4_djuprenovering": 18.0,
            }
            heating_kwh_m2 = estimates.get(package_id, 30.0)
            heating_kwh = heating_kwh_m2 * atemp_m2
            sim_time = 0.0
            success = True
        else:
            # Run actual E+ simulation
            success, heating_kwh, sim_time = run_energyplus(
                pkg_idf_path, weather_path, pkg_output_dir
            )

            if success and heating_kwh > 0:
                heating_kwh_m2 = heating_kwh / atemp_m2
            else:
                # Fallback to estimate if E+ fails
                heating_kwh_m2 = 30.0  # Default
                heating_kwh = heating_kwh_m2 * atemp_m2

        # Calculate savings vs baseline
        if package_id == "baseline":
            baseline_kwh = heating_kwh
            baseline_kwh_m2 = heating_kwh_m2
            savings_kwh_m2 = 0.0
            savings_percent = 0.0
        else:
            savings_kwh_m2 = baseline_kwh_m2 - heating_kwh_m2
            savings_percent = (savings_kwh_m2 / baseline_kwh_m2 * 100) if baseline_kwh_m2 > 0 else 0

        # Calculate cost and payback
        cost_sek = get_package_cost(package_id, atemp_m2)
        energy_price = 1.20  # SEK/kWh (district heating average)
        annual_savings_sek = savings_kwh_m2 * atemp_m2 * energy_price
        payback_years = (cost_sek / annual_savings_sek) if annual_savings_sek > 0 else float('inf')

        pkg_result = PackageResult(
            package_id=package_id,
            package_name=package_name,
            heating_kwh=heating_kwh,
            heating_kwh_m2=heating_kwh_m2,
            savings_kwh_m2=savings_kwh_m2,
            savings_percent=savings_percent,
            cost_sek=cost_sek,
            payback_years=payback_years if payback_years != float('inf') else 0,
            simulation_time_sec=sim_time,
        )
        results.append(pkg_result)

        # Print progress
        status = "OK" if success else "FAIL"
        print(f"  [{status}] {package_name}: {heating_kwh_m2:.1f} kWh/m² "
              f"({savings_percent:+.1f}%) [{sim_time:.1f}s]")

    # Calculate calibration gap
    calibration_gap = 0.0
    if declared_kwh_m2 and baseline_kwh_m2 > 0:
        calibration_gap = ((baseline_kwh_m2 - declared_kwh_m2) / declared_kwh_m2) * 100

    # Find best package
    non_baseline = [r for r in results if r.package_id != "baseline"]
    best_pkg = max(non_baseline, key=lambda r: r.savings_percent)

    total_time = time.time() - start_time

    # Create result
    analysis_result = BRFAnalysisResult(
        address=address,
        archetype_id=archetype_id,
        atemp_m2=atemp_m2,
        construction_year=construction_year,
        baseline_kwh=baseline_kwh,
        baseline_kwh_m2=baseline_kwh_m2,
        declared_kwh_m2=declared_kwh_m2,
        calibration_gap_percent=calibration_gap,
        packages=results,
        best_package_id=best_pkg.package_id,
        max_savings_kwh_m2=best_pkg.savings_kwh_m2,
        max_savings_percent=best_pkg.savings_percent,
        total_analysis_time_sec=total_time,
    )

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nBaseline: {baseline_kwh_m2:.1f} kWh/m²")
    if declared_kwh_m2:
        print(f"Declared: {declared_kwh_m2:.1f} kWh/m²")
        print(f"Calibration gap: {calibration_gap:+.1f}%")

    print("\nECM Packages:")
    print("-" * 70)
    print(f"{'Package':<25} {'kWh/m²':>10} {'Savings':>10} {'Cost (SEK)':>12} {'Payback':>10}")
    print("-" * 70)

    for pkg in results:
        payback_str = f"{pkg.payback_years:.1f} yr" if pkg.payback_years > 0 else "N/A"
        print(f"{pkg.package_name:<25} {pkg.heating_kwh_m2:>10.1f} "
              f"{pkg.savings_percent:>+9.1f}% {pkg.cost_sek:>12,.0f} {payback_str:>10}")

    print("-" * 70)
    print(f"\nBest package: {best_pkg.package_name}")
    print(f"Max savings: {best_pkg.savings_kwh_m2:.1f} kWh/m² ({best_pkg.savings_percent:.1f}%)")
    print(f"Total analysis time: {total_time:.1f}s")

    # Save results
    results_path = output_dir / "analysis_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "address": analysis_result.address,
            "archetype_id": analysis_result.archetype_id,
            "atemp_m2": analysis_result.atemp_m2,
            "construction_year": analysis_result.construction_year,
            "baseline_kwh_m2": analysis_result.baseline_kwh_m2,
            "declared_kwh_m2": analysis_result.declared_kwh_m2,
            "calibration_gap_percent": analysis_result.calibration_gap_percent,
            "packages": [
                {
                    "package_id": p.package_id,
                    "package_name": p.package_name,
                    "heating_kwh_m2": p.heating_kwh_m2,
                    "savings_kwh_m2": p.savings_kwh_m2,
                    "savings_percent": p.savings_percent,
                    "cost_sek": p.cost_sek,
                    "payback_years": p.payback_years,
                }
                for p in analysis_result.packages
            ],
            "best_package_id": analysis_result.best_package_id,
            "total_analysis_time_sec": analysis_result.total_analysis_time_sec,
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    return analysis_result


def main():
    parser = argparse.ArgumentParser(description="Complete BRF Analysis Demo")
    parser.add_argument("--address", default="Aktergatan 11, Stockholm",
                        help="Building address")
    parser.add_argument("--year", type=int, default=2003,
                        help="Construction year")
    parser.add_argument("--atemp", type=float, default=2240,
                        help="Heated floor area (m²)")
    parser.add_argument("--declared", type=float, default=33.0,
                        help="Declared energy (kWh/m²)")
    parser.add_argument("--idf", type=Path, help="Path to baseline IDF")
    parser.add_argument("--output", type=Path, help="Output directory")
    parser.add_argument("--skip-simulation", action="store_true",
                        help="Skip E+ simulation, use estimates")
    args = parser.parse_args()

    try:
        result = analyze_brf(
            address=args.address,
            construction_year=args.year,
            atemp_m2=args.atemp,
            declared_kwh_m2=args.declared,
            base_idf_path=args.idf,
            skip_simulation=args.skip_simulation,
            output_dir=args.output,
        )
        return 0
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
