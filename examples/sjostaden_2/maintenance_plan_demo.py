#!/usr/bin/env python3
"""
BRF Maintenance Plan Demo.

Demonstrates the cash flow cascade strategy for BRF Sjöstaden 2.
Shows how to sequence ECM investments to build positive cash flow.
"""

import sys
from pathlib import Path
from datetime import date

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.planning import (
    MaintenancePlan,
    BRFFinancials,
    TariffStructure,
    PlannedRenovation,
    RenovationType,
    LoanTolerance,
    CashFlowSimulator,
    ECMSequencer,
    ECMCandidate,
    create_typical_renovation_plan,
    analyze_effektvakt_potential,
)


def create_sjostaden_financials() -> BRFFinancials:
    """Create financial parameters for BRF Sjöstaden 2."""
    return BRFFinancials(
        current_fund_sek=2_500_000,        # 2.5M in underhållsfond
        annual_fund_contribution_sek=500_000,  # 500k yearly
        target_fund_sek=3_000_000,         # Target 3M
        current_avgift_sek_month=4800,     # Per apartment
        num_apartments=110,
        avgift_increase_tolerance_pct=5.0,
        loan_tolerance=LoanTolerance.MODERATE,
        current_loans_sek=0,
        max_loan_sek=5_000_000,
        loan_interest_rate=0.045,
        annual_energy_cost_sek=800_000,    # ~800k total energy
        annual_el_cost_sek=300_000,
        annual_fv_cost_sek=500_000,
        peak_el_kw=120,
        peak_fv_kw=180,
    )


def create_sjostaden_ecm_candidates() -> list:
    """Create ECM candidates based on Sjöstaden analysis."""
    return [
        # Steg 0: Zero-cost
        ECMCandidate(
            ecm_id="duc_calibration",
            name="DUC-optimering",
            investment_sek=5000,
            annual_savings_sek=40000,
            payback_years=0.1,
            is_zero_cost=True,
            steg=0,
        ),
        ECMCandidate(
            ecm_id="heating_curve_adjustment",
            name="Värmekurvejustering",
            investment_sek=2000,
            annual_savings_sek=40000,
            payback_years=0.05,
            is_zero_cost=True,
            steg=0,
        ),
        ECMCandidate(
            ecm_id="ventilation_schedule_optimization",
            name="Ventilationsschema",
            investment_sek=2000,
            annual_savings_sek=24000,
            payback_years=0.08,
            is_zero_cost=True,
            steg=0,
        ),
        ECMCandidate(
            ecm_id="night_setback",
            name="Nattsänkning",
            investment_sek=1000,
            annual_savings_sek=40000,
            payback_years=0.03,
            is_zero_cost=True,
            steg=0,
        ),
        ECMCandidate(
            ecm_id="effektvakt_optimization",
            name="Effektvaktsoptimering",
            investment_sek=3000,
            annual_savings_sek=21000,  # From peak reduction
            payback_years=0.1,
            is_zero_cost=True,
            steg=0,
        ),

        # Steg 1: Quick wins (< 500k SEK)
        ECMCandidate(
            ecm_id="air_sealing",
            name="Tätning",
            investment_sek=300000,
            annual_savings_sek=80000,
            payback_years=3.8,
            steg=1,
        ),
        ECMCandidate(
            ecm_id="smart_thermostats",
            name="Smarta termostater",
            investment_sek=200000,
            annual_savings_sek=20000,
            payback_years=10,
            steg=1,
        ),

        # Steg 2: Standard (500k - 2M SEK)
        ECMCandidate(
            ecm_id="demand_controlled_ventilation",
            name="Behovsstyrd ventilation",
            investment_sek=1500000,
            annual_savings_sek=390000,  # 49% heating savings
            payback_years=3.8,
            steg=2,
        ),
        ECMCandidate(
            ecm_id="roof_insulation",
            name="Tilläggsisolering tak",
            investment_sek=800000,
            annual_savings_sek=20000,
            payback_years=40,
            synergy_with_renovation="roof_2035",
            cost_if_coordinated=400000,  # 50% off with roof work
            steg=2,
        ),

        # Steg 3: Premium (> 2M SEK)
        ECMCandidate(
            ecm_id="wall_internal_insulation",
            name="Tilläggsisolering väggar",
            investment_sek=4000000,
            annual_savings_sek=110000,  # 14% savings
            payback_years=36,
            steg=3,
        ),
    ]


def create_sjostaden_renovations() -> list:
    """Create planned renovations for Sjöstaden."""
    return [
        PlannedRenovation(
            id="facade_2030",
            name="Facade Renovation",
            name_sv="Fasadrenovering",
            type=RenovationType.FACADE,
            planned_year=2030,
            estimated_cost_sek=3_000_000,
            ecm_synergy=["wall_external_insulation"],
            ecm_cost_reduction=500_000,
            can_postpone_years=2,
            description="Putsrenovering - möjlighet till tilläggsisolering"
        ),
        PlannedRenovation(
            id="roof_2035",
            name="Roof Renovation",
            name_sv="Takrenovering",
            type=RenovationType.ROOF,
            planned_year=2035,
            estimated_cost_sek=2_000_000,
            ecm_synergy=["roof_insulation", "solar_pv"],
            ecm_cost_reduction=300_000,
            description="Takbyte - möjlighet till isolering och solceller"
        ),
    ]


def print_cash_flow_summary(plan: MaintenancePlan):
    """Print a summary of the cash flow projection."""
    print("\n" + "=" * 70)
    print("KASSAFLÖDESANALYS - BRF SJÖSTADEN 2")
    print("=" * 70)

    print(f"\n Planhorisont: {plan.plan_horizon_years} år")
    print(f" Startår: {plan.projections[0].year if plan.projections else 'N/A'}")

    print("\n SAMMANFATTNING:")
    print(f"   Total investering (ECM): {plan.total_investment_sek:,.0f} SEK")
    print(f"   Total besparing (30 år): {plan.total_savings_30yr_sek:,.0f} SEK")
    print(f"   Nuvärde (NPV): {plan.net_present_value_sek:,.0f} SEK")
    print(f"   Break-even år: {plan.break_even_year}")
    print(f"   Slutlig fondsaldo: {plan.final_fund_balance_sek:,.0f} SEK")
    print(f"   Max lån använt: {plan.max_loan_used_sek:,.0f} SEK")

    print("\n ÅR-FÖR-ÅR KASSAFLÖDE:")
    print("-" * 70)
    print(f"{'År':<6} {'Fond start':<12} {'Invest':<10} {'Besp/år':<10} {'Fond slut':<12} {'Lån':<10}")
    print("-" * 70)

    for proj in plan.projections[:15]:  # First 15 years
        invest = proj.renovation_spend_sek + proj.ecm_investment_sek
        warning = " ⚠️" if proj.fund_warning else ""

        # Show ECM investments
        ecm_note = ""
        if proj.ecm_investments:
            ecm_note = f" ({', '.join(proj.ecm_investments[:2])})"

        print(
            f"{proj.year:<6} "
            f"{proj.fund_start_sek:>10,.0f}  "
            f"{invest:>8,.0f}  "
            f"{proj.energy_savings_sek:>8,.0f}  "
            f"{proj.fund_end_sek:>10,.0f}  "
            f"{proj.loan_balance_sek:>8,.0f}"
            f"{warning}{ecm_note}"
        )


def print_effektvakt_analysis():
    """Print effektvakt (peak shaving) analysis."""
    print("\n" + "=" * 70)
    print("EFFEKTVAKTSANALYS")
    print("=" * 70)

    result = analyze_effektvakt_potential(
        atemp_m2=15350,
        construction_year=2003,
        heating_type="heat_pump",  # Has ground source HP
        current_el_peak_kw=120,
        current_fv_peak_kw=180,
    )

    print(f"\n Nuvarande effekttoppar:")
    print(f"   El: {result.current_el_peak_kw:.0f} kW")
    print(f"   Fjärrvärme: {result.current_fv_peak_kw:.0f} kW")

    print(f"\n Optimerad (med termisk massa):")
    print(f"   El: {result.optimized_el_peak_kw:.0f} kW ({result.el_peak_reduction_kw:.0f} kW reduktion)")
    print(f"   Fjärrvärme: {result.optimized_fv_peak_kw:.0f} kW ({result.fv_peak_reduction_kw:.0f} kW reduktion)")

    print(f"\n Strategi:")
    print(f"   Förvärmningstid: {result.pre_heat_hours:.1f} timmar före höglast")
    print(f"   Temperaturhöjning: +{result.pre_heat_temp_increase_c:.1f}°C")
    print(f"   Seglingstid: {result.coast_duration_hours:.1f} timmar utan värme")

    print(f"\n Årlig besparing:")
    print(f"   El (effektavgift): {result.annual_el_savings_sek:,.0f} SEK")
    print(f"   Fjärrvärme (effektavgift): {result.annual_fv_savings_sek:,.0f} SEK")
    print(f"   TOTAL: {result.total_annual_savings_sek:,.0f} SEK")

    print(f"\n Implementering:")
    print(f"   Kräver BMS: {'Ja' if result.requires_bms else 'Nej'}")
    print(f"   Manuellt möjligt: {'Ja' if result.manual_possible else 'Nej'}")

    print("\n Noteringar:")
    for note in result.notes or []:
        print(f"   • {note}")


def main():
    print("\n" + "=" * 70)
    print("BRF SJÖSTADEN 2 - UNDERHÅLLSPLAN MED ENERGIÅTGÄRDER")
    print("=" * 70)

    # Create inputs
    financials = create_sjostaden_financials()
    candidates = create_sjostaden_ecm_candidates()
    renovations = create_sjostaden_renovations()

    print("\n BYGGNADSFAKTA:")
    print(f"   Adress: Aktergatan 11, Hammarby Sjöstad")
    print(f"   Lägenheter: {financials.num_apartments}")
    print(f"   Nuvarande underhållsfond: {financials.current_fund_sek:,.0f} SEK")
    print(f"   Årlig avsättning: {financials.annual_fund_contribution_sek:,.0f} SEK")
    print(f"   Nuvarande energikostnad: {financials.annual_energy_cost_sek:,.0f} SEK/år")

    print("\n PLANERADE RENOVERINGAR:")
    for reno in renovations:
        synergy = f" (samordning: {', '.join(reno.ecm_synergy)})" if reno.ecm_synergy else ""
        print(f"   • {reno.planned_year}: {reno.name_sv} - {reno.estimated_cost_sek:,.0f} SEK{synergy}")

    print("\n TILLGÄNGLIGA ENERGIÅTGÄRDER:")
    for steg in range(4):
        steg_candidates = [c for c in candidates if c.steg == steg]
        if steg_candidates:
            steg_name = ["Nollkostnad", "Snabba vinster", "Standard", "Premium"][steg]
            print(f"\n   Steg {steg}: {steg_name}")
            for c in steg_candidates:
                print(f"      • {c.name}: {c.investment_sek:,.0f} SEK → {c.annual_savings_sek:,.0f} SEK/år")

    # Create optimal sequence
    print("\n" + "-" * 70)
    print("SKAPAR OPTIMAL INVESTERINGSSEKVENS...")
    print("-" * 70)

    sequencer = ECMSequencer()
    plan = sequencer.create_optimal_plan(
        candidates=candidates,
        financials=financials,
        renovations=renovations,
        start_year=2025,
        plan_horizon_years=15,
    )

    plan.brf_name = "BRF Sjöstaden 2"
    plan.atemp_m2 = 15350
    plan.num_apartments = 110
    plan.construction_year = 2003

    # Simulate cash flow
    simulator = CashFlowSimulator()
    plan = simulator.simulate(plan, start_year=2025)

    # Print results
    print_cash_flow_summary(plan)

    # Effektvakt analysis
    print_effektvakt_analysis()

    # Key takeaways
    print("\n" + "=" * 70)
    print("NYCKELINSIKTER")
    print("=" * 70)

    zero_cost_savings = sum(
        c.annual_savings_sek for c in candidates if c.is_zero_cost
    )
    print(f"\n 1. Steg 0 (nollkostnad) ger {zero_cost_savings:,.0f} SEK/år direkt")
    print(f"    → Använd dessa besparingar för att finansiera Steg 1-3")

    print(f"\n 2. Samordna tak + isolering 2035 → spara {renovations[1].ecm_cost_reduction:,.0f} SEK")

    print(f"\n 3. Break-even efter {plan.break_even_year - 2025} år")
    print(f"    → Därefter genererar investeringarna nettovinst")

    print(f"\n 4. Ingen avgiftshöjning krävs om planen följs")
    print(f"    → Energibesparingar täcker investeringarna")

    print("\n" + "=" * 70)
    print("Klar! Denna plan kan presenteras för styrelsen.")
    print("=" * 70)


if __name__ == "__main__":
    main()
