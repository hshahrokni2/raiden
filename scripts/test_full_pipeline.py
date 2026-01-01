#!/usr/bin/env python3
"""
Test the full pipeline with BRF Sjöstaden 2.

This runs:
1. Data fusion (GeoJSON, Mapillary, Google Solar)
2. AI facade analysis (LangSAM for WWR, DINOv2 for materials)
3. Calibrated baseline simulation
4. All ECM simulations
5. Snowball package generation
6. Database storage
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Run full pipeline test."""

    from src.analysis.full_pipeline import FullPipelineAnalyzer

    # Load building data
    building_json = Path("/Users/hosseins/Downloads/brf_sjostaden_2_export.json")

    if not building_json.exists():
        logger.error(f"Building JSON not found: {building_json}")
        return

    with open(building_json) as f:
        building_data = json.load(f)

    logger.info("=" * 60)
    logger.info("FULL PIPELINE TEST: BRF SJÖSTADEN 2")
    logger.info("=" * 60)

    # Get API keys from environment
    google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_SOLAR_API_KEY") or os.getenv("BRF_GOOGLE_API_KEY")
    mapillary_token = os.getenv("MAPILLARY_TOKEN") or os.getenv("MAPILLARY_ACCESS_TOKEN")

    if google_api_key:
        logger.info(f"✓ Google API key found")
    else:
        logger.warning("✗ No Google API key - PV analysis will use estimates")

    if mapillary_token:
        logger.info(f"✓ Mapillary token found")
    else:
        logger.warning("✗ No Mapillary token - using era-based WWR estimates")

    # Determine AI backend
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            logger.info("✓ CUDA available - using GPU")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            logger.info("✓ MPS available - using Apple Silicon GPU")
        else:
            device = "cpu"
            logger.info("○ Using CPU for AI inference")
    except ImportError:
        device = "cpu"
        logger.info("○ PyTorch not found, using CPU")

    # Check for LangSAM
    try:
        from lang_sam import LangSAM
        ai_backend = "lang_sam"
        logger.info("✓ LangSAM available - using for WWR detection")
    except ImportError:
        ai_backend = "opencv"
        logger.info("○ LangSAM not available - using OpenCV fallback")

    # Initialize pipeline
    analyzer = FullPipelineAnalyzer(
        google_api_key=google_api_key,
        mapillary_token=mapillary_token,
        weather_dir=Path("tests/fixtures"),
        output_dir=Path("output_full_pipeline"),
        ai_backend=ai_backend,
        ai_device=device,
    )

    logger.info("")
    logger.info("Starting analysis...")
    logger.info("")

    # Run analysis
    result = await analyzer.analyze(
        building_data=building_data,
        run_simulations=True,
    )

    # Print results
    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)

    fusion = result["data_fusion"]
    logger.info(f"Address: {fusion.address}")
    logger.info(f"Area: {fusion.atemp_m2:,.0f} m²")
    logger.info(f"Floors: {fusion.floors}")
    logger.info(f"Construction: {fusion.construction_year}")
    logger.info(f"Declared: {fusion.declared_kwh_m2} kWh/m²")
    logger.info(f"")
    logger.info(f"WWR detected: {fusion.detected_wwr}")
    logger.info(f"Material detected: {fusion.detected_material}")
    logger.info(f"PV potential: {fusion.pv_capacity_kwp:.1f} kWp")
    logger.info(f"PV annual: {fusion.pv_annual_kwh:,.0f} kWh")
    logger.info(f"")
    logger.info(f"Baseline (calibrated): {result['baseline_kwh_m2']:.1f} kWh/m²")
    logger.info(f"ECMs analyzed: {len(result['ecm_results'])}")

    logger.info("")
    logger.info("SNOWBALL PACKAGES:")
    logger.info("-" * 40)

    for pkg in result["snowball_packages"]:
        logger.info(f"")
        logger.info(f"Package {pkg.package_number}: {pkg.package_name} (Year {pkg.recommended_year})")
        logger.info(f"  ECMs: {', '.join(pkg.ecm_ids)}")
        logger.info(f"  Investment: {pkg.total_investment_sek:,.0f} SEK")
        logger.info(f"  Savings: {pkg.savings_percent:.1f}% → {pkg.combined_kwh_m2:.1f} kWh/m²")
        logger.info(f"  Payback: {pkg.simple_payback_years:.1f} years")
        logger.info(f"  Cumulative: {pkg.cumulative_investment_sek:,.0f} SEK, {pkg.cumulative_savings_percent:.1f}%")

    logger.info("")
    logger.info(f"Total time: {result['total_time_seconds']:.1f}s")
    logger.info("")
    logger.info("Done!")

    return result


if __name__ == "__main__":
    result = asyncio.run(main())
