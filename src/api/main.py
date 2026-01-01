"""
Raiden REST API - FastAPI Application.

Provides REST endpoints for building energy analysis.

Endpoints:
    GET  /                      - API info and health check
    POST /analyze/address       - Analyze building by address
    POST /analyze/building      - Analyze with full building data
    GET  /ecms                  - List available ECMs
    GET  /report/{report_id}    - Get generated report

Usage:
    uvicorn src.api.main:app --reload --port 8000

    # With Docker
    docker run -p 8000:8000 raiden-api
"""

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.responses import FileResponse, HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Create dummy classes for type hints
    class BaseModel:
        pass

logger = logging.getLogger(__name__)

# Storage for reports (in-memory for demo, use DB in production)
REPORTS_STORAGE: Dict[str, Path] = {}
ANALYSIS_RESULTS: Dict[str, dict] = {}


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class AddressAnalysisRequest(BaseModel):
    """Request to analyze a building by address."""
    address: str = Field(..., description="Swedish street address", example="Aktergatan 11, Stockholm")
    construction_year: Optional[int] = Field(None, description="Year of construction", example=2003)
    num_apartments: Optional[int] = Field(None, description="Number of apartments", example=110)
    atemp_m2: Optional[float] = Field(None, description="Heated area in m²", example=15350)
    current_fund_sek: Optional[float] = Field(None, description="Current maintenance fund (SEK)")
    annual_energy_cost_sek: Optional[float] = Field(None, description="Annual energy cost (SEK)")
    skip_simulation: bool = Field(False, description="Skip EnergyPlus simulation")


class BuildingDataRequest(BaseModel):
    """Full building data for analysis."""
    address: str
    construction_year: int
    building_type: str = "multi_family"
    facade_material: str = "concrete"
    atemp_m2: float
    num_floors: int = 4
    num_apartments: int = 0
    declared_energy_kwh_m2: float = 0
    energy_class: str = "Unknown"
    heating_system: str = "district"
    has_ftx: bool = False
    has_heat_pump: bool = False
    has_solar: bool = False
    current_fund_sek: float = 0
    annual_fund_contribution_sek: float = 0
    annual_energy_cost_sek: float = 0
    peak_el_kw: float = 0
    peak_fv_kw: float = 0


class EffektvaktResult(BaseModel):
    """Effektvakt (peak shaving) analysis result."""
    current_el_peak_kw: float
    current_fv_peak_kw: float
    optimized_el_peak_kw: float
    optimized_fv_peak_kw: float
    el_peak_reduction_kw: float
    fv_peak_reduction_kw: float
    annual_el_savings_sek: float
    annual_fv_savings_sek: float
    total_annual_savings_sek: float
    pre_heat_hours: float
    coast_duration_hours: float
    requires_bms: bool
    notes: List[str]


class MaintenancePlanSummary(BaseModel):
    """Summary of maintenance plan results."""
    total_investment_sek: float
    total_savings_30yr_sek: float
    net_present_value_sek: float
    break_even_year: int
    final_fund_balance_sek: float
    max_loan_used_sek: float
    zero_cost_annual_savings: float


class AnalysisResponse(BaseModel):
    """Response from analysis endpoint."""
    success: bool
    analysis_id: str
    address: str
    processing_time_seconds: float
    building_data: Optional[Dict] = None
    effektvakt: Optional[EffektvaktResult] = None
    maintenance_plan: Optional[MaintenancePlanSummary] = None
    report_url: Optional[str] = None
    error: Optional[str] = None


class ECMInfo(BaseModel):
    """Information about an ECM."""
    id: str
    name: str
    category: str
    description: str
    typical_savings_percent: float
    cost_category: str


class ECMListResponse(BaseModel):
    """List of available ECMs."""
    count: int
    ecms: List[ECMInfo]


# =============================================================================
# VISUAL ANALYSIS MODELS (for Komilion integration)
# =============================================================================

class VisualAnalysisRequest(BaseModel):
    """Request for visual building analysis."""
    lat: float = Field(..., description="Latitude (WGS84)", example=59.3044309)
    lon: float = Field(..., description="Longitude (WGS84)", example=18.0937078)
    include: Optional[List[str]] = Field(
        default=["height", "floors", "material", "footprint", "form", "era", "wwr"],
        description="Data to include in response"
    )
    # NEW: Address-based footprint resolution (v2.0)
    brf_addresses: Optional[List[str]] = Field(
        default=None,
        description="List of street addresses from energy declaration. "
                    "Used to correctly slice shared building complexes. "
                    "Example: ['Filmgatan 1', 'Filmgatan 3', 'Filmgatan 5']"
    )
    city: Optional[str] = Field(
        default=None,
        description="City name for better address resolution",
        example="Solna"
    )


class VisualAnalysisResponse(BaseModel):
    """Response from visual analysis."""
    success: bool
    lat: float
    lon: float
    height_m: Optional[float] = None
    floors: Optional[int] = None
    material: Optional[str] = None
    building_form: Optional[str] = None
    estimated_era: Optional[str] = None
    wwr: Optional[float] = None
    footprint_m2: Optional[float] = None
    footprint_geojson: Optional[Dict] = None
    footprint_source: Optional[str] = None  # NEW: 'osm', 'microsoft', 'satellite'
    footprint_osm_id: Optional[str] = None  # NEW: OSM building ID
    is_multi_building: bool = False         # NEW: True if BRF has multiple buildings
    buildings_count: int = 1                # NEW: Number of buildings
    all_footprints: Optional[List[Dict]] = None  # NEW: All building footprints
    confidence: Optional[float] = None
    cost: float = 0.0
    model: Optional[str] = None
    error: Optional[str] = None
    processing_time_seconds: float = 0.0


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Raiden API",
        description="Swedish Building Energy Analysis API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Add CORS middleware for web frontend access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # =============================================================================
    # ENDPOINTS
    # =============================================================================

    @app.get("/", tags=["General"])
    async def root():
        """API info and health check."""
        return {
            "name": "Raiden API",
            "version": "1.0.0",
            "description": "Swedish Building Energy Analysis",
            "status": "healthy",
            "endpoints": {
                "analyze_address": "POST /analyze/address",
                "analyze_building": "POST /analyze/building",
                "list_ecms": "GET /ecms",
                "get_report": "GET /report/{report_id}",
            },
            "documentation": "/docs",
        }

    @app.post("/analyze/address", response_model=AnalysisResponse, tags=["Analysis"])
    async def analyze_by_address(request: AddressAnalysisRequest):
        """
        Analyze a building by Swedish street address.

        This is the main endpoint implementing Raiden's vision:
        "Given just an address, automatically analyze and generate recommendations."

        The API will:
        1. Geocode the address
        2. Fetch building data from public sources
        3. Generate maintenance plan with cash flow cascade
        4. Analyze effektvakt potential
        5. Generate HTML report

        Returns analysis results including maintenance plan and effektvakt analysis.
        """
        try:
            from ..core.address_pipeline import AddressPipeline

            # Build known data from request
            known_data = {}
            if request.construction_year:
                known_data['construction_year'] = request.construction_year
            if request.num_apartments:
                known_data['num_apartments'] = request.num_apartments
            if request.atemp_m2:
                known_data['atemp_m2'] = request.atemp_m2
            if request.current_fund_sek:
                known_data['current_fund_sek'] = request.current_fund_sek
            if request.annual_energy_cost_sek:
                known_data['annual_energy_cost_sek'] = request.annual_energy_cost_sek

            # Run analysis
            output_dir = Path("./output/api")
            pipeline = AddressPipeline(output_dir=output_dir)

            result = pipeline.analyze(
                address=request.address,
                known_data=known_data if known_data else None,
                skip_simulation=request.skip_simulation,
                generate_report=True,
            )

            # Generate unique analysis ID
            analysis_id = str(uuid.uuid4())[:8]

            # Store report path
            if result.report_path:
                REPORTS_STORAGE[analysis_id] = result.report_path

            # Build response
            response_data = {
                "success": result.success,
                "analysis_id": analysis_id,
                "address": request.address,
                "processing_time_seconds": result.processing_time_seconds,
                "error": result.error if not result.success else None,
            }

            if result.building_data:
                bd = result.building_data
                response_data["building_data"] = {
                    "address": bd.address,
                    "latitude": bd.latitude,
                    "longitude": bd.longitude,
                    "construction_year": bd.construction_year,
                    "building_type": bd.building_type,
                    "facade_material": bd.facade_material,
                    "atemp_m2": bd.atemp_m2,
                    "num_floors": bd.num_floors,
                    "num_apartments": bd.num_apartments,
                    "energy_class": bd.energy_class,
                    "data_sources": bd.data_sources,
                    "confidence_score": bd.confidence_score,
                }

            if result.effektvakt_result:
                eff = result.effektvakt_result
                response_data["effektvakt"] = EffektvaktResult(
                    current_el_peak_kw=eff.current_el_peak_kw,
                    current_fv_peak_kw=eff.current_fv_peak_kw,
                    optimized_el_peak_kw=eff.optimized_el_peak_kw,
                    optimized_fv_peak_kw=eff.optimized_fv_peak_kw,
                    el_peak_reduction_kw=eff.el_peak_reduction_kw,
                    fv_peak_reduction_kw=eff.fv_peak_reduction_kw,
                    annual_el_savings_sek=eff.annual_el_savings_sek,
                    annual_fv_savings_sek=eff.annual_fv_savings_sek,
                    total_annual_savings_sek=eff.total_annual_savings_sek,
                    pre_heat_hours=eff.pre_heat_hours,
                    coast_duration_hours=eff.coast_duration_hours,
                    requires_bms=eff.requires_bms,
                    notes=eff.notes or [],
                )

            if result.maintenance_plan:
                mp = result.maintenance_plan
                response_data["maintenance_plan"] = MaintenancePlanSummary(
                    total_investment_sek=mp.total_investment_sek,
                    total_savings_30yr_sek=mp.total_savings_30yr_sek,
                    net_present_value_sek=mp.net_present_value_sek,
                    break_even_year=mp.break_even_year,
                    final_fund_balance_sek=mp.final_fund_balance_sek,
                    max_loan_used_sek=mp.max_loan_used_sek,
                    zero_cost_annual_savings=sum(
                        inv.annual_savings_sek
                        for inv in mp.ecm_investments
                        if inv.investment_sek < 20000
                    ),
                )

            if result.report_path:
                response_data["report_url"] = f"/report/{analysis_id}"

            # Store full results
            ANALYSIS_RESULTS[analysis_id] = response_data

            return AnalysisResponse(**response_data)

        except Exception as e:
            logger.error(f"Analysis error: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/ecms", response_model=ECMListResponse, tags=["ECMs"])
    async def list_ecms():
        """List all available Energy Conservation Measures (ECMs)."""
        try:
            from ..ecm.catalog import ECMCatalog
            from ..roi.costs_sweden import ECM_COSTS, CostCategory

            catalog = ECMCatalog()
            ecms = []

            for ecm in catalog.all():
                cost_info = ECM_COSTS.get(ecm.id, None)
                cost_category = cost_info.category.value if cost_info else "unknown"

                ecms.append(ECMInfo(
                    id=ecm.id,
                    name=ecm.name,
                    category=ecm.category.value,
                    description=ecm.description or "",
                    typical_savings_percent=ecm.typical_savings_percent if hasattr(ecm, 'typical_savings_percent') else 0,
                    cost_category=cost_category,
                ))

            return ECMListResponse(count=len(ecms), ecms=ecms)

        except Exception as e:
            logger.error(f"ECM list error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/report/{report_id}", tags=["Reports"])
    async def get_report(report_id: str):
        """Get generated HTML report by ID."""
        if report_id not in REPORTS_STORAGE:
            raise HTTPException(status_code=404, detail="Report not found")

        report_path = REPORTS_STORAGE[report_id]

        if not report_path.exists():
            raise HTTPException(status_code=404, detail="Report file not found")

        return FileResponse(
            report_path,
            media_type="text/html",
            filename=f"raiden_report_{report_id}.html"
        )

    @app.get("/analysis/{analysis_id}", tags=["Analysis"])
    async def get_analysis_result(analysis_id: str):
        """Get analysis results by ID."""
        if analysis_id not in ANALYSIS_RESULTS:
            raise HTTPException(status_code=404, detail="Analysis not found")

        return ANALYSIS_RESULTS[analysis_id]

    @app.post("/visual/analyze", response_model=VisualAnalysisResponse, tags=["Visual"])
    async def visual_analyze(request: VisualAnalysisRequest):
        """
        Analyze building visually from coordinates.

        NEW in v2.0: Uses OSM footprints by default (much more accurate).
        Provide brf_addresses to correctly slice shared building complexes.

        Features:
        - Building height (from OSM or geometric triangulation)
        - Floor count (from OSM or LLM-based)
        - Facade material (LLM + CLIP)
        - Building form (lamellhus, skivhus, punkthus, etc.)
        - Estimated construction era
        - Window-to-wall ratio (WWR)
        - Building footprint from OSM (preferred) or satellite

        This endpoint is designed for Komilion integration.
        """
        import time
        start_time = time.time()

        try:
            from ..visual import quick_visual_scan

            # Run visual analysis with optional address-based slicing
            result = quick_visual_scan(
                lat=request.lat, 
                lon=request.lon,
                brf_addresses=request.brf_addresses,
                city=request.city,
            )

            processing_time = time.time() - start_time

            return VisualAnalysisResponse(
                success=True,
                lat=request.lat,
                lon=request.lon,
                height_m=result.get("height_m"),
                floors=result.get("floors"),
                material=result.get("material"),
                building_form=result.get("building_form"),
                estimated_era=result.get("estimated_era"),
                wwr=result.get("wwr"),
                footprint_m2=result.get("footprint_area_m2"),
                footprint_geojson=result.get("footprint_geojson"),
                footprint_source=result.get("footprint_source"),
                footprint_osm_id=result.get("footprint_osm_id"),
                is_multi_building=result.get("is_multi_building", False),
                buildings_count=result.get("buildings_count", 1),
                all_footprints=result.get("all_footprints"),
                confidence=result.get("confidence"),
                cost=0.0,  # Free models
                model="nvidia/nemotron-nano-12b-v2-vl:free",
                processing_time_seconds=processing_time,
            )

        except Exception as e:
            logger.error(f"Visual analysis error: {e}")
            import traceback
            traceback.print_exc()
            return VisualAnalysisResponse(
                success=False,
                lat=request.lat,
                lon=request.lon,
                error=str(e),
                processing_time_seconds=time.time() - start_time,
            )

    @app.get("/health", tags=["General"])
    async def health_check():
        """Health check endpoint for Fly.io."""
        return {"status": "healthy", "version": "1.0.0"}

else:
    # Dummy app if FastAPI not installed
    app = None


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """Run the API server."""
    if not FASTAPI_AVAILABLE:
        print("FastAPI not installed. Run: pip install fastapi uvicorn")
        return 1

    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
