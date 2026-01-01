"""
Data models for Raiden database records.

These are Pydantic models for type safety and validation.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum


class OwnerType(str, Enum):
    """Building owner type."""
    PRIVATE = "private"
    BRF = "brf"
    RENTAL = "rental"
    COMMERCIAL = "commercial"


class BuildingType(str, Enum):
    """Building type."""
    MULTI_FAMILY = "multi_family"
    SINGLE_FAMILY = "single_family"
    COMMERCIAL = "commercial"


class DataQuality(str, Enum):
    """Data quality level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class BuildingRecord:
    """Building database record."""

    # Required fields
    address: str

    # Identification
    id: Optional[str] = None
    org_number: Optional[str] = None
    property_designation: Optional[str] = None
    name: Optional[str] = None

    # Physical characteristics
    building_type: str = "multi_family"
    construction_year: Optional[int] = None
    renovation_year: Optional[int] = None
    heated_area_m2: Optional[float] = None
    num_apartments: Optional[int] = None
    num_floors: Optional[int] = None
    facade_material: Optional[str] = None
    roof_type: Optional[str] = None

    # Location
    municipality: Optional[str] = None
    region: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    # Energy data
    energy_class: Optional[str] = None
    declared_energy_kwh_m2: Optional[float] = None
    heating_system: Optional[str] = None
    ventilation_system: Optional[str] = None

    # Constraints
    heritage_listed: bool = False

    # Owner
    owner_type: str = "brf"

    # Metadata
    data_source: Optional[str] = None
    data_quality: str = "medium"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion."""
        data = asdict(self)
        # Remove None values and timestamps (handled by DB)
        return {
            k: v for k, v in data.items()
            if v is not None and k not in ["created_at", "updated_at"]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BuildingRecord":
        """Create from database record."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class BaselineRecord:
    """Baseline simulation record."""

    building_id: str

    # Results
    annual_heating_kwh: Optional[float] = None
    annual_cooling_kwh: Optional[float] = None
    annual_electricity_kwh: Optional[float] = None
    heating_kwh_m2: Optional[float] = None

    # Calibration
    is_calibrated: bool = False
    calibration_gap_percent: Optional[float] = None

    # Settings
    weather_file: Optional[str] = None
    archetype_id: Optional[str] = None
    idf_file_path: Optional[str] = None
    energyplus_version: Optional[str] = None
    simulation_duration_seconds: Optional[float] = None

    # Metadata
    id: Optional[str] = None
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion."""
        data = asdict(self)
        return {
            k: v for k, v in data.items()
            if v is not None and k not in ["id", "created_at"]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaselineRecord":
        """Create from database record."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ECMResultRecord:
    """ECM analysis result record."""

    building_id: str
    ecm_id: str

    # ECM info
    ecm_name: Optional[str] = None
    ecm_category: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

    # Energy results
    annual_heating_kwh: Optional[float] = None
    annual_cooling_kwh: Optional[float] = None
    annual_electricity_kwh: Optional[float] = None
    heating_kwh_m2: Optional[float] = None

    # Savings
    heating_savings_kwh: Optional[float] = None
    heating_savings_percent: Optional[float] = None
    electricity_savings_kwh: Optional[float] = None
    total_savings_kwh: Optional[float] = None

    # Costs (SEK)
    installation_cost: Optional[float] = None
    material_cost: Optional[float] = None
    labor_cost: Optional[float] = None
    total_cost: Optional[float] = None
    annual_maintenance: Optional[float] = None

    # Deductions
    rot_deduction: float = 0
    green_tech_deduction: float = 0
    net_cost: Optional[float] = None

    # Financial metrics
    annual_savings_sek: Optional[float] = None
    simple_payback_years: Optional[float] = None
    npv_20yr: Optional[float] = None
    irr_percent: Optional[float] = None

    # CO2
    annual_co2_reduction_kg: Optional[float] = None

    # Applicability
    is_applicable: bool = True
    constraint_violations: Optional[List[str]] = None

    # Simulation
    simulated: bool = False
    baseline_id: Optional[str] = None
    idf_file_path: Optional[str] = None

    # Metadata
    id: Optional[str] = None
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion."""
        data = asdict(self)
        # Convert constraint_violations list to JSONB-compatible format
        if data.get("constraint_violations"):
            data["constraint_violations"] = data["constraint_violations"]
        return {
            k: v for k, v in data.items()
            if v is not None and k not in ["id", "created_at"]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ECMResultRecord":
        """Create from database record."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PackageRecord:
    """ECM package record."""

    building_id: str
    package_name: str

    # ECMs included
    ecm_ids: List[str] = field(default_factory=list)
    package_type: Optional[str] = None

    # Combined results
    combined_heating_kwh_m2: Optional[float] = None
    combined_savings_percent: Optional[float] = None
    synergy_factor: float = 1.0

    # Costs
    total_cost: Optional[float] = None
    package_discount: float = 0
    net_cost: Optional[float] = None

    # Financial
    annual_savings_sek: Optional[float] = None
    simple_payback_years: Optional[float] = None
    npv_20yr: Optional[float] = None

    # Validation
    is_valid: bool = True
    validation_issues: Optional[List[str]] = None

    # Metadata
    id: Optional[str] = None
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion."""
        data = asdict(self)
        return {
            k: v for k, v in data.items()
            if v is not None and k not in ["id", "created_at"]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PackageRecord":
        """Create from database record."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ReportRecord:
    """Analysis report record."""

    building_id: str

    # Report metadata
    report_type: str = "full"
    language: str = "sv"

    # Content
    html_content: Optional[str] = None
    pdf_url: Optional[str] = None

    # Summary
    baseline_kwh_m2: Optional[float] = None
    best_package: Optional[str] = None
    best_package_savings_percent: Optional[float] = None
    best_package_payback_years: Optional[float] = None

    # Recommendations (stored as JSONB)
    top_ecms: Optional[List[Dict]] = None
    packages: Optional[List[Dict]] = None

    # Metadata
    id: Optional[str] = None
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion."""
        data = asdict(self)
        return {
            k: v for k, v in data.items()
            if v is not None and k not in ["id", "created_at"]
        }


class PortfolioStatus(str, Enum):
    """Portfolio analysis status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class AnalysisTier(str, Enum):
    """Analysis tier for portfolio buildings."""
    SKIP = "skip"
    FAST = "fast"
    STANDARD = "standard"
    DEEP = "deep"


@dataclass
class PortfolioRecord:
    """Portfolio database record."""

    # Required fields
    name: str

    # Identification
    id: Optional[str] = None
    description: Optional[str] = None
    owner_id: Optional[str] = None

    # Settings
    skip_energy_classes: List[str] = field(default_factory=lambda: ["A", "B"])
    standard_workers: int = 50
    deep_workers: int = 10

    # Aggregated metrics
    total_buildings: int = 0
    analyzed_buildings: int = 0
    skipped_buildings: int = 0
    failed_buildings: int = 0

    # Portfolio-level savings
    total_savings_potential_kwh: float = 0.0
    total_investment_sek: float = 0.0
    portfolio_npv_sek: float = 0.0
    portfolio_payback_years: Optional[float] = None

    # Status
    status: str = "pending"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    # Metadata
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion."""
        data = asdict(self)
        return {
            k: v for k, v in data.items()
            if v is not None and k not in ["id", "created_at", "updated_at"]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PortfolioRecord":
        """Create from database record."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PortfolioBuildingRecord:
    """Portfolio building junction record."""

    portfolio_id: str
    building_id: str

    # Analysis result
    tier: str = "standard"
    analysis_status: str = "pending"

    # Building-specific results
    baseline_kwh_m2: Optional[float] = None
    savings_kwh_m2: Optional[float] = None
    savings_percent: Optional[float] = None
    investment_sek: Optional[float] = None
    payback_years: Optional[float] = None
    npv_sek: Optional[float] = None

    # Archetype
    archetype_id: Optional[str] = None
    archetype_confidence: Optional[float] = None

    # QC flags
    needs_qc: bool = False
    qc_triggers: List[str] = field(default_factory=list)
    qc_completed: bool = False
    qc_result: Optional[Dict[str, Any]] = None

    # Recommended ECMs
    recommended_ecms: List[Dict[str, Any]] = field(default_factory=list)

    # Uncertainty
    savings_uncertainty_kwh_m2: Optional[float] = None
    savings_ci_90_lower: Optional[float] = None
    savings_ci_90_upper: Optional[float] = None

    # Priority/ranking
    roi_rank: Optional[int] = None
    payback_rank: Optional[int] = None

    # Processing
    processing_time_sec: Optional[float] = None
    error_message: Optional[str] = None

    # Metadata
    id: Optional[str] = None
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion."""
        data = asdict(self)
        return {
            k: v for k, v in data.items()
            if v is not None and k not in ["id", "created_at"]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PortfolioBuildingRecord":
        """Create from database record."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class QCLogRecord:
    """QC intervention log record."""

    portfolio_building_id: str
    trigger_type: str
    agent_type: str

    # Result
    success: bool = False
    action_taken: Optional[str] = None
    explanation: Optional[str] = None

    # Updates made
    updated_values: Optional[Dict[str, Any]] = None
    new_confidence: Optional[float] = None

    # Flags
    needs_human_review: bool = False
    escalated: bool = False

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    # Metadata
    id: Optional[str] = None
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion."""
        data = asdict(self)
        return {
            k: v for k, v in data.items()
            if v is not None and k not in ["id", "created_at"]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QCLogRecord":
        """Create from database record."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SurrogateModelRecord:
    """Surrogate model database record."""

    archetype_id: str

    # Model info
    model_type: str = "gaussian_process"

    # Training metrics
    n_samples: Optional[int] = None
    train_r2: Optional[float] = None
    test_r2: Optional[float] = None
    train_rmse_kwh_m2: Optional[float] = None
    test_rmse_kwh_m2: Optional[float] = None

    # Overfitting detection
    has_overfitting_warning: bool = False

    # Parameter bounds
    param_bounds: Optional[Dict[str, Any]] = None

    # Model storage
    model_path: Optional[str] = None

    # Metadata
    trained_date: Optional[str] = None
    training_time_sec: Optional[float] = None
    id: Optional[str] = None
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion."""
        data = asdict(self)
        return {
            k: v for k, v in data.items()
            if v is not None and k not in ["id", "created_at"]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SurrogateModelRecord":
        """Create from database record."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
