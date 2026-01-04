"""AI-powered analysis modules."""

from .wwr_detector import WWRDetector
from .material_classifier import MaterialClassifier
from .facade_analyzer import FacadeAnalyzer
from .facade_analyzer_llm import FacadeAnalyzerLLM, analyze_facade_with_llm

# 2026 Roadmap: Advanced detectors (stubs)
from .wwr_detector_v2 import (
    SOLOv2WWRDetector,
    WindowDetection,
    WindowToWallRatio,
    get_wwr_detector,
)
from .material_ensemble import (
    MaterialEnsembleClassifier,
    EnsemblePrediction,
    FacadeMaterial,
    get_material_classifier,
)

__all__ = [
    # Current implementations
    "WWRDetector",
    "MaterialClassifier",
    "FacadeAnalyzer",
    "FacadeAnalyzerLLM",
    "analyze_facade_with_llm",
    # 2026 Roadmap: SOLOv2 WWR (stub)
    "SOLOv2WWRDetector",
    "WindowDetection",
    "WindowToWallRatio",
    "get_wwr_detector",
    # 2026 Roadmap: Ensemble material classifier (stub)
    "MaterialEnsembleClassifier",
    "EnsemblePrediction",
    "FacadeMaterial",
    "get_material_classifier",
]
