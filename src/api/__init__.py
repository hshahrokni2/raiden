"""
Raiden REST API.

FastAPI-based REST API for building energy analysis.

Usage:
    uvicorn src.api.main:app --reload

    # Or with the CLI
    python -m src.api.main
"""

from .main import app

__all__ = ["app"]
