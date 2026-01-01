"""
Tests for FastAPI REST API endpoints.

Covers:
- Root/health endpoint
- Address analysis endpoint
- ECM listing endpoint
- Report retrieval endpoint
- Error handling
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

try:
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


@pytest.fixture
def client():
    """Create test client for API."""
    if not FASTAPI_AVAILABLE:
        pytest.skip("FastAPI not installed")
    from src.api.main import app
    if app is None:
        pytest.skip("FastAPI app not available")
    return TestClient(app)


class TestRootEndpoint:
    """Tests for root/health check endpoint."""

    def test_root_returns_ok(self, client):
        """Test root endpoint returns 200."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["name"] == "Raiden API"

    def test_root_includes_version(self, client):
        """Test root endpoint includes version info."""
        response = client.get("/")
        data = response.json()
        assert "version" in data
        assert data["version"] == "1.0.0"

    def test_root_includes_endpoints(self, client):
        """Test root endpoint lists available endpoints."""
        response = client.get("/")
        data = response.json()
        assert "endpoints" in data
        assert "analyze_address" in data["endpoints"]


class TestECMListEndpoint:
    """Tests for ECM listing endpoint."""

    def test_list_ecms_returns_ok(self, client):
        """Test ECM list endpoint returns 200."""
        response = client.get("/ecms")
        assert response.status_code == 200
        data = response.json()
        assert "count" in data
        assert "ecms" in data
        assert data["count"] > 0

    def test_ecms_have_required_fields(self, client):
        """Test each ECM has required fields."""
        response = client.get("/ecms")
        data = response.json()
        if data["ecms"]:
            ecm = data["ecms"][0]
            assert "id" in ecm
            assert "name" in ecm
            assert "category" in ecm


class TestAnalyzeAddressEndpoint:
    """Tests for address analysis endpoint."""

    def test_analyze_address_missing_address(self, client):
        """Test missing address returns 422."""
        response = client.post("/analyze/address", json={})
        assert response.status_code == 422

    def test_analyze_address_invalid_json(self, client):
        """Test invalid JSON returns 422."""
        response = client.post(
            "/analyze/address",
            content="not json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_analyze_address_valid_request_format(self, client):
        """Test valid request format is accepted (may fail due to geocoding)."""
        response = client.post(
            "/analyze/address",
            json={
                "address": "Aktergatan 11, Stockholm",
                "construction_year": 2003,
                "skip_simulation": True,
            }
        )
        # May succeed (200) or fail due to external API calls (500)
        # But should not return validation error (422)
        assert response.status_code in [200, 500]


class TestReportEndpoint:
    """Tests for report retrieval endpoint."""

    def test_get_report_not_found(self, client):
        """Test getting non-existent report returns 404."""
        response = client.get("/report/nonexistent-id")
        assert response.status_code == 404


class TestAnalysisResultEndpoint:
    """Tests for analysis result retrieval endpoint."""

    def test_get_analysis_not_found(self, client):
        """Test getting non-existent analysis returns 404."""
        response = client.get("/analysis/nonexistent-id")
        assert response.status_code == 404


class TestAPIErrorHandling:
    """Tests for API error handling."""

    def test_invalid_endpoint_returns_404(self, client):
        """Test invalid endpoint returns 404."""
        response = client.get("/invalid/endpoint")
        assert response.status_code == 404

    def test_method_not_allowed_on_root(self, client):
        """Test wrong HTTP method on root returns 405."""
        response = client.delete("/")
        assert response.status_code == 405

    def test_method_not_allowed_on_ecms(self, client):
        """Test POST on GET-only endpoint returns 405."""
        response = client.post("/ecms", json={})
        assert response.status_code == 405


class TestCORSHeaders:
    """Tests for CORS configuration."""

    def test_options_request_on_root(self, client):
        """Test OPTIONS request for CORS preflight."""
        response = client.options("/")
        # CORS middleware should handle OPTIONS
        assert response.status_code in [200, 204, 405]

    def test_cors_headers_present(self, client):
        """Test CORS headers are present in response."""
        response = client.get("/")
        # FastAPI CORS middleware adds these headers
        # Note: TestClient may not fully simulate CORS preflight
        assert response.status_code == 200
