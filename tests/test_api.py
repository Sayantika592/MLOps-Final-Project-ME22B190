"""
Integration Tests — FastAPI Endpoints
=======================================
Tests for the REST API endpoints including health, prediction, and monitoring.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


class TestHealthEndpoints:
    """Tests for health and readiness probes."""

    def test_health_endpoint_structure(self):
        """Test that /health returns expected fields."""
        from src.api.main import app
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data

    def test_ready_without_model(self):
        """Test that /ready returns 503 when model is not loaded."""
        from src.api import main as api_main
        original = api_main.predictor
        api_main.predictor = None

        client = TestClient(api_main.app)
        response = client.get("/ready")
        assert response.status_code == 503

        api_main.predictor = original


class TestPredictionEndpoints:
    """Tests for prediction endpoints."""

    def test_predict_empty_text(self):
        """Test that empty text is rejected."""
        from src.api.main import app
        client = TestClient(app)
        response = client.post("/predict", json={"text": ""})
        # Should fail validation (min_length=1)
        assert response.status_code == 422

    def test_predict_missing_field(self):
        """Test that missing text field is rejected."""
        from src.api.main import app
        client = TestClient(app)
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_valid_input_without_model(self):
        """Test prediction when model is not loaded returns 503."""
        from src.api import main as api_main
        original = api_main.predictor
        api_main.predictor = None

        client = TestClient(api_main.app)
        response = client.post("/predict", json={"text": "Some news article"})
        assert response.status_code == 503

        api_main.predictor = original


class TestMonitoringEndpoints:
    """Tests for monitoring endpoints."""

    def test_metrics_endpoint(self):
        """Test that /metrics returns Prometheus format."""
        from src.api.main import app
        from src.api import main as api_main
        
        original_detector = api_main.drift_detector
        api_main.drift_detector = MagicMock()
        api_main.drift_detector.compute_drift_score.return_value = {"overall_drift": 0.0}
        
        client = TestClient(app)
        response = client.get("/metrics")
        
        api_main.drift_detector = original_detector
        assert response.status_code == 200
        assert "prediction_requests_total" in response.text

    def test_drift_endpoint(self):
        """Test that /drift returns drift information."""
        from src.api.main import app
        from src.api import main as api_main
        
        original_detector = api_main.drift_detector
        api_main.drift_detector = MagicMock()
        api_main.drift_detector.compute_drift_score.return_value = {
            "overall_drift": 0.1, 
            "alert": False,
            "features_drift": {},
            "threshold": 0.1,
            "samples_analyzed": 100
        }
        
        client = TestClient(app)
        response = client.get("/drift")
        
        api_main.drift_detector = original_detector
        assert response.status_code == 200
        data = response.json()
        assert "overall_drift" in data
        assert "alert" in data

    def test_pipeline_info_endpoint(self):
        """Test that /pipeline/info returns config details."""
        from src.api.main import app
        client = TestClient(app)
        response = client.get("/pipeline/info")
        assert response.status_code == 200


class TestAPIDocumentation:
    """Tests for API documentation availability."""

    def test_openapi_docs(self):
        """Test that OpenAPI docs are accessible."""
        from src.api.main import app
        client = TestClient(app)
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_schema(self):
        """Test that OpenAPI JSON schema is available."""
        from src.api.main import app
        client = TestClient(app)
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "paths" in schema
        assert "/predict" in schema["paths"]
