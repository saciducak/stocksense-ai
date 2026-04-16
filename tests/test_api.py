"""Tests for the FastAPI application."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.schemas import PredictionRequest, SentimentRequest


# ─── API Tests ────────────────────────────────────────────────────────────────

@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


class TestRootEndpoint:
    """Test the root endpoint."""

    def test_root_returns_200(self, client: TestClient) -> None:
        """Root should return 200 with service info."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["service"] == "StockSense AI"
        assert "endpoints" in data

    def test_root_has_version(self, client: TestClient) -> None:
        """Root should include version."""
        response = client.get("/")
        data = response.json()
        assert "version" in data


class TestHealthEndpoint:
    """Test the health check endpoint."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """Health endpoint should return 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_status(self, client: TestClient) -> None:
        """Health status should be 'healthy'."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert "uptime_seconds" in data

    def test_health_models_loaded(self, client: TestClient) -> None:
        """Health should report model loading status."""
        response = client.get("/health")
        data = response.json()
        assert "models_loaded" in data


class TestModelInfoEndpoint:
    """Test the model info endpoint."""

    def test_models_returns_200(self, client: TestClient) -> None:
        """Models endpoint should return 200."""
        response = client.get("/models")
        assert response.status_code == 200

    def test_available_models(self, client: TestClient) -> None:
        """Should list available models."""
        response = client.get("/models")
        data = response.json()
        assert "available_models" in data
        assert "lstm" in data["available_models"]
        assert "transformer" in data["available_models"]


class TestSchemaValidation:
    """Test Pydantic schema validation."""

    def test_prediction_request_defaults(self) -> None:
        """Default prediction request should be valid."""
        req = PredictionRequest()
        assert req.ticker == "AAPL"
        assert req.forecast_days == 5
        assert req.model_type == "ensemble"

    def test_prediction_request_ticker_uppercase(self) -> None:
        """Ticker should be normalized to uppercase."""
        req = PredictionRequest(ticker="aapl")
        assert req.ticker == "AAPL"

    def test_prediction_request_invalid_model(self) -> None:
        """Invalid model type should raise validation error."""
        with pytest.raises(Exception):
            PredictionRequest(model_type="invalid_model")

    def test_sentiment_request_min_texts(self) -> None:
        """Sentiment request requires at least one text."""
        req = SentimentRequest(texts=["test"])
        assert len(req.texts) == 1

    def test_prediction_request_forecast_range(self) -> None:
        """Forecast days should be within valid range."""
        req = PredictionRequest(forecast_days=1)
        assert req.forecast_days == 1

        with pytest.raises(Exception):
            PredictionRequest(forecast_days=0)

        with pytest.raises(Exception):
            PredictionRequest(forecast_days=31)
