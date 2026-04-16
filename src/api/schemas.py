"""Pydantic v2 schemas for API request/response validation.

Provides strict type validation, serialization, and automatic
Swagger documentation for all API endpoints.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ─── Request Schemas ──────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    """Request body for stock price prediction.

    Example:
        {
            "ticker": "AAPL",
            "forecast_days": 5,
            "include_sentiment": true
        }
    """

    ticker: str = Field(
        default="AAPL",
        description="Stock ticker symbol (e.g., AAPL, GOOGL, MSFT)",
        min_length=1,
        max_length=10,
    )
    forecast_days: int = Field(
        default=5,
        description="Number of days to forecast",
        ge=1,
        le=30,
    )
    include_sentiment: bool = Field(
        default=True,
        description="Include FinBERT sentiment analysis in prediction",
    )
    model_type: str = Field(
        default="ensemble",
        description="Model to use: 'lstm', 'transformer', or 'ensemble'",
    )

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        """Normalize ticker to uppercase."""
        return v.upper().strip()

    @field_validator("model_type")
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        """Validate model type selection."""
        valid = {"lstm", "transformer", "ensemble"}
        if v.lower() not in valid:
            raise ValueError(f"model_type must be one of {valid}")
        return v.lower()


class SentimentRequest(BaseModel):
    """Request body for sentiment analysis.

    Example:
        {
            "texts": ["Apple reports record earnings", "Stock market crashes"],
            "include_entities": true
        }
    """

    texts: list[str] = Field(
        description="List of financial texts to analyze",
        min_length=1,
        max_length=50,
    )
    include_entities: bool = Field(
        default=False,
        description="Also extract named entities from texts",
    )


class ReportRequest(BaseModel):
    """Request body for LLM Executive Report.

    Example:
        {
            "ticker": "AAPL",
            "forecast_days": 5
        }
    """

    ticker: str = Field(
        default="AAPL",
        description="Stock ticker symbol",
        min_length=1,
        max_length=10,
    )
    forecast_days: int = Field(
        default=5,
        description="Number of days to forecast",
        ge=1,
        le=30,
    )

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        """Normalize ticker to uppercase."""
        return v.upper().strip()


# ─── Response Schemas ─────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """Response for stock price prediction endpoint.

    Example:
        {
            "ticker": "AAPL",
            "predictions": [178.5, 179.2, 180.1, 179.8, 181.3],
            "confidence": 0.87,
            "model_used": "ensemble",
            "sentiment_score": 0.42,
            "timestamp": "2026-04-16T12:00:00"
        }
    """

    ticker: str
    predictions: list[float] = Field(description="Predicted prices for each forecast day")
    confidence: float = Field(description="Model confidence (0-1)")
    model_used: str = Field(description="Model variant used for prediction")
    sentiment_score: Optional[float] = Field(
        default=None,
        description="Aggregate sentiment score (-1 to 1)",
    )
    forecast_days: int = Field(description="Number of forecast days")
    timestamp: datetime = Field(default_factory=datetime.now)


class SentimentItem(BaseModel):
    """Sentiment analysis result for a single text."""

    text: str
    label: str = Field(description="Sentiment: positive, negative, or neutral")
    score: float = Field(description="Confidence of predicted label (0-1)")
    numeric_score: float = Field(description="Numeric score: -1 (neg) to 1 (pos)")


class EntityItem(BaseModel):
    """Named entity extracted from text."""

    text: str = Field(description="Entity text")
    label: str = Field(description="Entity type (ORG, PER, LOC, etc.)")
    score: float = Field(description="Detection confidence")


class SentimentResponse(BaseModel):
    """Response for sentiment analysis endpoint."""

    results: list[SentimentItem]
    mean_sentiment: float = Field(description="Average sentiment score")
    distribution: dict[str, int] = Field(description="Count of each label")
    entities: Optional[list[EntityItem]] = Field(
        default=None,
        description="Extracted entities (if requested)",
    )


class ReportResponse(BaseModel):
    """Response for LLM Executive Report."""

    ticker: str
    report_markdown: str = Field(description="The RAG-generated executive summary")
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthResponse(BaseModel):
    """Response for health check endpoint."""

    status: str = Field(default="healthy")
    version: str
    models_loaded: dict[str, bool]
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    """Response for model information endpoint."""

    available_models: list[str]
    active_model: str
    model_configs: dict
    optimization_status: dict


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str
    status_code: int
    timestamp: datetime = Field(default_factory=datetime.now)
