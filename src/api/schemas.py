"""
API Schemas
===========
Pydantic models for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class PredictionRequest(BaseModel):
    """Schema for a single prediction request."""
    text: str = Field(..., min_length=1, max_length=50000, description="News headline or article text")


class PredictionResponse(BaseModel):
    """Schema for a single prediction response."""
    prediction: int = Field(..., description="Numeric prediction (0=Real, 1=Fake)")
    label: str = Field(..., description="Human-readable label (REAL / FAKE)")
    confidence: float = Field(..., description="Model confidence score")
    text_length: int = Field(..., description="Length of cleaned input text")
    word_count: int = Field(..., description="Word count of cleaned input text")
    container_id: str = Field(default="", description="Container ID for load-balance tracking")


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction requests."""
    texts: List[str] = Field(..., min_length=1, max_length=100, description="List of news texts")


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction responses."""
    predictions: List[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    """Schema for health check response."""
    status: str
    model_loaded: bool
    vectorizer_loaded: bool
    version: str


class DriftResponse(BaseModel):
    """Schema for drift detection response."""
    overall_drift: float
    alert: bool
    threshold: float
    samples_analyzed: int
    features: Optional[dict] = None
    message: Optional[str] = None


class PipelineStatusResponse(BaseModel):
    """Schema for pipeline status response."""
    status: str
    stages: dict
    metrics: Optional[dict] = None
    error: Optional[str] = None