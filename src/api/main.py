"""
FastAPI Application
===================
Main API server for the Fake News Detection system.
Exposes prediction, health, drift, and metrics endpoints.
"""

import os
import time
import logging
import pickle
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from src.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    DriftResponse,
)
from src.model.predict import FakeNewsPredictor
from src.monitoring.metrics import (
    PREDICTION_REQUESTS,
    PREDICTION_CLASS,
    PREDICTION_LATENCY,
    PREDICTION_ERRORS,
    INPUT_TEXT_LENGTH,
    PREDICTION_CONFIDENCE,
    DATA_DRIFT_SCORE,
    ACTIVE_REQUESTS,
    get_metrics,
    get_content_type,
)
from src.monitoring.drift import DriftDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
predictor: FakeNewsPredictor = None
drift_detector: DriftDetector = None
app_config: dict = {}


def load_config() -> dict:
    """Load application configuration."""
    config_path = os.environ.get("CONFIG_PATH", "configs/model_config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def initialize_model(config: dict) -> FakeNewsPredictor:
    """Load model and vectorizer from disk."""
    model_path = config["api"].get("model_path", "models/best_model")

    model_file = os.path.join(model_path, "model.pkl")
    vectorizer_file = os.path.join(model_path, "vectorizer.pkl")

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")
    if not os.path.exists(vectorizer_file):
        raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_file}")

    with open(model_file, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_file, "rb") as f:
        vectorizer = pickle.load(f)

    logger.info("Model and vectorizer loaded from %s", model_path)
    return FakeNewsPredictor(model, vectorizer)


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and config on startup."""
    global predictor, drift_detector, app_config

    app_config = load_config()
    try:
        predictor = initialize_model(app_config)
        logger.info("Model initialized successfully.")
    except FileNotFoundError as e:
        logger.error("Model initialization failed: %s", e)
        predictor = None

    # Initialize drift detector
    baseline_path = "data/processed/baseline_stats.json"
    drift_detector = DriftDetector(baseline_path=baseline_path)

    yield

    logger.info("Shutting down application.")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Fake News Detection API",
    description="MLOps-powered API for detecting fake news articles using NLP.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health & Readiness Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint for orchestration."""
    return HealthResponse(
        status="healthy" if predictor else "unhealthy",
        model_loaded=predictor is not None,
        vectorizer_loaded=predictor is not None and predictor.vectorizer is not None,
        version="1.0.0",
    )


@app.get("/ready", response_model=HealthResponse, tags=["Health"])
async def readiness_check():
    """Readiness probe – returns 503 if model is not loaded."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Service is not ready.")
    return HealthResponse(
        status="ready",
        model_loaded=True,
        vectorizer_loaded=True,
        version="1.0.0",
    )


# ---------------------------------------------------------------------------
# Prediction Endpoints
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Predict whether a news article is fake or real.

    - **text**: The news headline or article text to classify.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    ACTIVE_REQUESTS.inc()
    PREDICTION_REQUESTS.inc()
    start_time = time.time()

    try:
        result = predictor.predict(request.text)

        if result.get("error"):
            PREDICTION_ERRORS.inc()
            raise HTTPException(status_code=400, detail=result["error"])

        # Record metrics
        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)
        PREDICTION_CLASS.labels(class_label=result["label"]).inc()
        INPUT_TEXT_LENGTH.observe(result["text_length"])
        PREDICTION_CONFIDENCE.observe(result["confidence"])

        # Record for drift detection
        drift_detector.record_prediction(
            text_length=result["text_length"],
            word_count=result["word_count"],
            label=result["label"],
        )

        return PredictionResponse(
            prediction=result["prediction"],
            label=result["label"],
            confidence=result["confidence"],
            text_length=result["text_length"],
            word_count=result["word_count"],
        )

    except HTTPException:
        raise
    except Exception as e:
        PREDICTION_ERRORS.inc()
        logger.exception("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        ACTIVE_REQUESTS.dec()


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction endpoint for multiple texts."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    results = []
    for text in request.texts:
        result = predictor.predict(text)
        results.append(PredictionResponse(
            prediction=result["prediction"],
            label=result["label"],
            confidence=result["confidence"],
            text_length=result["text_length"],
            word_count=result["word_count"],
        ))

    return BatchPredictionResponse(predictions=results, count=len(results))


# ---------------------------------------------------------------------------
# Monitoring Endpoints
# ---------------------------------------------------------------------------

@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    # Update drift gauge
    drift_result = drift_detector.compute_drift_score()
    DATA_DRIFT_SCORE.set(drift_result.get("overall_drift", 0.0))

    return Response(
        content=get_metrics(),
        media_type=get_content_type(),
    )


@app.get("/drift", response_model=DriftResponse, tags=["Monitoring"])
async def get_drift_status():
    """Get current data drift status."""
    result = drift_detector.compute_drift_score()
    return DriftResponse(**result)


# ---------------------------------------------------------------------------
# Pipeline Info Endpoint
# ---------------------------------------------------------------------------

@app.get("/pipeline/info", tags=["Pipeline"])
async def pipeline_info():
    """Return information about the ML pipeline configuration."""
    return {
        "model_algorithm": app_config.get("model", {}).get("algorithm", "unknown"),
        "preprocessing": app_config.get("preprocessing", {}),
        "monitoring": app_config.get("monitoring", {}),
        "data_config": app_config.get("data", {}),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
