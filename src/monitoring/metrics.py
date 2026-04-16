"""
Monitoring Metrics Module
=========================
Prometheus instrumentation for the Fake News Detection API.
Tracks prediction counts, latency, errors, and data characteristics.
"""

from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST


# ---------------------------------------------------------------------------
# Metric Definitions
# ---------------------------------------------------------------------------

# Application info
APP_INFO = Info("app", "Fake News Detection application info")
APP_INFO.info({
    "version": "1.0.0",
    "model": "passive_aggressive",
    "framework": "fastapi",
})

# Prediction request counter
PREDICTION_REQUESTS = Counter(
    "prediction_requests_total",
    "Total number of prediction requests received",
)

# Prediction result counter (by class)
PREDICTION_CLASS = Counter(
    "prediction_class_total",
    "Total predictions by class label",
    ["class_label"],
)

# Prediction latency histogram
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time spent processing a prediction request",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
)

# Prediction error counter
PREDICTION_ERRORS = Counter(
    "prediction_errors_total",
    "Total number of prediction errors",
)

# Input text length histogram
INPUT_TEXT_LENGTH = Histogram(
    "input_text_length",
    "Length of input text in characters",
    buckets=[50, 100, 250, 500, 1000, 2500, 5000, 10000],
)

# Confidence score histogram
PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence",
    "Model confidence score distribution",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# Data drift gauge
DATA_DRIFT_SCORE = Gauge(
    "data_drift_score",
    "Current data drift score (0 = no drift, 1 = full drift)",
)

# Model version gauge
MODEL_VERSION = Gauge(
    "model_version_loaded",
    "Currently loaded model version",
)

# Active requests gauge
ACTIVE_REQUESTS = Gauge(
    "active_requests",
    "Number of currently active requests",
)


def get_metrics():
    """Generate latest Prometheus metrics in exposition format."""
    return generate_latest()


def get_content_type():
    """Return the Prometheus content type header."""
    return CONTENT_TYPE_LATEST
