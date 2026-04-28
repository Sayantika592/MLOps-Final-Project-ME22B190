"""
Monitoring Metrics Module
=========================
Prometheus instrumentation for the Fake News Detection API.
Tracks prediction counts, latency, errors, data characteristics.

Metric Types Used:
- Counter: Monotonically increasing values (total requests, errors)
- Gauge: Values that go up and down (drift score, active requests)
- Histogram: Distribution of values in buckets (latency, text length)
- Summary: Quantile-based metrics with _sum and _total (inference time)
- Custom Labels: source_ip, session_id, model_version on relevant metrics
"""

from prometheus_client import (
    Counter, Histogram, Gauge, Info, Summary,
    generate_latest, CONTENT_TYPE_LATEST,
)


# Application Info

APP_INFO = Info("app", "Fake News Detection application info")
APP_INFO.info({
    "version": "1.0.0",
    "model": "passive_aggressive",
    "framework": "fastapi",
})

# COUNTER Metrics

# Total prediction requests (with custom label: model_version)
PREDICTION_REQUESTS = Counter(
    "prediction_requests_total",
    "Total number of prediction requests received",
    ["model_version"],
)

# Prediction result counter (with custom label: class_label)
PREDICTION_CLASS = Counter(
    "prediction_class_total",
    "Total predictions by class label",
    ["class_label"],
)

# Prediction error counter
PREDICTION_ERRORS = Counter(
    "prediction_errors_total",
    "Total number of prediction errors",
    ["error_type"],
)

# Requests by source IP (custom label for client tracking)
REQUESTS_BY_SOURCE = Counter(
    "requests_by_source_total",
    "Requests grouped by client source",
    ["source_ip"],
)

# GAUGE Metrics

# Data drift gauge
DATA_DRIFT_SCORE = Gauge(
    "data_drift_score",
    "Current data drift score (0 = no drift, 1 = full drift)",
)

# Active requests gauge
ACTIVE_REQUESTS = Gauge(
    "active_requests",
    "Number of currently active requests",
)

# Model load status
MODEL_LOADED = Gauge(
    "model_loaded",
    "Whether the ML model is currently loaded (1=yes, 0=no)",
)

# Memory usage estimate gauge
MODEL_MEMORY_MB = Gauge(
    "model_memory_mb",
    "Estimated memory usage of the loaded model in MB",
)

# HISTOGRAM Metrics

# Prediction latency histogram
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time spent processing a prediction request",
    ["endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
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

# SUMMARY Metrics (generates _sum and _total suffixes)

INFERENCE_TIME_SUMMARY = Summary(
    "inference_time_seconds",
    "Summary of model inference time (provides _sum and _count)",
)

TEXT_PROCESSING_TIME = Summary(
    "text_processing_seconds",
    "Summary of text preprocessing time (provides _sum and _count)",
)

# Helpers

def get_metrics():
    """Generate latest Prometheus metrics in exposition format."""
    return generate_latest()


def get_content_type():
    """Return the Prometheus content type header."""
    return CONTENT_TYPE_LATEST
