# High-Level Design (HLD) — Fake News Detection System

## 1. Introduction

This document describes the high-level design of the Fake News Detection System, an MLOps-powered NLP application that classifies news articles as real or fake.

## 2. System Goals

- Classify news text as REAL or FAKE with ≥90% accuracy.
- Serve predictions with latency < 200ms per request.
- Provide a user-friendly web interface for non-technical users.
- Implement full MLOps lifecycle: versioning, tracking, monitoring, reproducibility.
- Ensure environment parity via Docker containerization.

## 3. Design Paradigm

The project uses a **functional programming** approach for data pipeline stages (pure functions for data transformation) combined with **object-oriented design** for the prediction engine (`FakeNewsPredictor` class) and drift detector (`DriftDetector` class).

## 4. High-Level Components

### 4.1 Data Layer
- **Ingestion**: Reads CSV data, validates schema, checks for missing values.
- **Preprocessing**: Text cleaning (lowercasing, URL/HTML removal, special character stripping), feature engineering (TF-IDF, text length, word count).
- **Versioning**: DVC tracks raw data and processed artifacts.

### 4.2 Model Layer
- **Algorithm**: PassiveAggressiveClassifier (primary), with LogisticRegression and RandomForest as alternatives.
- **Features**: TF-IDF vectors (up to 50,000 features, unigrams + bigrams).
- **Training**: Automated via pipeline with stratified train/test split (80/20).
- **Tracking**: All experiments logged to MLflow (params, metrics, artifacts).

### 4.3 Serving Layer
- **API Server**: FastAPI with Pydantic validation.
- **Endpoints**: RESTful API with OpenAPI documentation auto-generated at `/docs`.
- **Model Loading**: Pickle-based serialization loaded at application startup.

### 4.4 Presentation Layer
- **Web UI**: Streamlit with four pages (Prediction, Pipeline, Monitoring, Manual).
- **Loose Coupling**: Frontend communicates with backend only via HTTP REST calls.

### 4.5 Monitoring Layer
- **Metrics**: Prometheus client library instruments the API (counters, histograms, gauges).
- **Drift Detection**: Statistical comparison of production input distributions vs. training baselines.
- **Visualization**: Grafana dashboards for real-time monitoring.

## 5. Technology Choices

| Component | Technology | Justification |
|-----------|-----------|---------------|
| ML Model | scikit-learn | Lightweight, no GPU needed, fast training |
| API | FastAPI | Async, auto-docs, validation, modern Python |
| Frontend | Streamlit | Rapid prototyping, built-in widgets |
| Experiment Tracking | MLflow | Industry standard, model registry |
| Data Versioning | DVC | Git-compatible, pipeline orchestration |
| Containerization | Docker + Compose | Environment parity, multi-service orchestration |
| Monitoring | Prometheus + Grafana | Standard observability stack |

## 6. Security Considerations

- CORS middleware configured for controlled access.
- Input validation via Pydantic (text length limits, type checking).
- No cloud services used (all on-premise).
- Docker containers run as non-root where possible.

## 7. Scalability

- Stateless API design allows horizontal scaling.
- Docker Compose can be extended to Swarm mode for multi-node deployment.
- TF-IDF + linear model keeps inference fast and memory-efficient.
