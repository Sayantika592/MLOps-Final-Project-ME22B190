# Architecture Document — Fake News Detection System

## 1. System Overview

The Fake News Detection System is an end-to-end MLOps application that classifies news articles as **REAL** or **FAKE** using Natural Language Processing. The system follows a microservices architecture with loosely coupled frontend and backend components connected exclusively via REST APIs.

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DOCKER COMPOSE NETWORK                          │
│                                                                        │
│  ┌──────────────┐     REST API      ┌──────────────────────────────┐  │
│  │   FRONTEND   │ ◄──────────────► │         BACKEND API          │  │
│  │  (Streamlit) │   HTTP :8000      │        (FastAPI)             │  │
│  │   Port 8501  │                   │                              │  │
│  │              │                   │  ┌────────────────────────┐  │  │
│  │  - Prediction│                   │  │   Prediction Engine    │  │  │
│  │    UI        │                   │  │  ┌──────────────────┐  │  │  │
│  │  - Pipeline  │                   │  │  │  TF-IDF          │  │  │  │
│  │    Dashboard │                   │  │  │  Vectorizer      │  │  │  │
│  │  - Monitoring│                   │  │  └──────────────────┘  │  │  │
│  │  - User      │                   │  │  ┌──────────────────┐  │  │  │
│  │    Manual    │                   │  │  │  Passive         │  │  │  │
│  └──────────────┘                   │  │  │  Aggressive      │  │  │  │
│                                     │  │  │  Classifier      │  │  │  │
│                                     │  │  └──────────────────┘  │  │  │
│                                     │  └────────────────────────┘  │  │
│                                     │                              │  │
│                                     │  Endpoints:                  │  │
│                                     │  POST /predict               │  │
│                                     │  POST /predict/batch         │  │
│                                     │  GET  /health                │  │
│                                     │  GET  /ready                 │  │
│                                     │  GET  /metrics               │  │
│                                     │  GET  /drift                 │  │
│                                     │  GET  /pipeline/info         │  │
│                                     └──────────────────────────────┘  │
│                                              │                        │
│                    ┌─────────────────────────┼──────────────────┐     │
│                    │                         │                  │     │
│                    ▼                         ▼                  ▼     │
│           ┌──────────────┐         ┌──────────────┐   ┌────────────┐ │
│           │   MLflow     │         │  Prometheus  │   │  Grafana   │ │
│           │  Port 5000   │         │  Port 9090   │   │  Port 3000 │ │
│           │              │         │              │   │            │ │
│           │ - Experiment │         │ - Scrapes    │   │ - Dashbord │ │
│           │   Tracking   │         │   /metrics   │   │   Visuals  │ │
│           │ - Model      │         │ - Stores     │   │ - Alerts   │ │
│           │   Registry   │         │   time-series│   │            │ │
│           │ - Artifacts  │         │              │   │            │ │
│           └──────────────┘         └──────────────┘   └────────────┘ │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘

                    ML PIPELINE (DVC Orchestrated)
┌───────────────────────────────────────────────────────────────────────┐
│                                                                       │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐          │
│  │  Data    │──►│  Text    │──►│  TF-IDF  │──►│  Model   │          │
│  │ Ingestion│   │ Cleaning │   │ Feature  │   │ Training │          │
│  │          │   │          │   │ Eng.     │   │          │          │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘          │
│       │                                             │                │
│       ▼                                             ▼                │
│  ┌──────────┐                                 ┌──────────┐          │
│  │ Baseline │                                 │  Model   │          │
│  │  Stats   │                                 │ Export   │          │
│  │ (Drift)  │                                 │ (.pkl)   │          │
│  └──────────┘                                 └──────────┘          │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

## 3. Component Descriptions

### 3.1 Frontend (Streamlit)
- **Purpose**: User-facing web interface for interacting with the system.
- **Technology**: Streamlit (Python)
- **Pages**: Prediction UI, Pipeline Dashboard, Monitoring, User Manual
- **Communication**: Connects to backend exclusively via REST API calls (loose coupling).

### 3.2 Backend API (FastAPI)
- **Purpose**: Hosts the ML model and serves predictions via REST endpoints.
- **Technology**: FastAPI with Uvicorn ASGI server.
- **Key Endpoints**:
  - `POST /predict` — Single text classification
  - `POST /predict/batch` — Batch classification
  - `GET /health` and `GET /ready` — Health and readiness probes for orchestration
  - `GET /metrics` — Prometheus-compatible metrics endpoint
  - `GET /drift` — Data drift detection status

### 3.3 ML Pipeline (DVC)
- **Purpose**: Reproducible, version-controlled training pipeline.
- **Stages**: Data Ingestion → Preprocessing → Feature Engineering → Training → Export
- **Orchestration**: DVC pipeline with Git for code versioning.

### 3.4 MLflow Tracking Server
- **Purpose**: Experiment tracking, model registry, artifact storage.
- **Tracks**: Hyperparameters, metrics (accuracy, F1, precision, recall), model artifacts.

### 3.5 Prometheus
- **Purpose**: Time-series metrics collection.
- **Scrapes**: Backend `/metrics` endpoint every 15 seconds.
- **Metrics**: Prediction count, latency, error rates, drift scores.

### 3.6 Grafana
- **Purpose**: Real-time visualization of monitoring data.
- **Dashboards**: Prediction volume, latency percentiles, error rates, drift scores.

## 4. Data Flow

1. User enters news text in the Streamlit frontend.
2. Frontend sends HTTP POST request to `backend:8000/predict`.
3. Backend cleans and vectorizes the text using the saved TF-IDF vectorizer.
4. The PassiveAggressiveClassifier predicts the label.
5. Prometheus metrics are updated (latency, class count, drift).
6. Response (label, confidence) is returned to the frontend.
7. Prometheus scrapes metrics; Grafana visualizes them.

## 5. Design Decisions

| Decision | Rationale |
|----------|-----------|
| PassiveAggressiveClassifier | Fast, lightweight, good for text classification. Suitable for on-prem deployment. |
| TF-IDF over deep learning | No GPU required; trains in seconds; competitive accuracy on this task. |
| FastAPI | Async support, automatic OpenAPI docs, built-in validation. |
| Streamlit | Rapid UI development; rich widgets; Python-native. |
| Docker Compose | Multi-container orchestration without Kubernetes complexity. |
| DVC | Data and model versioning alongside Git. |
