# 🔍 Fake News Detection System

An end-to-end MLOps-powered NLP application that classifies news articles as **REAL** or **FAKE** using machine learning.

---

## Features

- **NLP Classification** — TF-IDF + PassiveAggressiveClassifier for fast, accurate text classification.
- **Web Interface** — Streamlit UI with prediction, pipeline dashboard, monitoring, and user manual.
- **REST API** — FastAPI backend with OpenAPI docs, health probes, and batch prediction.
- **Experiment Tracking** — MLflow for logging parameters, metrics, and model artifacts.
- **Data Versioning** — DVC pipeline for reproducible data and model workflows.
- **Monitoring** — Prometheus metrics + Grafana dashboards + data drift detection.
- **Containerized** — Docker Compose with 5 services (backend, frontend, MLflow, Prometheus, Grafana).

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Git

### 1. Clone & Setup

```bash
git clone <repository-url>
cd fake-news-detection
pip install -r requirements.txt
```

### 2. Prepare Data

Download the Fake News dataset and place it at `data/raw/news.csv`. The dataset should have columns: `id`, `title`, `author`, `text`, `label` (0=Real, 1=Fake).

```bash
# Example: Kaggle Fake News dataset
# https://www.kaggle.com/c/fake-news/data
# Place train.csv as data/raw/news.csv
```

### 3. Train the Model

```bash
# Without MLflow
make train

# With MLflow tracking
make train-mlflow
```

### 4. Run with Docker

```bash
# Build and start all services
make docker-up

# Access:
# Frontend:   http://localhost:8501
# API Docs:   http://localhost:8000/docs
# MLflow:     http://localhost:5000
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000 (admin/admin)
```

### 5. Run Locally (without Docker)

```bash
# Terminal 1: Start API
make serve

# Terminal 2: Start frontend
API_URL=http://localhost:8000 make serve-frontend

# Terminal 3: Start MLflow
make mlflow-server
```

---

## Project Structure

```
fake-news-detection/
├── configs/                 # Configuration files
├── data/                    # Raw + processed data (DVC tracked)
├── docs/                    # Architecture, HLD, LLD, test plan, manual
├── frontend/                # Streamlit web UI
├── grafana/                 # Grafana dashboard configs
├── models/                  # Trained model artifacts
├── src/
│   ├── api/                 # FastAPI application
│   ├── data/                # Data ingestion & preprocessing
│   ├── model/               # Training & prediction
│   ├── monitoring/          # Prometheus metrics & drift detection
│   └── pipeline/            # ML pipeline orchestrator
├── tests/                   # Unit & integration tests
├── docker-compose.yml       # Multi-container setup
├── dvc.yaml                 # DVC pipeline definition
├── MLproject                # MLflow project definition
├── Makefile                 # Common commands
└── requirements.txt         # Python dependencies
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Classify single news text |
| POST | `/predict/batch` | Classify multiple texts |
| GET | `/health` | Health check |
| GET | `/ready` | Readiness probe |
| GET | `/metrics` | Prometheus metrics |
| GET | `/drift` | Data drift status |
| GET | `/pipeline/info` | Pipeline configuration |
| GET | `/docs` | Interactive API documentation |

---

## DVC Pipeline

```bash
# View the pipeline DAG
dvc dag

# Reproduce the pipeline
dvc repro

# Check metrics
dvc metrics show
```

---

## Testing

```bash
# Run all tests
make test

# Run with coverage
make test-coverage
```

---

## Documentation

- [Architecture Document](docs/architecture.md)
- [High-Level Design](docs/hld.md)
- [Low-Level Design](docs/lld.md)
- [Test Plan](docs/test_plan.md)
- [User Manual](docs/user_manual.md)

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Model | scikit-learn (PassiveAggressiveClassifier) |
| Features | TF-IDF Vectorizer |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Experiment Tracking | MLflow |
| Data Versioning | DVC + Git |
| Monitoring | Prometheus + Grafana |
| Containerization | Docker + Docker Compose |
| Testing | pytest |
