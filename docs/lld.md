# Low-Level Design (LLD) — Fake News Detection System

## 1. API Endpoint Definitions

### 1.1 POST /predict — Single Prediction

**Description**: Classify a single news text as REAL or FAKE.

**Request**:
```json
{
  "text": "string (1–50000 chars, required)"
}
```

**Response (200 OK)**:
```json
{
  "prediction": 0,
  "label": "REAL",
  "confidence": 0.8734,
  "text_length": 142,
  "word_count": 25
}
```

**Error Responses**:
- `422 Unprocessable Entity` — Invalid input (empty text, missing field).
- `503 Service Unavailable` — Model not loaded.
- `500 Internal Server Error` — Unexpected prediction failure.

---

### 1.2 POST /predict/batch — Batch Prediction

**Request**:
```json
{
  "texts": ["article one...", "article two..."]
}
```

**Response (200 OK)**:
```json
{
  "predictions": [
    {"prediction": 0, "label": "REAL", "confidence": 0.92, "text_length": 100, "word_count": 18},
    {"prediction": 1, "label": "FAKE", "confidence": 0.87, "text_length": 85, "word_count": 15}
  ],
  "count": 2
}
```

---

### 1.3 GET /health — Health Check

**Response (200 OK)**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "vectorizer_loaded": true,
  "version": "1.0.0"
}
```

---

### 1.4 GET /ready — Readiness Probe

**Response (200 OK)**: Same as `/health` when model is loaded.
**Response (503)**: When model is not yet loaded.

---

### 1.5 GET /metrics — Prometheus Metrics

**Response (200)**: Prometheus exposition format text containing:
- `prediction_requests_total` (counter)
- `prediction_class_total{class_label="REAL|FAKE"}` (counter)
- `prediction_latency_seconds` (histogram)
- `prediction_errors_total` (counter)
- `input_text_length` (histogram)
- `prediction_confidence` (histogram)
- `data_drift_score` (gauge)
- `active_requests` (gauge)

---

### 1.6 GET /drift — Drift Detection Status

**Response (200 OK)**:
```json
{
  "overall_drift": 0.0234,
  "alert": false,
  "threshold": 0.1,
  "samples_analyzed": 150,
  "features": {
    "text_length": {
      "baseline_mean": 1200.5,
      "current_mean": 1180.3,
      "drift_score": 0.015
    }
  }
}
```

---

### 1.7 GET /pipeline/info — Pipeline Configuration

**Response (200 OK)**:
```json
{
  "model_algorithm": "passive_aggressive",
  "preprocessing": {
    "max_features": 50000,
    "ngram_range": [1, 2],
    "min_df": 2,
    "max_df": 0.95
  },
  "monitoring": {
    "drift_threshold": 0.1,
    "alert_error_rate": 0.05
  }
}
```

---

## 2. Module Specifications

### 2.1 src/data/ingest.py
| Function | Input | Output |
|----------|-------|--------|
| `load_config(path)` | YAML file path | dict |
| `validate_schema(df, cols)` | DataFrame, list[str] | bool |
| `check_missing_values(df, threshold)` | DataFrame, float | DataFrame |
| `load_data(path)` | CSV file path | DataFrame |
| `compute_baseline_statistics(df)` | DataFrame | dict |

### 2.2 src/data/preprocess.py
| Function | Input | Output |
|----------|-------|--------|
| `clean_text(text)` | str | str |
| `preprocess_dataframe(df)` | DataFrame | DataFrame |
| `build_tfidf_vectorizer(texts, ...)` | Series, params | TfidfVectorizer |
| `split_data(df, ...)` | DataFrame, params | (X_train, X_test, y_train, y_test) |
| `save_vectorizer(vec, path)` | TfidfVectorizer, str | None |
| `load_vectorizer(path)` | str | TfidfVectorizer |

### 2.3 src/model/train.py
| Function | Input | Output |
|----------|-------|--------|
| `get_model(algorithm, hyperparams)` | str, dict | sklearn estimator |
| `train_model(model, X, y)` | estimator, matrix, array | estimator |
| `evaluate_model(model, X, y)` | estimator, matrix, array | dict |
| `train_with_mlflow(...)` | model + data + config | dict (metrics + run_id) |
| `save_model(model, path)` | estimator, str | None |
| `load_model(path)` | str | estimator |

### 2.4 src/model/predict.py
| Class | Method | Input | Output |
|-------|--------|-------|--------|
| `FakeNewsPredictor` | `predict(text)` | str | dict |
| `FakeNewsPredictor` | `predict_batch(texts)` | list[str] | list[dict] |

### 2.5 src/monitoring/drift.py
| Class | Method | Input | Output |
|-------|--------|-------|--------|
| `DriftDetector` | `record_prediction(...)` | int, int, str | None |
| `DriftDetector` | `compute_drift_score()` | — | dict |

## 3. Data Flow Detail

```
User Input (text)
    │
    ▼
clean_text()          → lowercase, remove URLs/HTML/special chars
    │
    ▼
TfidfVectorizer.transform()  → sparse matrix (1 × N features)
    │
    ▼
model.predict()       → [0] or [1]
    │
    ▼
decision_function()   → confidence score
    │
    ▼
Return JSON response
```

## 4. File Structure

```
fake-news-detection/
├── configs/                 # YAML configs, Prometheus config
├── data/raw/                # Raw dataset (DVC tracked)
├── data/processed/          # Processed data + baselines
├── docs/                    # Architecture, HLD, LLD, test plan, manual
├── frontend/                # Streamlit app
├── grafana/dashboards/      # Grafana dashboard JSON
├── models/best_model/       # Exported model + vectorizer + metrics
├── notebooks/               # Exploratory notebooks
├── src/api/                 # FastAPI app + schemas
├── src/data/                # Ingestion + preprocessing
├── src/model/               # Training + prediction
├── src/monitoring/          # Prometheus metrics + drift detection
├── src/pipeline/            # Pipeline orchestrator
├── tests/                   # Unit + integration tests
├── docker-compose.yml       # Multi-container orchestration
├── Dockerfile.backend       # Backend container
├── Dockerfile.frontend      # Frontend container
├── dvc.yaml                 # DVC pipeline stages
├── MLproject                # MLflow project definition
├── Makefile                 # Common commands
└── requirements.txt         # Python dependencies
```
