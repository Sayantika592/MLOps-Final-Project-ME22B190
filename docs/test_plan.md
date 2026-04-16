# Test Plan — Fake News Detection System

## 1. Test Strategy

Testing is structured in three tiers: unit tests, integration tests, and end-to-end (E2E) tests. All automated tests are executed via `pytest`.

**Acceptance Criteria**: All unit and integration tests must pass. Model accuracy must exceed 85% on the test set. API response time must be under 200ms for single predictions.

## 2. Test Cases

### 2.1 Unit Tests — Data Preprocessing (`tests/test_preprocess.py`)

| ID | Test Case | Input | Expected Output | Status |
|----|-----------|-------|-----------------|--------|
| UT-01 | Lowercase conversion | "HELLO WORLD" | "hello world" | PASS |
| UT-02 | URL removal | "Visit http://example.com" | "visit" | PASS |
| UT-03 | HTML tag removal | "<p>Hello</p>" | "hello" | PASS |
| UT-04 | Special character removal | "Hello! @World #2024" | "hello world" | PASS |
| UT-05 | Extra whitespace removal | "hello    world" | "hello world" | PASS |
| UT-06 | Empty string handling | "" | "" | PASS |
| UT-07 | None input handling | None | "" | PASS |
| UT-08 | DataFrame preprocessing | Valid DataFrame | DataFrame with content column | PASS |
| UT-09 | Empty row removal | DataFrame with empty text | Rows dropped | PASS |
| UT-10 | TF-IDF vectorizer shape | 4 sample texts | Correct sparse matrix | PASS |
| UT-11 | TF-IDF vocabulary check | Sample texts | Expected words in vocabulary | PASS |

### 2.2 Unit Tests — Model (`tests/test_model.py`)

| ID | Test Case | Input | Expected Output | Status |
|----|-----------|-------|-----------------|--------|
| UT-12 | Create PassiveAggressive | Algorithm name + params | Correct model instance | PASS |
| UT-13 | Unknown algorithm error | "unknown_algo" | ValueError raised | PASS |
| UT-14 | All registry models | Each algorithm name | Non-null model | PASS |
| UT-15 | Train model | Sparse matrix + labels | Model with predict method | PASS |
| UT-16 | Evaluate metrics keys | Trained model + test data | Dict with accuracy, f1, etc. | PASS |
| UT-17 | Metrics value range | Trained model + test data | All metrics in [0, 1] | PASS |
| UT-18 | Predictor returns dict | News text | Dict with prediction, label | PASS |
| UT-19 | Predictor label values | News text | Label in {REAL, FAKE} | PASS |
| UT-20 | Predictor empty text | "" | Label = INVALID | PASS |
| UT-21 | Batch prediction | List of 2 texts | List of 2 results | PASS |

### 2.3 Integration Tests — API (`tests/test_api.py`)

| ID | Test Case | Method | Endpoint | Expected | Status |
|----|-----------|--------|----------|----------|--------|
| IT-01 | Health check structure | GET | /health | 200 + status field | PASS |
| IT-02 | Ready without model | GET | /ready | 503 | PASS |
| IT-03 | Predict empty text | POST | /predict | 422 | PASS |
| IT-04 | Predict missing field | POST | /predict | 422 | PASS |
| IT-05 | Predict without model | POST | /predict | 503 | PASS |
| IT-06 | Prometheus metrics | GET | /metrics | 200 + metrics text | PASS |
| IT-07 | Drift endpoint | GET | /drift | 200 + drift data | PASS |
| IT-08 | Pipeline info | GET | /pipeline/info | 200 | PASS |
| IT-09 | OpenAPI docs | GET | /docs | 200 | PASS |
| IT-10 | OpenAPI schema | GET | /openapi.json | 200 + paths | PASS |

### 2.4 End-to-End Tests (Manual)

| ID | Test Case | Steps | Expected Result |
|----|-----------|-------|-----------------|
| E2E-01 | Full prediction flow | 1. Open UI 2. Enter text 3. Click Analyze | Prediction displayed with confidence |
| E2E-02 | Example buttons | Click "Try Real News" then Analyze | Shows REAL prediction |
| E2E-03 | Pipeline dashboard | Navigate to Pipeline Dashboard | Shows pipeline stages and config |
| E2E-04 | Monitoring page | Navigate to Monitoring | Shows drift status |
| E2E-05 | Docker startup | Run `docker-compose up` | All 5 services start and respond |

## 3. Test Execution

```bash
# Run all tests
make test

# Run with coverage report
make test-coverage

# Run specific test file
pytest tests/test_preprocess.py -v
pytest tests/test_model.py -v
pytest tests/test_api.py -v
```

## 4. Test Report Summary

| Category | Total | Passed | Failed |
|----------|-------|--------|--------|
| Unit Tests — Preprocessing | 11 | 11 | 0 |
| Unit Tests — Model | 10 | 10 | 0 |
| Integration Tests — API | 10 | 10 | 0 |
| E2E Tests (Manual) | 5 | 5 | 0 |
| **Total** | **36** | **36** | **0** |

**Result**: All tests passed. Acceptance criteria met.
