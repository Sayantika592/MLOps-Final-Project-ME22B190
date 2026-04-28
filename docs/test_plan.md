# Test Plan — Fake News Detection System

## 1. Test Strategy

Testing is structured into:
- Unit Tests (Preprocessing, Model)
- Integration Tests (API Endpoints)

All tests are executed using `pytest`.

### Acceptance Criteria
- All automated tests must pass  
- API endpoints must return correct status codes and structure  
- Model functions must return valid outputs  

---

## 2. Test Cases

### 2.1 Unit Tests — Data Preprocessing (`tests/test_preprocess.py`)

| ID | Test Case | Input | Expected Output | Status |
|----|-----------|-------|-----------------|--------|
| UT-01 | Lowercase conversion | "HELLO WORLD" | "hello world" | PASS |
| UT-02 | URL removal | Text with URLs | URLs removed | PASS |
| UT-03 | HTML tag removal | "<p>Hello</p>" | "hello" | PASS |
| UT-04 | Special character removal | "Hello! @World #2024" | "hello world" | PASS |
| UT-05 | Extra whitespace removal | "hello    world" | "hello world" | PASS |
| UT-06 | Empty string handling | "" | "" | PASS |
| UT-07 | None input handling | None | "" | PASS |
| UT-08 | Numeric input handling | 123 | "" | PASS |
| UT-09 | DataFrame preprocessing | Valid DataFrame | New features added | PASS |
| UT-10 | Empty row handling | DataFrame with empty text | Rows handled/dropped | PASS |
| UT-11 | Feature engineering | Sample text | word_count, length > 0 | PASS |
| UT-12 | TF-IDF shape | Sample texts | Correct matrix shape | PASS |
| UT-13 | TF-IDF vocabulary | Sample texts | Words in vocabulary | PASS |

---

### 2.2 Unit Tests — Model (`tests/test_model.py`)

| ID | Test Case | Input | Expected Output | Status |
|----|-----------|-------|-----------------|--------|
| UT-14 | Create PassiveAggressive model | Config | Model instance | PASS |
| UT-15 | Unknown algorithm error | Invalid name | ValueError | PASS |
| UT-16 | Registry models | All model names | Valid models | PASS |
| UT-17 | Train model | Sparse matrix | Model with predict() | PASS |
| UT-18 | Evaluate model metrics | Test data | accuracy, precision, recall, f1 | PASS |
| UT-19 | Metrics type validation | Metrics dict | Correct data types | PASS |
| UT-20 | Predictor returns dict | Text input | Dict output | PASS |
| UT-21 | Predictor label values | Text input | REAL / FAKE | PASS |
| UT-22 | Predictor empty text | "" | INVALID label | PASS |
| UT-23 | Batch prediction | List of texts | List output | PASS |

---

### 2.3 Integration Tests — API (`tests/test_api.py`)

| ID | Test Case | Method | Endpoint | Expected Output | Status |
|----|-----------|--------|----------|-----------------|--------|
| IT-01 | Health endpoint | GET | /health | 200 + fields | PASS |
| IT-02 | Ready without model | GET | /ready | 503 | PASS |
| IT-03 | Predict empty text | POST | /predict | 422 | PASS |
| IT-04 | Predict missing field | POST | /predict | 422 | PASS |
| IT-05 | Predict without model | POST | /predict | 503 | PASS |
| IT-06 | Metrics endpoint | GET | /metrics | 200 + metrics | PASS |
| IT-07 | Drift endpoint | GET | /drift | 200 + drift data | PASS |
| IT-08 | Pipeline info | GET | /pipeline/info | 200 | PASS |
| IT-09 | OpenAPI docs | GET | /docs | 200 | PASS |
| IT-10 | OpenAPI schema | GET | /openapi.json | 200 + paths | PASS |

---

## 3. Test Execution

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest -v

# Run individual files
pytest tests/test_preprocess.py -v
pytest tests/test_model.py -v
pytest tests/test_api.py -v

---

## 4. Test Results (Actual Execution)

- Total tests collected: **33**  
- Passed: **33**  
- Failed: **0**  
- Warnings: **8 (sklearn deprecation warnings)**  
- Execution time: **~3.59 seconds**

---

## 5. Test Report Summary

| Category | Total | Passed | Failed |
|----------|-------|--------|--------|
| Unit Tests — Preprocessing | 13 | 13 | 0 |
| Unit Tests — Model | 10 | 10 | 0 |
| Integration Tests — API | 10 | 10 | 0 |
| **Total** | **33** | **33** | **0** |

---

## 6. Notes

- All tests passed successfully, confirming system correctness  
- No functional failures observed  
- Warnings are due to deprecation of `PassiveAggressiveClassifier` in future sklearn versions  

---

## Final Result

All test cases passed successfully. The system is stable, validated, and meets functional requirements.
