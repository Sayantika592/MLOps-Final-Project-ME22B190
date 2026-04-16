.PHONY: setup train serve test docker-up docker-down clean lint

# ============================================================
# Setup
# ============================================================
setup:
	pip install -r requirements.txt
	python -m nltk.downloader stopwords punkt

# ============================================================
# Training
# ============================================================
train:
	python src/pipeline/run_pipeline.py --config configs/model_config.yaml

train-mlflow:
	python src/pipeline/run_pipeline.py --config configs/model_config.yaml --mlflow

# ============================================================
# DVC Pipeline
# ============================================================
dvc-run:
	dvc repro

dvc-dag:
	dvc dag

# ============================================================
# Serving
# ============================================================
serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

serve-frontend:
	cd frontend && streamlit run app.py --server.port 8501

mlflow-server:
	mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db

# ============================================================
# Docker
# ============================================================
docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# ============================================================
# Testing
# ============================================================
test:
	pytest tests/ -v --tb=short

test-coverage:
	pytest tests/ -v --cov=src --cov-report=html --tb=short

# ============================================================
# Code Quality
# ============================================================
lint:
	flake8 src/ tests/ --max-line-length=120 --ignore=E501

# ============================================================
# Cleanup
# ============================================================
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage
