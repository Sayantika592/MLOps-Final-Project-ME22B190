.PHONY: setup train serve test docker-up docker-down clean lint airflow-init

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

dvc-metrics:
	dvc metrics show

# ============================================================
# Serving
# ============================================================
serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

serve-frontend:
	cd frontend && API_URL=http://localhost:8000 streamlit run app.py --server.port 8501

mlflow-server:
	mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db

# MLflow Model Serving (A7 requirement)
mlflow-serve:
	mlflow models serve -m "models/best_model" --port 5001 --no-conda

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
# Airflow (A6 requirement)
# ============================================================
setup-airflow:
	pip install "apache-airflow==2.8.1" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.8.1/constraints-3.11.txt"

airflow-init: setup-airflow
	airflow db init
	airflow users create --username admin --password admin \
		--firstname Admin --lastname User --role Admin --email admin@example.com
	airflow pools set ml_pipeline_pool 3 "Pool for ML pipeline tasks"

airflow-web:
	airflow webserver --port 8080

airflow-scheduler:
	airflow scheduler

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