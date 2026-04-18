"""
Airflow DAG: Fake News Detection ML Pipeline
=============================================
Orchestrates the end-to-end ML pipeline using Apache Airflow.

Features (per A6 requirements):
- FileSensor: Watches for new dataset files in data/raw/
- Worker Pool: Limits concurrency for CPU-heavy tasks (pool size 3)
- Task Dependencies: Explicit >> operator chaining
- Retry Logic: Exponential backoff on failure
- Email Alerts: Notifications on pipeline success/failure/dry pipe
"""

import os
import json
import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from airflow.operators.email import EmailOperator
from airflow.utils.trigger_rule import TriggerRule

# ---------------------------------------------------------------------------
# Default DAG arguments with retry logic (A6 requirement)
# ---------------------------------------------------------------------------

default_args = {
    "owner": "fake-news-team",
    "depends_on_past": False,
    "email": ["team@example.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=2),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=15),
    "start_date": datetime(2026, 4, 1),
}

# ---------------------------------------------------------------------------
# Task Functions
# ---------------------------------------------------------------------------

def task_data_ingestion(**kwargs):
    """Stage 1: Load and validate raw CSV data."""
    import pandas as pd
    import yaml

    params = yaml.safe_load(open("params.yaml"))
    raw_path = params["data"]["raw_path"]

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Data file not found: {raw_path}")

    df = pd.read_csv(raw_path)
    logging.info("Loaded %d rows from %s", len(df), raw_path)

    # Validate schema
    required = ["text", "label"]
    for col in required:
        if col not in df.columns and col == "text" and "title" in df.columns:
            continue
        elif col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Fill NaN in text columns
    for col in ["title", "text", "author"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/ingested.csv", index=False)
    logging.info("Ingestion complete. Saved %d rows.", len(df))

    # Push row count to XCom for downstream tasks
    kwargs["ti"].xcom_push(key="row_count", value=len(df))
    return len(df)


def task_compute_baselines(**kwargs):
    """Stage 2: Compute baseline statistics for drift detection."""
    import pandas as pd
    import numpy as np

    df = pd.read_csv("data/processed/ingested.csv")
    stats = {}

    if "text" in df.columns:
        lengths = df["text"].str.len()
        stats["text_length"] = {
            "mean": float(lengths.mean()),
            "std": float(lengths.std()),
            "median": float(lengths.median()),
            "min": float(lengths.min()),
            "max": float(lengths.max()),
        }

    if "label" in df.columns:
        dist = df["label"].value_counts(normalize=True).to_dict()
        stats["label_distribution"] = {str(k): float(v) for k, v in dist.items()}

    with open("data/processed/baseline_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logging.info("Baseline stats computed: %s", list(stats.keys()))
    return stats


def task_preprocessing(**kwargs):
    """Stage 3: Clean text and engineer features."""
    import sys
    sys.path.insert(0, os.getcwd())
    import pandas as pd
    from src.data.preprocess import preprocess_dataframe

    df = pd.read_csv("data/processed/ingested.csv")
    df = preprocess_dataframe(df)
    df.to_csv("data/processed/preprocessed.csv", index=False)

    logging.info("Preprocessing complete. %d rows remaining.", len(df))
    kwargs["ti"].xcom_push(key="preprocessed_rows", value=len(df))
    return len(df)


def task_train_model(**kwargs):
    """Stage 4: Train model, evaluate, and save artifacts."""
    import sys
    sys.path.insert(0, os.getcwd())
    import yaml
    import pandas as pd
    from src.data.preprocess import build_tfidf_vectorizer, split_data, save_vectorizer
    from src.model.train import get_model, train_model, evaluate_model, save_model

    params = yaml.safe_load(open("params.yaml"))
    df = pd.read_csv("data/processed/preprocessed.csv")

    # Split
    X_train, X_test, y_train, y_test = split_data(
        df,
        test_size=params["data"]["test_size"],
        random_state=params["data"]["random_state"],
    )

    # Vectorize
    vectorizer = build_tfidf_vectorizer(
        X_train,
        max_features=params["preprocessing"]["max_features"],
        ngram_range=(params["preprocessing"]["ngram_range_min"], params["preprocessing"]["ngram_range_max"]),
        min_df=params["preprocessing"]["min_df"],
        max_df=params["preprocessing"]["max_df"],
    )
    X_train_tfidf = vectorizer.transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train
    algorithm = params["model"]["algorithm"]
    hyperparameters = params["model"]
    model = get_model(algorithm, hyperparameters)
    model = train_model(model, X_train_tfidf, y_train)

    # Evaluate
    metrics = evaluate_model(model, X_test_tfidf, y_test)

    # Save artifacts
    os.makedirs("models/best_model", exist_ok=True)
    save_model(model, "models/best_model/model.pkl")
    save_vectorizer(vectorizer, "models/best_model/vectorizer.pkl")

    metrics_save = {k: v for k, v in metrics.items() if k not in ["classification_report", "confusion_matrix"]}
    with open("models/best_model/metrics.json", "w") as f:
        json.dump(metrics_save, f, indent=2)

    logging.info("Training complete. Accuracy: %.4f, F1: %.4f", metrics["accuracy"], metrics["f1_score"])
    kwargs["ti"].xcom_push(key="accuracy", value=metrics["accuracy"])
    kwargs["ti"].xcom_push(key="f1_score", value=metrics["f1_score"])
    return metrics_save


def task_validate_model(**kwargs):
    """Stage 5: Validate trained model meets acceptance criteria."""
    ti = kwargs["ti"]
    accuracy = ti.xcom_pull(task_ids="train_model", key="accuracy")
    f1_score = ti.xcom_pull(task_ids="train_model", key="f1_score")

    min_accuracy = 0.85
    if accuracy < min_accuracy:
        raise ValueError(
            f"Model accuracy {accuracy:.4f} below threshold {min_accuracy}. "
            "Model not deployed."
        )

    logging.info("Model validation passed. Accuracy=%.4f, F1=%.4f", accuracy, f1_score)
    return {"status": "validated", "accuracy": accuracy, "f1_score": f1_score}


# ---------------------------------------------------------------------------
# DAG Definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="fake_news_ml_pipeline",
    default_args=default_args,
    description="End-to-end ML pipeline for Fake News Detection with Airflow orchestration",
    schedule_interval="@weekly",
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "fake-news", "ml-pipeline"],
) as dag:

    # -----------------------------------------------------------------------
    # FileSensor: Wait for new data file (A6 requirement)
    # -----------------------------------------------------------------------
    wait_for_data = FileSensor(
        task_id="wait_for_data",
        filepath="data/raw/news.csv",
        poke_interval=60,             # Check every 60 seconds
        timeout=12 * 60 * 60,         # 12-hour timeout
        mode="poke",
        soft_fail=False,
    )

    # -----------------------------------------------------------------------
    # Pipeline Tasks (using worker pool for concurrency control)
    # -----------------------------------------------------------------------
    ingest = PythonOperator(
        task_id="data_ingestion",
        python_callable=task_data_ingestion,
        pool="ml_pipeline_pool",       # A6: Airflow Pool for concurrency
    )

    baselines = PythonOperator(
        task_id="compute_baselines",
        python_callable=task_compute_baselines,
        pool="ml_pipeline_pool",
    )

    preprocess = PythonOperator(
        task_id="preprocessing",
        python_callable=task_preprocessing,
        pool="ml_pipeline_pool",
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=task_train_model,
        pool="ml_pipeline_pool",
    )

    validate = PythonOperator(
        task_id="validate_model",
        python_callable=task_validate_model,
        pool="ml_pipeline_pool",
    )

    # -----------------------------------------------------------------------
    # Email Notifications (A6 requirement)
    # -----------------------------------------------------------------------
    success_email = EmailOperator(
        task_id="send_success_email",
        to="team@example.com",
        subject="[Fake News Pipeline] Training Completed Successfully",
        html_content="""
        <h2>ML Pipeline Completed</h2>
        <p>The Fake News Detection model has been retrained successfully.</p>
        <p>Check MLflow UI for detailed metrics.</p>
        """,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    failure_email = EmailOperator(
        task_id="send_failure_email",
        to="team@example.com",
        subject="[Fake News Pipeline] FAILURE - Pipeline Error",
        html_content="""
        <h2>Pipeline Failure Alert</h2>
        <p>The Fake News Detection pipeline has failed.</p>
        <p>Please check Airflow logs for details.</p>
        """,
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    # -----------------------------------------------------------------------
    # Task Dependencies (A6 requirement: explicit >> operator)
    # -----------------------------------------------------------------------
    wait_for_data >> ingest >> [baselines, preprocess]
    preprocess >> train >> validate >> success_email
    train >> failure_email