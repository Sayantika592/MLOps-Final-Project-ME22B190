"""
Airflow DAG: Fake News Detection ML Pipeline
=============================================
Orchestrates the end-to-end ML pipeline using Apache Airflow.

Features (per A6 requirements):
- FileSensor: Watches for new dataset files in data/raw/
- Dry Pipeline Alert: Email when no data detected within timeout
- Worker Pool: Limits concurrency for CPU-heavy tasks (pool size 3)
- Task Dependencies: Explicit >> operator chaining
- Retry Logic: Exponential backoff on failure
- Email Alerts: Success, failure, and dry pipeline notifications via smtplib
"""

import os
import json
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.sensors.filesystem import FileSensor
from airflow.utils.trigger_rule import TriggerRule

# ---------------------------------------------------------------------------
# SMTP Configuration (Mailtrap)
# ---------------------------------------------------------------------------
SMTP_HOST = "sandbox.smtp.mailtrap.io"
SMTP_PORT = 2525
SMTP_USER = "03d947f0b2b667"
SMTP_PASS = "5b499655df4b6a"
SMTP_FROM = "airflow@fakenews.local"
SMTP_TO = "team@example.com"

# ---------------------------------------------------------------------------
# Email Helper (same pattern as A6 web_scraper assignment)
# ---------------------------------------------------------------------------
def send_email(subject, html_content):
    """Send email via SMTP using smtplib."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = SMTP_FROM
    msg["To"] = SMTP_TO
    msg.attach(MIMEText(html_content, "html"))

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_FROM, SMTP_TO, msg.as_string())
    logging.info("Email sent: %s", subject)

# ---------------------------------------------------------------------------
# Task Functions
# ---------------------------------------------------------------------------
def task_data_ingestion(**kwargs):
    """Stage 1: Load and validate raw CSV data."""
    import pandas as pd
    import yaml

    base_dir = os.environ.get("AIRFLOW_HOME", "/opt/airflow")
    params = yaml.safe_load(open(os.path.join(base_dir, "params.yaml")))
    raw_path = os.path.join(base_dir, params["data"]["raw_path"])

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Data file not found: {raw_path}")

    df = pd.read_csv(raw_path)
    logging.info("Loaded %d rows from %s", len(df), raw_path)

    for col in ["text", "label"]:
        if col not in df.columns and col == "text" and "title" in df.columns:
            continue
        elif col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    for col in ["title", "text", "author"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    out_dir = os.path.join(base_dir, "data/processed")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "ingested.csv"), index=False)
    logging.info("Ingestion complete. Saved %d rows.", len(df))

    kwargs["ti"].xcom_push(key="row_count", value=len(df))
    return len(df)


def task_compute_baselines(**kwargs):
    """Stage 2: Compute baseline statistics for drift detection."""
    import pandas as pd
    import numpy as np

    base_dir = os.environ.get("AIRFLOW_HOME", "/opt/airflow")
    df = pd.read_csv(os.path.join(base_dir, "data/processed/ingested.csv"))
    stats = {}

    if "text" in df.columns:
        lengths = df["text"].str.len()
        stats["text_length"] = {
            "mean": float(lengths.mean()), "std": float(lengths.std()),
            "median": float(lengths.median()), "min": float(lengths.min()),
            "max": float(lengths.max()),
        }

    if "label" in df.columns:
        dist = df["label"].value_counts(normalize=True).to_dict()
        stats["label_distribution"] = {str(k): float(v) for k, v in dist.items()}

    with open(os.path.join(base_dir, "data/processed/baseline_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    logging.info("Baseline stats computed: %s", list(stats.keys()))
    return stats


def task_preprocessing(**kwargs):
    """Stage 3: Clean text and engineer features."""
    import sys
    base_dir = os.environ.get("AIRFLOW_HOME", "/opt/airflow")
    sys.path.insert(0, base_dir)
    import pandas as pd
    from src.data.preprocess import preprocess_dataframe

    df = pd.read_csv(os.path.join(base_dir, "data/processed/ingested.csv"))
    df = preprocess_dataframe(df)
    df.to_csv(os.path.join(base_dir, "data/processed/preprocessed.csv"), index=False)

    logging.info("Preprocessing complete. %d rows remaining.", len(df))
    kwargs["ti"].xcom_push(key="preprocessed_rows", value=len(df))
    return len(df)


def task_train_model(**kwargs):
    """Stage 4: Run MLflow experiments - 4 progressive models, register best."""
    PROJECT_DIR = "/home/hp/Downloads/fake-news-detection" 
    import sys
    sys.path.insert(0, PROJECT_DIR)
    import yaml
    import pickle
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    import mlflow
    import mlflow.sklearn
    from src.data.preprocess import preprocess_dataframe, split_data

    params = yaml.safe_load(open(os.path.join(PROJECT_DIR, "params.yaml")))
    df = pd.read_csv(os.path.join(PROJECT_DIR, "data/processed/preprocessed.csv"))

    X_train_text, X_test_text, y_train, y_test = split_data(
        df, test_size=params["data"]["test_size"],
        random_state=params["data"]["random_state"])

    EXPERIMENTS = [
        {"name": "Exp1_NaiveBayes_BOW_1K",
         "vec_class": CountVectorizer, "vec_params": {"max_features": 1000, "ngram_range": (1,1), "stop_words": "english"},
         "model_class": MultinomialNB, "model_params": {"alpha": 1.0}},
        {"name": "Exp2_LogReg_TFIDF_5K",
         "vec_class": TfidfVectorizer, "vec_params": {"max_features": 5000, "ngram_range": (1,1), "stop_words": "english", "sublinear_tf": True},
         "model_class": LogisticRegression, "model_params": {"max_iter": 300, "C": 0.5, "solver": "lbfgs", "random_state": 42}},
        {"name": "Exp3_LogReg_TFIDF_20K",
         "vec_class": TfidfVectorizer, "vec_params": {"max_features": 20000, "ngram_range": (1,2), "min_df": 2, "stop_words": "english", "sublinear_tf": True},
         "model_class": LogisticRegression, "model_params": {"max_iter": 1000, "C": 1.0, "solver": "lbfgs", "random_state": 42}},
        {"name": "Exp4_SGD_TFIDF_50K",
         "vec_class": TfidfVectorizer, "vec_params": {"max_features": 50000, "ngram_range": (1,2), "min_df": 2, "max_df": 0.95, "stop_words": "english", "sublinear_tf": True},
         "model_class": SGDClassifier, "model_params": {"loss": "log_loss", "penalty": "l2", "max_iter": 100, "random_state": 42}},
    ]

    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("fake-news-detection")

    all_results = []
    for exp in EXPERIMENTS:
        with mlflow.start_run(run_name=exp["name"]) as run:
            mlflow.log_param("algorithm", exp["model_class"].__name__)
            mlflow.log_param("n_features", exp["vec_params"]["max_features"])
            for k, v in exp["model_params"].items():
                mlflow.log_param(k, v)

            vec = exp["vec_class"](**exp["vec_params"])
            X_train = vec.fit_transform(X_train_text)
            X_test = vec.transform(X_test_text)

            # 5-fold CV with per-fold logging
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
                m = exp["model_class"](**exp["model_params"])
                m.fit(X_train[tr_idx], y_train.iloc[tr_idx])
                yp = m.predict(X_train[val_idx])
                mlflow.log_metric("cv_accuracy", accuracy_score(y_train.iloc[val_idx], yp), step=fold)
                mlflow.log_metric("cv_f1", f1_score(y_train.iloc[val_idx], yp, average="weighted"), step=fold)

            # Final model
            model = exp["model_class"](**exp["model_params"])
            model.fit(X_train, y_train)
            yp = model.predict(X_test)
            acc = accuracy_score(y_test, yp)
            f1 = f1_score(y_test, yp, average="weighted")
            mlflow.log_metric("test_accuracy", acc)
            mlflow.log_metric("test_f1", f1)
            mlflow.sklearn.log_model(model, "model")
            mlflow.register_model(f"runs:/{run.info.run_id}/model", "fake-news-classifier")
            logging.info("%s: acc=%.4f f1=%.4f", exp["name"], acc, f1)
            all_results.append({"name": exp["name"], "acc": acc, "f1": f1, "model": model, "vec": vec})

    # Save best model
    best = max(all_results, key=lambda x: x["f1"])
    model_dir = os.path.join(PROJECT_DIR, "models/best_model")
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
        pickle.dump(best["model"], f)
    with open(os.path.join(model_dir, "vectorizer.pkl"), "wb") as f:
        pickle.dump(best["vec"], f)
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump({"accuracy": best["acc"], "f1_score": best["f1"]}, f, indent=2)

    logging.info("BEST: %s (F1=%.4f)", best["name"], best["f1"])
    kwargs["ti"].xcom_push(key="accuracy", value=best["acc"])
    kwargs["ti"].xcom_push(key="f1_score", value=best["f1"])

def task_validate_model(**kwargs):
    """Stage 5: Validate trained model meets acceptance criteria."""
    ti = kwargs["ti"]
    accuracy = ti.xcom_pull(task_ids="train_model", key="accuracy")
    f1 = ti.xcom_pull(task_ids="train_model", key="f1_score")

    min_accuracy = 0.85
    if accuracy < min_accuracy:
        raise ValueError(f"Model accuracy {accuracy:.4f} below threshold {min_accuracy}.")

    logging.info("Model validation passed. Accuracy=%.4f, F1=%.4f", accuracy, f1)
    return {"status": "validated", "accuracy": accuracy, "f1_score": f1}


# ---------------------------------------------------------------------------
# Email Task Functions
# ---------------------------------------------------------------------------
def send_success(**kwargs):
    send_email(
        "[Fake News Pipeline] Training Completed Successfully",
        """<h2>ML Pipeline Completed</h2>
        <p>The Fake News Detection model has been retrained successfully.</p>
        <p>Check MLflow UI for detailed metrics.</p>"""
    )

def send_failure(**kwargs):
    import sqlite3
    import os
    
    ti = kwargs.get("ti")
    dag_id = ti.dag_id if ti else "unknown"
    run_id = ti.run_id if ti else "unknown"
    try_number = ti.try_number if ti else 0
    
    failed_tasks = "upstream"
    try:
        db_path = os.path.expanduser("~/airflow/airflow.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT task_id FROM task_instance WHERE dag_id=? AND run_id=? AND state='failed'", (dag_id, run_id))
        failed_tasks = ", ".join([row[0] for row in cursor.fetchall()]) or "wait_for_data"
        conn.close()
    except Exception as e:
        failed_tasks = "upstream task"

    send_email(
        f"[Fake News Pipeline] FAILURE in {failed_tasks}",
        f"""<h2>Pipeline Failure Alert</h2>
        <p><b>DAG:</b> {dag_id}</p>
        <p><b>Failed Task:</b> {failed_tasks}</p>
        <p><b>Run ID:</b> {run_id}</p>
        <p><b>Attempt:</b> {try_number}</p>
        <p><b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Please check Airflow logs for full traceback.</p>"""
    )

def send_dry_alert(**kwargs):
    send_email(
        "[Fake News Pipeline] DRY PIPE - No New Data",
        """<h2>Dry Pipeline Alert</h2>
        <p>The FileSensor timed out waiting for <code>data/raw/news.csv</code>.</p>
        <p>No new data was detected within the configured timeout window.</p>
        <p>Please verify the data source is producing files correctly.</p>"""
    )


# ---------------------------------------------------------------------------
# Default DAG arguments with retry logic (A6 requirement)
# ---------------------------------------------------------------------------
default_args = {
    "owner": "fake-news-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=2),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=15),
    "start_date": datetime(2026, 4, 1),
}

# ---------------------------------------------------------------------------
# DAG Definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="fake_news_ml_pipeline",
    default_args=default_args,
    description="End-to-end ML pipeline for Fake News Detection with Airflow orchestration",
    schedule="@weekly",
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "fake-news", "ml-pipeline"],
) as dag:

    # FileSensor: Wait for data file
    wait_for_data = FileSensor(
        task_id="wait_for_data",
        filepath="data/raw/news.csv",
        fs_conn_id="fs_default",
        poke_interval=60,
        timeout=12 * 60 * 60,
        mode="poke",
    )

    # Pipeline Tasks (pool for concurrency control)
    ingest = PythonOperator(
        task_id="data_ingestion",
        python_callable=task_data_ingestion,
        pool="ml_pipeline_pool",
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

    # Email Tasks (PythonOperator + smtplib — same pattern as A6 assignment)
    success_email = PythonOperator(
        task_id="send_success_email",
        python_callable=send_success,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    failure_email = PythonOperator(
        task_id="send_failure_email",
        python_callable=send_failure,
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    dry_pipeline_email = PythonOperator(
        task_id="send_dry_pipeline_alert",
        python_callable=send_dry_alert,
        trigger_rule=TriggerRule.ALL_FAILED,
    )

    # Task Dependencies
    wait_for_data >> ingest >> [baselines, preprocess]
    preprocess >> train >> validate >> success_email
    [ingest, preprocess, train, validate] >> failure_email
    wait_for_data >> dry_pipeline_email

