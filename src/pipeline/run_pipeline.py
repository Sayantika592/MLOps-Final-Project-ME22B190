"""
ML Pipeline Orchestrator
========================
End-to-end pipeline that runs data ingestion, preprocessing, training,
evaluation, and model export in a single reproducible workflow.
"""

import os
import sys
import json
import time
import logging
import yaml
import pickle

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.ingest import load_data, compute_baseline_statistics, load_config
from src.data.preprocess import preprocess_dataframe, build_tfidf_vectorizer, split_data, save_vectorizer
from src.model.train import get_model, train_model, evaluate_model, save_model, train_with_mlflow


class PipelineStatus:
    """Tracks the status of each pipeline stage."""

    def __init__(self):
        self.stages = {}
        self.start_time = time.time()

    def start_stage(self, name: str):
        self.stages[name] = {"status": "running", "start": time.time()}
        logger.info("=" * 60)
        logger.info("STAGE: %s — STARTED", name.upper())
        logger.info("=" * 60)

    def complete_stage(self, name: str, details: dict = None):
        stage = self.stages[name]
        stage["status"] = "completed"
        stage["duration"] = round(time.time() - stage["start"], 2)
        stage["details"] = details or {}
        logger.info("STAGE: %s — COMPLETED in %.2fs", name.upper(), stage["duration"])

    def fail_stage(self, name: str, error: str):
        stage = self.stages[name]
        stage["status"] = "failed"
        stage["duration"] = round(time.time() - stage["start"], 2)
        stage["error"] = error
        logger.error("STAGE: %s — FAILED: %s", name.upper(), error)

    def summary(self) -> dict:
        total = round(time.time() - self.start_time, 2)
        return {
            "total_duration": total,
            "stages": self.stages,
            "overall_status": "failed" if any(
                s["status"] == "failed" for s in self.stages.values()
            ) else "completed",
        }


def run_pipeline(config_path: str = "configs/model_config.yaml", use_mlflow: bool = False):
    """
    Execute the full ML pipeline.

    Args:
        config_path: Path to the YAML configuration file.
        use_mlflow: Whether to use MLflow for experiment tracking.

    Returns:
        PipelineStatus object with results from each stage.
    """
    config = load_config(config_path)
    status = PipelineStatus()

    # -----------------------------------------------------------------------
    # Stage 1: Data Ingestion
    # -----------------------------------------------------------------------
    status.start_stage("data_ingestion")
    try:
        df = load_data(config["data"]["raw_path"])
        status.complete_stage("data_ingestion", {
            "rows": len(df),
            "columns": list(df.columns),
        })
    except Exception as e:
        status.fail_stage("data_ingestion", str(e))
        logger.exception("Pipeline aborted at data ingestion.")
        return status

    # -----------------------------------------------------------------------
    # Stage 2: Baseline Statistics (for drift detection)
    # -----------------------------------------------------------------------
    status.start_stage("baseline_statistics")
    try:
        baseline_stats = compute_baseline_statistics(df)
        os.makedirs("data/processed", exist_ok=True)
        with open("data/processed/baseline_stats.json", "w") as f:
            json.dump(baseline_stats, f, indent=2)
        status.complete_stage("baseline_statistics", baseline_stats)
    except Exception as e:
        status.fail_stage("baseline_statistics", str(e))
        logger.warning("Baseline computation failed, continuing without drift baselines.")

    # -----------------------------------------------------------------------
    # Stage 3: Data Preprocessing
    # -----------------------------------------------------------------------
    status.start_stage("preprocessing")
    try:
        df = preprocess_dataframe(df)
        status.complete_stage("preprocessing", {
            "rows_after_cleaning": len(df),
            "features_created": ["content", "text_length", "word_count", "avg_word_length"],
        })
    except Exception as e:
        status.fail_stage("preprocessing", str(e))
        logger.exception("Pipeline aborted at preprocessing.")
        return status

    # -----------------------------------------------------------------------
    # Stage 4: Train/Test Split
    # -----------------------------------------------------------------------
    status.start_stage("data_split")
    try:
        X_train, X_test, y_train, y_test = split_data(
            df,
            test_size=config["data"]["test_size"],
            random_state=config["data"]["random_state"],
        )
        status.complete_stage("data_split", {
            "train_size": len(X_train),
            "test_size": len(X_test),
        })
    except Exception as e:
        status.fail_stage("data_split", str(e))
        return status

    # -----------------------------------------------------------------------
    # Stage 5: Feature Engineering (TF-IDF)
    # -----------------------------------------------------------------------
    status.start_stage("feature_engineering")
    try:
        prep_config = config["preprocessing"]
        vectorizer = build_tfidf_vectorizer(
            X_train,
            max_features=prep_config["max_features"],
            ngram_range=tuple(prep_config["ngram_range"]),
            min_df=prep_config["min_df"],
            max_df=prep_config["max_df"],
        )

        X_train_tfidf = vectorizer.transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # Save vectorizer
        vectorizer_path = "models/best_model/vectorizer.pkl"
        save_vectorizer(vectorizer, vectorizer_path)

        status.complete_stage("feature_engineering", {
            "n_features": X_train_tfidf.shape[1],
            "vectorizer_path": vectorizer_path,
        })
    except Exception as e:
        status.fail_stage("feature_engineering", str(e))
        return status

    # -----------------------------------------------------------------------
    # Stage 6: Model Training & Evaluation
    # -----------------------------------------------------------------------
    status.start_stage("model_training")
    try:
        algorithm = config["model"]["algorithm"]
        hyperparameters = config["model"]["hyperparameters"]

        if use_mlflow:
            model_obj = get_model(algorithm, hyperparameters)
            metrics = train_with_mlflow(
                model=model_obj,
                X_train=X_train_tfidf,
                y_train=y_train,
                X_test=X_test_tfidf,
                y_test=y_test,
                algorithm=algorithm,
                hyperparameters=hyperparameters,
                tracking_uri=config["mlflow"]["tracking_uri"],
                experiment_name=config["mlflow"]["experiment_name"],
                vectorizer_path=vectorizer_path,
            )
        else:
            model_obj = get_model(algorithm, hyperparameters)
            model_obj = train_model(model_obj, X_train_tfidf, y_train)
            metrics = evaluate_model(model_obj, X_test_tfidf, y_test)

        status.complete_stage("model_training", {
            "algorithm": algorithm,
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
        })
    except Exception as e:
        status.fail_stage("model_training", str(e))
        return status

    # -----------------------------------------------------------------------
    # Stage 7: Model Export
    # -----------------------------------------------------------------------
    status.start_stage("model_export")
    try:
        model_path = "models/best_model/model.pkl"
        save_model(model_obj, model_path)

        # Save metrics
        metrics_path = "models/best_model/metrics.json"
        serializable_metrics = {
            k: v for k, v in metrics.items()
            if k not in ["classification_report", "confusion_matrix"]
        }
        with open(metrics_path, "w") as f:
            json.dump(serializable_metrics, f, indent=2)

        status.complete_stage("model_export", {
            "model_path": model_path,
            "metrics_path": metrics_path,
        })
    except Exception as e:
        status.fail_stage("model_export", str(e))
        return status

    # -----------------------------------------------------------------------
    # Pipeline Summary
    # -----------------------------------------------------------------------
    summary = status.summary()
    logger.info("=" * 60)
    logger.info("PIPELINE %s in %.2fs", summary["overall_status"].upper(), summary["total_duration"])
    logger.info("=" * 60)

    # Save pipeline status
    with open("models/best_model/pipeline_status.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return status


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Fake News Detection ML pipeline.")
    parser.add_argument("--config", default="configs/model_config.yaml", help="Path to config file.")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow tracking.")
    args = parser.parse_args()

    run_pipeline(config_path=args.config, use_mlflow=args.mlflow)
