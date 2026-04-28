"""
Model Training Module
=====================
Handles model selection, training, and experiment tracking via MLflow.
"""

import os
import logging
import pickle
import json

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# Model factory

MODEL_REGISTRY = {
    "passive_aggressive": PassiveAggressiveClassifier,
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
}


def get_model(algorithm: str, hyperparameters: dict):
    """
    Instantiate a model from the registry.

    Args:
        algorithm: Name of the algorithm (must be in MODEL_REGISTRY).
        hyperparameters: Dict of hyperparameters for the chosen algorithm.

    Returns:
        Instantiated sklearn estimator.
    """
    if algorithm not in MODEL_REGISTRY:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from {list(MODEL_REGISTRY)}")

    params = hyperparameters.get(algorithm, {})
    model = MODEL_REGISTRY[algorithm](**params)
    logger.info("Created model: %s with params %s", algorithm, params)
    return model

# Training

def train_model(model, X_train, y_train):
    """
    Train the model on the training data.

    Args:
        model: Sklearn estimator.
        X_train: Training feature matrix (sparse or dense).
        y_train: Training labels.

    Returns:
        Fitted model.
    """
    logger.info("Training model: %s", type(model).__name__)
    model.fit(X_train, y_train)
    logger.info("Training complete.")
    return model

# Evaluation

def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluate the model and return a metrics dictionary.

    Args:
        model: Trained sklearn estimator.
        X_test: Test feature matrix.
        y_test: Test labels.

    Returns:
        Dictionary with accuracy, precision, recall, f1, confusion_matrix,
        and classification_report.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted")),
        "recall": float(recall_score(y_test, y_pred, average="weighted")),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted")),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    logger.info("Evaluation Metrics:")
    logger.info("  Accuracy : %.4f", metrics["accuracy"])
    logger.info("  Precision: %.4f", metrics["precision"])
    logger.info("  Recall   : %.4f", metrics["recall"])
    logger.info("  F1-Score : %.4f", metrics["f1_score"])

    return metrics

# MLflow tracking


def train_with_mlflow(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    algorithm: str,
    hyperparameters: dict,
    tracking_uri: str = "http://localhost:5000",
    experiment_name: str = "fake-news-detection",
    vectorizer_path: str = None,
) -> dict:
    """
    Train model with full MLflow experiment tracking.

    Args:
        model: Sklearn estimator.
        X_train, y_train: Training data.
        X_test, y_test: Test data.
        algorithm: Name of algorithm used.
        hyperparameters: Hyperparameters dict.
        tracking_uri: MLflow tracking server URI.
        experiment_name: MLflow experiment name.
        vectorizer_path: Path to the saved vectorizer artifact.

    Returns:
        Dictionary with metrics and MLflow run_id.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info("MLflow run started: %s", run_id)

        # Log parameters
        mlflow.log_param("algorithm", algorithm)
        params = hyperparameters.get(algorithm, {})
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_param("train_size", X_train.shape[0])
        mlflow.log_param("test_size", X_test.shape[0])
        mlflow.log_param("n_features", X_train.shape[1])

        # Train
        model = train_model(model, X_train, y_train)

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)

        # Log metrics
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("f1_score", metrics["f1_score"])

        # Log classification report as artifact
        report_path = "/tmp/classification_report.json"
        with open(report_path, "w") as f:
            json.dump(metrics["classification_report"], f, indent=2)
        mlflow.log_artifact(report_path)

        # Log the vectorizer as an artifact
        if vectorizer_path and os.path.exists(vectorizer_path):
            mlflow.log_artifact(vectorizer_path)

        # Log the model
        mlflow.sklearn.log_model(model, "model")

        # Register the model in MLflow Model Registry
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri, "fake-news-classifier")

        logger.info("MLflow run completed: %s", run_id)

        metrics["run_id"] = run_id
        return metrics

# Save / Load


def save_model(model, path: str) -> None:
    """Save model to disk using pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved to %s", path)


def load_model(path: str):
    """Load model from disk."""
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded from %s", path)
    return model
