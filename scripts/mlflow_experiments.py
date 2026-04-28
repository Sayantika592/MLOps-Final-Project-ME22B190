"""
MLflow Multi-Experiment Training Script (v3 - proper probabilities)
====================================================================
Uses loss='log_loss' for SGDClassifier so predict_proba works correctly.
4 experiments with progressively better features showing clear improvement.

Usage:
    python scripts/mlflow_experiments.py
"""

import os
import sys
import json
import logging
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

import mlflow
import mlflow.sklearn
import yaml

sys.path.insert(0, os.getcwd())
from src.data.ingest import load_data
from src.data.preprocess import preprocess_dataframe, split_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENTS = [
    {
        "name": "Exp1_NaiveBayes_BOW_1K",
        "description": "Baseline: Naive Bayes with basic Bag-of-Words (1K features)",
        "vectorizer_class": CountVectorizer,
        "vectorizer_params": {"max_features": 1000, "ngram_range": (1, 1), "stop_words": "english"},
        "model_class": MultinomialNB,
        "model_params": {"alpha": 1.0},
    },
    {
        "name": "Exp2_LogReg_TFIDF_5K",
        "description": "Improved: Logistic Regression with TF-IDF (5K features)",
        "vectorizer_class": TfidfVectorizer,
        "vectorizer_params": {"max_features": 5000, "ngram_range": (1, 1), "stop_words": "english", "sublinear_tf": True},
        "model_class": LogisticRegression,
        "model_params": {"max_iter": 300, "C": 0.5, "solver": "lbfgs", "random_state": 42},
    },
    {
        "name": "Exp3_LogReg_TFIDF_20K",
        "description": "Better: Logistic Regression with TF-IDF (20K features, bigrams)",
        "vectorizer_class": TfidfVectorizer,
        "vectorizer_params": {"max_features": 20000, "ngram_range": (1, 2), "min_df": 2, "stop_words": "english", "sublinear_tf": True},
        "model_class": LogisticRegression,
        "model_params": {"max_iter": 1000, "C": 1.0, "solver": "lbfgs", "random_state": 42},
    },
    {
        "name": "Exp4_SGD_TFIDF_50K_Final",
        "description": "Best: SGD with log_loss, TF-IDF (50K features, bigrams, tuned)",
        "vectorizer_class": TfidfVectorizer,
        "vectorizer_params": {"max_features": 50000, "ngram_range": (1, 2), "min_df": 2, "max_df": 0.95, "stop_words": "english", "sublinear_tf": True},
        "model_class": SGDClassifier,
        "model_params": {"loss": "log_loss", "penalty": "l2", "max_iter": 100, "tol": 0.001, "random_state": 42},
    },
]

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
EXPERIMENT_NAME = "fake-news-detection"
N_FOLDS = 5


def plot_confusion_matrix(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["REAL", "FAKE"])
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_fold_metrics(fold_metrics, title, path):
    folds = list(range(1, len(fold_metrics) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(folds, [m["accuracy"] for m in fold_metrics], "o-", label="Accuracy", color="#2ecc71", linewidth=2)
    axes[0].plot(folds, [m["f1"] for m in fold_metrics], "s-", label="F1-Score", color="#3498db", linewidth=2)
    axes[0].set_xlabel("CV Fold"); axes[0].set_ylabel("Score")
    axes[0].set_title(f"{title} - Accuracy & F1"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(folds, [m["precision"] for m in fold_metrics], "^-", label="Precision", color="#e74c3c", linewidth=2)
    axes[1].plot(folds, [m["recall"] for m in fold_metrics], "D-", label="Recall", color="#f39c12", linewidth=2)
    axes[1].set_xlabel("CV Fold"); axes[1].set_ylabel("Score")
    axes[1].set_title(f"{title} - Precision & Recall"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_experiment_comparison(all_results, path):
    names = [r["name"].replace("_", "\n") for r in all_results]
    metrics_keys = ["accuracy", "f1", "precision", "recall"]
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12"]
    x = np.arange(len(names)); width = 0.2
    fig, ax = plt.subplots(figsize=(14, 7))
    for i, (m, c) in enumerate(zip(metrics_keys, colors)):
        vals = [r[m] for r in all_results]
        bars = ax.bar(x + i * width, vals, width, label=m.capitalize(), color=c)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xlabel("Experiment"); ax.set_ylabel("Score")
    ax.set_title("Experiment Comparison - Progressive Improvement", fontweight="bold")
    ax.set_xticks(x + width * 1.5); ax.set_xticklabels(names, fontsize=9); ax.legend()
    all_v = [r[m] for r in all_results for m in metrics_keys]
    ax.set_ylim(max(0, min(all_v) - 0.05), 1.02); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_improvement_line(all_results, path):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(1, len(all_results) + 1)
    for name, color in [("accuracy","#2ecc71"),("f1","#3498db"),("precision","#e74c3c"),("recall","#f39c12")]:
        vals = [r[name] for r in all_results]
        ax.plot(x, vals, "o-", label=name.capitalize(), color=color, linewidth=2.5, markersize=8)
        for xi, v in zip(x, vals):
            ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Experiment #"); ax.set_ylabel("Score")
    ax.set_title("Model Improvement Across Experiments", fontweight="bold")
    ax.set_xticks(list(x)); ax.set_xticklabels([f"Exp {i}" for i in x]); ax.legend(loc="lower right")
    all_v = [r[m] for r in all_results for m in ["accuracy","f1","precision","recall"]]
    ax.set_ylim(max(0, min(all_v) - 0.05), 1.02); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def run_experiment(exp_config, X_train_text, y_train, X_test_text, y_test):
    name = exp_config["name"]
    vec_class = exp_config["vectorizer_class"]
    vec_params = exp_config["vectorizer_params"]
    model_class = exp_config["model_class"]
    model_params = exp_config["model_params"]

    logger.info("=" * 60)
    logger.info("EXPERIMENT: %s", name)
    logger.info("  %s", exp_config["description"])
    logger.info("=" * 60)

    with mlflow.start_run(run_name=name) as run:
        run_id = run.info.run_id
        mlflow.log_param("experiment_name", name)
        mlflow.log_param("description", exp_config["description"])
        mlflow.log_param("algorithm", model_class.__name__)
        mlflow.log_param("vectorizer", vec_class.__name__)
        mlflow.log_param("n_folds", N_FOLDS)
        mlflow.log_param("train_size", len(X_train_text))
        mlflow.log_param("test_size", len(X_test_text))
        for k, v in vec_params.items():
            mlflow.log_param(f"vec_{k}", v)
        for k, v in model_params.items():
            mlflow.log_param(k, v)

        vectorizer = vec_class(**vec_params)
        X_train = vectorizer.fit_transform(X_train_text)
        X_test = vectorizer.transform(X_test_text)
        n_features = X_train.shape[1]
        mlflow.log_param("n_features", n_features)
        logger.info("  Vectorizer: %s with %d features", vec_class.__name__, n_features)

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
            Xf_tr, yf_tr = X_train[train_idx], y_train.iloc[train_idx] if hasattr(y_train,"iloc") else y_train[train_idx]
            Xf_val, yf_val = X_train[val_idx], y_train.iloc[val_idx] if hasattr(y_train,"iloc") else y_train[val_idx]
            model = model_class(**model_params)
            model.fit(Xf_tr, yf_tr)
            yp = model.predict(Xf_val)
            fa, ff = accuracy_score(yf_val, yp), f1_score(yf_val, yp, average="weighted")
            fp, fr = precision_score(yf_val, yp, average="weighted"), recall_score(yf_val, yp, average="weighted")
            mlflow.log_metric("cv_accuracy", fa, step=fold)
            mlflow.log_metric("cv_f1_score", ff, step=fold)
            mlflow.log_metric("cv_precision", fp, step=fold)
            mlflow.log_metric("cv_recall", fr, step=fold)
            fold_metrics.append({"fold": fold, "accuracy": fa, "f1": ff, "precision": fp, "recall": fr})
            logger.info("  Fold %d: acc=%.4f, f1=%.4f", fold, fa, ff)

        final_model = model_class(**model_params)
        final_model.fit(X_train, y_train)
        y_test_pred = final_model.predict(X_test)
        ta = accuracy_score(y_test, y_test_pred)
        tf = f1_score(y_test, y_test_pred, average="weighted")
        tp = precision_score(y_test, y_test_pred, average="weighted")
        tr = recall_score(y_test, y_test_pred, average="weighted")
        mlflow.log_metric("test_accuracy", ta)
        mlflow.log_metric("test_f1_score", tf)
        mlflow.log_metric("test_precision", tp)
        mlflow.log_metric("test_recall", tr)
        mlflow.log_metric("cv_mean_accuracy", np.mean([m["accuracy"] for m in fold_metrics]))
        mlflow.log_metric("cv_std_accuracy", np.std([m["accuracy"] for m in fold_metrics]))
        logger.info("  TEST: acc=%.4f, f1=%.4f, prec=%.4f, rec=%.4f", ta, tf, tp, tr)

        os.makedirs("/tmp/mlflow_artifacts", exist_ok=True)
        rpt = classification_report(y_test, y_test_pred, output_dict=True)
        rpath = f"/tmp/mlflow_artifacts/{name}_report.json"
        with open(rpath, "w") as f: json.dump(rpt, f, indent=2)
        mlflow.log_artifact(rpath)

        cpath = f"/tmp/mlflow_artifacts/{name}_cm.png"
        plot_confusion_matrix(y_test, y_test_pred, name, cpath)
        mlflow.log_artifact(cpath)

        fpath = f"/tmp/mlflow_artifacts/{name}_folds.png"
        plot_fold_metrics(fold_metrics, name, fpath)
        mlflow.log_artifact(fpath)

        vpath = f"/tmp/mlflow_artifacts/{name}_vectorizer.pkl"
        with open(vpath, "wb") as f: pickle.dump(vectorizer, f)
        mlflow.log_artifact(vpath)

        mlflow.sklearn.log_model(final_model, "model")
        try:
            mlflow.register_model(f"runs:/{run_id}/model", "fake-news-classifier")
            logger.info("  Registered model for %s", name)
        except Exception as e:
            logger.warning("  Registry note: %s", e)

        return {"name": name, "run_id": run_id, "accuracy": ta, "f1": tf,
                "precision": tp, "recall": tr, "cv_folds": fold_metrics,
                "n_features": n_features, "model": final_model, "vectorizer": vectorizer}


def main():
    params = yaml.safe_load(open("params.yaml"))
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info("Loading and preprocessing data...")
    df = load_data(params["data"]["raw_path"])
    df = preprocess_dataframe(df)
    X_train_text, X_test_text, y_train, y_test = split_data(
        df, test_size=params["data"]["test_size"], random_state=params["data"]["random_state"])
    logger.info("Train: %d, Test: %d", len(X_train_text), len(X_test_text))

    all_results = []
    for exp in EXPERIMENTS:
        result = run_experiment(exp, X_train_text, y_train, X_test_text, y_test)
        all_results.append(result)

    os.makedirs("/tmp/mlflow_artifacts", exist_ok=True)
    plot_experiment_comparison(all_results, "/tmp/mlflow_artifacts/comparison.png")
    plot_improvement_line(all_results, "/tmp/mlflow_artifacts/improvement.png")

    with mlflow.start_run(run_name="Final_Comparison"):
        mlflow.log_artifact("/tmp/mlflow_artifacts/comparison.png")
        mlflow.log_artifact("/tmp/mlflow_artifacts/improvement.png")
        for r in all_results:
            mlflow.log_metric(f"{r['name']}_accuracy", r["accuracy"])
            mlflow.log_metric(f"{r['name']}_f1", r["f1"])

    # Save BEST model for API serving
    best = max(all_results, key=lambda x: x["f1"])
    logger.info("\nBEST MODEL: %s (F1=%.4f)", best["name"], best["f1"])

    os.makedirs("models/best_model", exist_ok=True)
    with open("models/best_model/model.pkl", "wb") as f:
        pickle.dump(best["model"], f)
    with open("models/best_model/vectorizer.pkl", "wb") as f:
        pickle.dump(best["vectorizer"], f)
    with open("models/best_model/metrics.json", "w") as f:
        json.dump({"accuracy": best["accuracy"], "f1_score": best["f1"],
                    "precision": best["precision"], "recall": best["recall"]}, f, indent=2)
    logger.info("Best model saved to models/best_model/")

    # Quick sanity check - test the example texts
    logger.info("\n" + "=" * 60)
    logger.info("SANITY CHECK - Example Predictions")
    logger.info("=" * 60)
    from src.model.predict import FakeNewsPredictor
    predictor = FakeNewsPredictor(best["model"], best["vectorizer"])

    test_texts = [
        ("The Federal Reserve announced a quarter-point interest rate increase on Wednesday, citing continued economic growth and stable employment numbers.", "Should be REAL"),
        ("BREAKING: Scientists discover that the moon is actually made of cheese, NASA confirms in shocking press conference.", "Should be FAKE"),
    ]
    for text, expected in test_texts:
        result = predictor.predict(text)
        logger.info("  Input: %s...", text[:60])
        logger.info("  Result: %s (confidence=%.1f%%) — %s", result["label"], result["confidence"]*100, expected)

    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 70)
    logger.info("%-35s %-10s %-10s %-10s %-10s %-10s", "Experiment", "Features", "Acc", "F1", "Prec", "Rec")
    logger.info("-" * 70)
    for r in all_results:
        logger.info("%-35s %-10d %-10.4f %-10.4f %-10.4f %-10.4f",
                     r["name"], r["n_features"], r["accuracy"], r["f1"], r["precision"], r["recall"])
    logger.info("=" * 70)
    logger.info("BEST: %s (F1=%.4f)", best["name"], best["f1"])


if __name__ == "__main__":
    main()