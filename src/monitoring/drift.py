"""
Data Drift Detection Module
============================
Monitors changes in input data distribution compared to training baselines.
"""

import logging
import json
import os
import numpy as np
from collections import deque

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Detects data drift by comparing recent input statistics against
    training baselines using Population Stability Index (PSI).

    Attributes:
        baseline_stats: Dictionary of baseline statistics from training data.
        window_size: Number of recent samples to keep in memory.
        threshold: Drift score above which an alert is triggered.
        recent_lengths: Sliding window of recent text lengths.
    """

    def __init__(self, baseline_path: str = None, window_size: int = 1000, threshold: float = 0.1):
        self.baseline_stats = {}
        self.window_size = window_size
        self.threshold = threshold
        self.recent_lengths = deque(maxlen=window_size)
        self.recent_word_counts = deque(maxlen=window_size)
        self.prediction_counts = {"REAL": 0, "FAKE": 0}

        if baseline_path and os.path.exists(baseline_path):
            self.load_baseline(baseline_path)

    def load_baseline(self, path: str) -> None:
        """Load baseline statistics from a JSON file."""
        with open(path, "r") as f:
            self.baseline_stats = json.load(f)
        logger.info("Loaded baseline statistics from %s", path)

    def save_baseline(self, stats: dict, path: str) -> None:
        """Save baseline statistics to a JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info("Saved baseline statistics to %s", path)

    def record_prediction(self, text_length: int, word_count: int, label: str) -> None:
        """
        Record a new prediction for drift monitoring.

        Args:
            text_length: Length of the input text in characters.
            word_count: Number of words in the input text.
            label: Predicted label string.
        """
        self.recent_lengths.append(text_length)
        self.recent_word_counts.append(word_count)
        if label in self.prediction_counts:
            self.prediction_counts[label] += 1

    def compute_drift_score(self) -> dict:
        """
        Compute the drift score by comparing recent data against baselines.

        Returns:
            Dictionary with individual and overall drift scores plus alert flag.
        """
        if len(self.recent_lengths) < 50:
            return {"overall_drift": 0.0, "alert": False, "message": "Insufficient data for drift detection."}

        result = {"features": {}}
        drift_scores = []

        # Text length drift (using z-score approach)
        if "text_length" in self.baseline_stats:
            baseline = self.baseline_stats["text_length"]
            recent_mean = float(np.mean(list(self.recent_lengths)))
            recent_std = float(np.std(list(self.recent_lengths))) or 1.0

            z_score = abs(recent_mean - baseline["mean"]) / (baseline["std"] or 1.0)
            # Normalize z-score to 0-1 range
            length_drift = min(z_score / 3.0, 1.0)

            result["features"]["text_length"] = {
                "baseline_mean": baseline["mean"],
                "current_mean": recent_mean,
                "drift_score": round(length_drift, 4),
            }
            drift_scores.append(length_drift)

        # Label distribution drift
        total_preds = sum(self.prediction_counts.values())
        if total_preds > 0 and "label_distribution" in self.baseline_stats:
            baseline_dist = self.baseline_stats["label_distribution"]
            current_dist = {k: v / total_preds for k, v in self.prediction_counts.items()}

            # Simple distribution difference
            label_drift = 0.0
            for label_key in ["0", "1"]:
                mapped_key = "REAL" if label_key == "0" else "FAKE"
                baseline_val = baseline_dist.get(label_key, 0.5)
                current_val = current_dist.get(mapped_key, 0.5)
                label_drift += abs(baseline_val - current_val)

            label_drift = min(label_drift, 1.0)
            result["features"]["label_distribution"] = {
                "baseline": baseline_dist,
                "current": current_dist,
                "drift_score": round(label_drift, 4),
            }
            drift_scores.append(label_drift)

        # Overall drift score
        overall = float(np.mean(drift_scores)) if drift_scores else 0.0
        result["overall_drift"] = round(overall, 4)
        result["alert"] = overall > self.threshold
        result["threshold"] = self.threshold
        result["samples_analyzed"] = len(self.recent_lengths)

        if result["alert"]:
            logger.warning("DRIFT ALERT: Overall drift score %.4f exceeds threshold %.2f", overall, self.threshold)

        return result
