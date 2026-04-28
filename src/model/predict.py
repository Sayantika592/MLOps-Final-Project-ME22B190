"""
Prediction Module
=================
Provides inference functionality for the fake news classifier.
"""

import math
import logging
import numpy as np
from src.data.preprocess import clean_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class FakeNewsPredictor:
    """
    Encapsulates the trained model and vectorizer for real-time prediction.

    Attributes:
        model: Trained sklearn classifier.
        vectorizer: Fitted TfidfVectorizer.
        label_map: Mapping from numeric labels to human-readable strings.
    """

    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer
        self.label_map = {0: "REAL", 1: "FAKE"}

    def predict(self, text: str) -> dict:
        """
        Predict whether a given news text is fake or real.

        Args:
            text: Raw news headline or article text.

        Returns:
            Dictionary with keys: prediction, label, confidence, cleaned_text.
        """
        # Clean the input text
        cleaned = clean_text(text)

        if not cleaned.strip():
            return {
                "prediction": -1,
                "label": "INVALID",
                "confidence": 0.0,
                "cleaned_text": "",
                "error": "Input text is empty after cleaning.",
            }

        # Vectorize
        features = self.vectorizer.transform([cleaned])

        # Predict
        prediction = int(self.model.predict(features)[0])
        label = self.label_map.get(prediction, "UNKNOWN")

        # Confidence as probability (always 0.0 to 1.0)
        confidence = 0.5  # default = uncertain

        if hasattr(self.model, "predict_proba"):
            # Models with predict_proba (LogisticRegression, RandomForest, SGD with log_loss)
            proba = self.model.predict_proba(features)[0]
            # Confidence = probability of the PREDICTED class
            confidence = float(proba[prediction])
        elif hasattr(self.model, "decision_function"):
            # Models with only decision_function (SVM, SGD with hinge)
            score = self.model.decision_function(features)[0]
            # Convert raw score to probability via sigmoid
            sigmoid = 1.0 / (1.0 + math.exp(-float(score)))
            # sigmoid > 0.5 means class 1 (FAKE), < 0.5 means class 0 (REAL)
            if prediction == 1:
                confidence = sigmoid
            else:
                confidence = 1.0 - sigmoid

        result = {
            "prediction": prediction,
            "label": label,
            "confidence": round(confidence, 4),
            "cleaned_text": cleaned,
            "text_length": len(cleaned),
            "word_count": len(cleaned.split()),
        }

        logger.info("Prediction: %s (confidence=%.4f)", label, confidence)
        return result

    def predict_batch(self, texts: list) -> list:
        """
        Predict on a batch of texts.

        Args:
            texts: List of raw news text strings.

        Returns:
            List of prediction dictionaries.
        """
        return [self.predict(t) for t in texts]