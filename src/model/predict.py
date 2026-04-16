"""
Prediction Module
=================
Provides inference functionality for the fake news classifier.
"""

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

        # Confidence via decision function (if available)
        confidence = 0.0
        if hasattr(self.model, "decision_function"):
            score = self.model.decision_function(features)[0]
            confidence = float(abs(score))
        elif hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(features)[0]
            confidence = float(max(proba))

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
