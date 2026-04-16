"""
Unit Tests — Model Training & Prediction
==========================================
"""

import pytest
import numpy as np
from scipy.sparse import random as sparse_random
from sklearn.linear_model import PassiveAggressiveClassifier

from src.model.train import get_model, train_model, evaluate_model, MODEL_REGISTRY
from src.model.predict import FakeNewsPredictor


class TestModelFactory:
    """Tests for model instantiation."""

    def test_get_passive_aggressive(self):
        model = get_model("passive_aggressive", {"passive_aggressive": {"max_iter": 50}})
        assert isinstance(model, PassiveAggressiveClassifier)

    def test_unknown_algorithm_raises(self):
        with pytest.raises(ValueError, match="Unknown algorithm"):
            get_model("unknown_algo", {})

    def test_all_registry_models(self):
        for name in MODEL_REGISTRY:
            model = get_model(name, {name: {}})
            assert model is not None


class TestTrainEvaluate:
    """Tests for training and evaluation."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        X_train = sparse_random(100, 50, density=0.3, format="csr")
        X_test = sparse_random(30, 50, density=0.3, format="csr")
        y_train = np.random.randint(0, 2, size=100)
        y_test = np.random.randint(0, 2, size=30)
        return X_train, X_test, y_train, y_test

    def test_train_model(self, sample_data):
        X_train, _, y_train, _ = sample_data
        model = get_model("passive_aggressive", {"passive_aggressive": {"max_iter": 50}})
        trained = train_model(model, X_train, y_train)
        assert hasattr(trained, "predict")

    def test_evaluate_model(self, sample_data):
        X_train, X_test, y_train, y_test = sample_data
        model = get_model("passive_aggressive", {"passive_aggressive": {"max_iter": 50}})
        model = train_model(model, X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_evaluate_metrics_types(self, sample_data):
        X_train, X_test, y_train, y_test = sample_data
        model = get_model("logistic_regression", {"logistic_regression": {"max_iter": 100}})
        model = train_model(model, X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        assert isinstance(metrics["accuracy"], float)
        assert isinstance(metrics["confusion_matrix"], list)


class TestPredictor:
    """Tests for the FakeNewsPredictor class."""

    @pytest.fixture
    def predictor(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts = [
            "breaking news about politics today",
            "scientists discover new planet in galaxy",
            "fake story about aliens landing on earth",
            "real report on economic growth this quarter",
        ] * 10
        labels = [0, 0, 1, 0] * 10

        vectorizer = TfidfVectorizer(max_features=100)
        X = vectorizer.fit_transform(texts)

        model = PassiveAggressiveClassifier(max_iter=50)
        model.fit(X, labels)

        return FakeNewsPredictor(model, vectorizer)

    def test_predict_returns_dict(self, predictor):
        result = predictor.predict("Some news headline about politics")
        assert isinstance(result, dict)
        assert "prediction" in result
        assert "label" in result
        assert "confidence" in result

    def test_predict_label_values(self, predictor):
        result = predictor.predict("A test news article")
        assert result["label"] in ["REAL", "FAKE"]
        assert result["prediction"] in [0, 1]

    def test_predict_empty_text(self, predictor):
        result = predictor.predict("")
        assert result["label"] == "INVALID"

    def test_predict_batch(self, predictor):
        results = predictor.predict_batch(["News one", "News two"])
        assert len(results) == 2
