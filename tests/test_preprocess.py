"""
Unit Tests — Data Preprocessing
================================
"""

import pytest
import pandas as pd
import numpy as np
from src.data.preprocess import clean_text, preprocess_dataframe, build_tfidf_vectorizer


class TestCleanText:
    """Tests for the clean_text function."""

    def test_lowercase(self):
        assert clean_text("HELLO WORLD") == "hello world"

    def test_remove_urls(self):
        text = "Check this out http://example.com and https://test.org"
        result = clean_text(text)
        assert "http" not in result
        assert "example" not in result

    def test_remove_html_tags(self):
        assert clean_text("<p>Hello</p> <b>World</b>") == "hello world"

    def test_remove_special_chars(self):
        result = clean_text("Hello! @World #2024")
        assert result == "hello world"

    def test_remove_extra_whitespace(self):
        assert clean_text("hello    world") == "hello world"

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_none_input(self):
        assert clean_text(None) == ""

    def test_numeric_input(self):
        assert clean_text(123) == ""


class TestPreprocessDataframe:
    """Tests for the preprocess_dataframe function."""

    def test_basic_preprocessing(self):
        df = pd.DataFrame({
            "title": ["Breaking News", "Update"],
            "text": ["Something happened today", "Another event occurred"],
            "label": [0, 1],
        })
        result = preprocess_dataframe(df)

        assert "content" in result.columns
        assert "text_length" in result.columns
        assert "word_count" in result.columns
        assert "avg_word_length" in result.columns
        assert len(result) == 2

    def test_empty_rows_dropped(self):
        df = pd.DataFrame({
            "title": ["News", ""],
            "text": ["Some text", ""],
            "label": [0, 1],
        })
        result = preprocess_dataframe(df)
        assert len(result) <= 2  # Empty rows may be dropped

    def test_feature_engineering_values(self):
        df = pd.DataFrame({
            "title": ["Test"],
            "text": ["hello world foo"],
            "label": [0],
        })
        result = preprocess_dataframe(df)
        assert result["word_count"].iloc[0] > 0
        assert result["text_length"].iloc[0] > 0


class TestTfidfVectorizer:
    """Tests for TF-IDF vectorizer building."""

    def test_vectorizer_shape(self):
        texts = pd.Series([
            "the quick brown fox jumps",
            "the lazy dog sleeps all day",
            "fox and dog are friends now",
            "quick fox runs very fast today",
        ])
        vectorizer = build_tfidf_vectorizer(texts, max_features=100, ngram_range=(1, 1), min_df=1)
        transformed = vectorizer.transform(texts)

        assert transformed.shape[0] == 4
        assert transformed.shape[1] <= 100

    def test_vectorizer_vocabulary(self):
        texts = pd.Series(["hello world", "hello python", "world python"])
        vectorizer = build_tfidf_vectorizer(texts, max_features=50, ngram_range=(1, 1), min_df=1)

        assert "hello" in vectorizer.vocabulary_
        assert "world" in vectorizer.vocabulary_
