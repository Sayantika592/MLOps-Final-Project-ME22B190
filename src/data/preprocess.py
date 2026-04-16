"""
Data Preprocessing Module
=========================
Handles text cleaning, tokenization, and feature engineering for the
fake news detection pipeline.
"""

import re
import logging
import pickle
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean a single text string by removing noise.

    Args:
        text: Raw input text.

    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove special characters and digits (keep letters and spaces)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply cleaning to all text columns and create combined feature.

    Args:
        df: Raw DataFrame with text columns.

    Returns:
        DataFrame with cleaned text and combined content column.
    """
    logger.info("Starting text preprocessing on %d rows", len(df))

    # Clean individual columns
    if "title" in df.columns:
        df["clean_title"] = df["title"].apply(clean_text)

    if "text" in df.columns:
        df["clean_text"] = df["text"].apply(clean_text)

    # Combine title and text into a single feature
    if "clean_title" in df.columns and "clean_text" in df.columns:
        df["content"] = df["clean_title"] + " " + df["clean_text"]
    elif "clean_text" in df.columns:
        df["content"] = df["clean_text"]
    elif "clean_title" in df.columns:
        df["content"] = df["clean_title"]
    else:
        raise ValueError("No text columns found after cleaning.")

    # Feature engineering: text length, word count, avg word length
    df["text_length"] = df["content"].str.len()
    df["word_count"] = df["content"].str.split().str.len()
    df["avg_word_length"] = df["content"].apply(
        lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0
    )

    # Drop rows with empty content
    initial_len = len(df)
    df = df[df["content"].str.strip().str.len() > 0].reset_index(drop=True)
    logger.info("Dropped %d empty rows. Remaining: %d", initial_len - len(df), len(df))

    return df


def build_tfidf_vectorizer(
    texts: pd.Series,
    max_features: int = 50000,
    ngram_range: tuple = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
) -> TfidfVectorizer:
    """
    Fit a TF-IDF vectorizer on the training texts.

    Args:
        texts: Series of cleaned text strings.
        max_features: Maximum number of features.
        ngram_range: Range of n-grams to consider.
        min_df: Minimum document frequency.
        max_df: Maximum document frequency.

    Returns:
        Fitted TfidfVectorizer object.
    """
    logger.info(
        "Building TF-IDF vectorizer (max_features=%d, ngram_range=%s)",
        max_features,
        ngram_range,
    )

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        stop_words="english",
        sublinear_tf=True,
    )

    vectorizer.fit(texts)
    logger.info("TF-IDF vectorizer fitted with %d features", len(vectorizer.vocabulary_))
    return vectorizer


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Split data into train and test sets with stratification.

    Args:
        df: Preprocessed DataFrame with 'content' and 'label' columns.
        test_size: Fraction of data to use for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X = df["content"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info("Train size: %d, Test size: %d", len(X_train), len(X_test))
    logger.info("Label distribution (train): %s", y_train.value_counts().to_dict())
    logger.info("Label distribution (test):  %s", y_test.value_counts().to_dict())

    return X_train, X_test, y_train, y_test


def save_vectorizer(vectorizer: TfidfVectorizer, path: str) -> None:
    """Save the fitted vectorizer to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(vectorizer, f)
    logger.info("Vectorizer saved to %s", path)


def load_vectorizer(path: str) -> TfidfVectorizer:
    """Load a fitted vectorizer from disk."""
    with open(path, "rb") as f:
        vectorizer = pickle.load(f)
    logger.info("Vectorizer loaded from %s", path)
    return vectorizer
