"""
Data Ingestion Module
=====================
Handles downloading, loading, and initial validation of the fake news dataset.
"""

import os
import logging
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/model_config.yaml") -> dict:
    """Load project configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def validate_schema(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Validate that the DataFrame contains all required columns.

    Args:
        df: Input DataFrame to validate.
        required_columns: List of column names that must be present.

    Returns:
        True if schema is valid.

    Raises:
        ValueError: If required columns are missing.
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    logger.info("Schema validation passed. Columns: %s", list(df.columns))
    return True


def check_missing_values(df: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
    """
    Check and report missing values. Drop columns exceeding threshold.

    Args:
        df: Input DataFrame.
        threshold: Maximum fraction of missing values allowed per column.

    Returns:
        DataFrame with high-missing columns dropped.
    """
    missing_pct = df.isnull().mean()
    logger.info("Missing value percentages:\n%s", missing_pct.to_string())

    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
    if cols_to_drop:
        logger.warning("Dropping columns with >%.0f%% missing: %s", threshold * 100, cols_to_drop)
        df = df.drop(columns=cols_to_drop)

    return df


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file with validation.

    Args:
        data_path: Path to the CSV file.

    Returns:
        Validated DataFrame.

    Raises:
        FileNotFoundError: If the data file does not exist.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path)
    logger.info("Loaded %d rows and %d columns", df.shape[0], df.shape[1])

    # Validate schema - we expect at minimum a text column and a label column
    # The dataset may have: id, title, author, text, label
    if "text" not in df.columns and "title" not in df.columns:
        raise ValueError("Dataset must contain at least 'text' or 'title' column.")

    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column (0=Real, 1=Fake).")

    # Handle missing values
    df = check_missing_values(df)

    # Fill remaining NaN in text columns with empty string
    text_cols = [c for c in ["title", "text", "author"] if c in df.columns]
    for col in text_cols:
        df[col] = df[col].fillna("")

    logger.info("Data ingestion complete. Final shape: %s", df.shape)
    return df


def compute_baseline_statistics(df: pd.DataFrame) -> dict:
    """
    Compute baseline statistics for drift detection.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary of baseline statistics per feature.
    """
    stats = {}

    # Text length statistics
    if "text" in df.columns:
        text_lengths = df["text"].str.len()
        stats["text_length"] = {
            "mean": float(text_lengths.mean()),
            "std": float(text_lengths.std()),
            "median": float(text_lengths.median()),
            "min": float(text_lengths.min()),
            "max": float(text_lengths.max()),
        }

    if "title" in df.columns:
        title_lengths = df["title"].str.len()
        stats["title_length"] = {
            "mean": float(title_lengths.mean()),
            "std": float(title_lengths.std()),
            "median": float(title_lengths.median()),
            "min": float(title_lengths.min()),
            "max": float(title_lengths.max()),
        }

    # Label distribution
    if "label" in df.columns:
        label_dist = df["label"].value_counts(normalize=True).to_dict()
        stats["label_distribution"] = {str(k): float(v) for k, v in label_dist.items()}

    logger.info("Baseline statistics computed: %s", list(stats.keys()))
    return stats


if __name__ == "__main__":
    config = load_config()
    df = load_data(config["data"]["raw_path"])
    stats = compute_baseline_statistics(df)
    print("Baseline Statistics:", stats)
