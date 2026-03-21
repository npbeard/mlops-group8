import json
import logging
from pathlib import Path
from typing import Optional

import joblib  # type: ignore
import pandas as pd  # type: ignore
from src.logger import configure_logging

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    if not log_file:
        root = logging.getLogger()
        root.setLevel(level.upper())
        return

    configure_logging(log_level=level, log_file=log_file)


def load_csv(filepath: Path) -> pd.DataFrame:
    logger.info("Reading CSV from %s", filepath)

    if not isinstance(filepath, Path):
        raise TypeError(f"filepath must be a Path, got {type(filepath)}")

    if not filepath.exists():
        raise FileNotFoundError(f"CSV not found: {filepath}")

    try:
        df = pd.read_csv(filepath)
    except Exception:
        logger.exception("Failed to read CSV from %s", filepath)
        raise

    return df


def save_csv(df: pd.DataFrame, filepath: Path) -> None:
    """
    Inputs:
    - df: DataFrame to save.
    - filepath: Path object for destination.
    Outputs:
    - None
    Why this contract matters for reliable ML delivery:
    - Automatically handles directory creation to prevent pipeline crashes.
    """
    logger.info("Saving CSV to %s", filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(filepath, index=False)


def save_model(model, filepath: Path) -> None:
    """
    Inputs:
    - model: Trained scikit-learn model or pipeline.
    - filepath: Path object for destination.
    Outputs:
    - None
    Why this contract matters for reliable ML delivery:
    - Ensures models are versioned and stored in a consistent format.
    """
    logger.info("Saving model to %s", filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, filepath)


def save_json(obj: dict, filepath: Path) -> None:
    logger.info("Saving JSON to %s", filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_model(filepath: Path):
    logger.info("Loading model from %s", filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    return joblib.load(filepath)
