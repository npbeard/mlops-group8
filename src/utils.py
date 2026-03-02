import logging
from pathlib import Path
from typing import Optional
import pandas as pd
import joblib
from pathlib import Path
import json
from pathlib import Path

logger = logging.getLogger(__name__)

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())

    if root.handlers:
        return

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        root.addHandler(fh)

def load_csv(filepath: Path) -> pd.DataFrame:
    """
    Inputs:
    - filepath: Path object pointing to a CSV file.
    Outputs:
    - pd.DataFrame: Loaded data.
    Why this contract matters for reliable ML delivery:
    - Provides a single point of failure and fix for data ingestion issues.
    """
    logger.info("Reading CSV from %s", filepath)
    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Adjust read parameters (sep, encoding, index_col)
    return pd.read_csv(filepath)
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

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
    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Adjust save parameters (compression, index)
    df.to_csv(filepath, index=False)
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

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
    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Adjust compression or serialization library
    joblib.dump(model, filepath)
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

def load_model(filepath: Path):
    """
    Inputs:
    - filepath: Path object to a serialized model.
    Outputs:
    - model: Loaded Python object.
    Why this contract matters for reliable ML delivery:
    - Simplifies loading for inference without repeating code.
    """
    logger.info("Loading model from %s", filepath)
    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Add error handling for missing files
    return joblib.load(filepath)
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

def save_json(obj: dict, filepath: Path) -> None:
    logger.info("Saving JSON to %s", filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)