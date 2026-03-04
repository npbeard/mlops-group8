"""
Module: Data Loader
-------------------
Role: Ingest raw data from sources (CSV, SQL, API).
Input: Path to file or connection string.
Output: pandas.DataFrame (Raw).

Educational Goal:
- Why this module exists in an MLOps system:
    Decouples data retrieval from processing.
- Responsibility (separation of concerns):
    Fetches raw data from source.
- Pipeline contract (inputs and outputs):
    Takes a path; returns a raw DataFrame.

TO_DO: Replace print statements with
standard library logging in a later session
TO_DO: Any temporary or hardcoded variable or parameter
will be imported from config.yml in a later session

"""

import logging
from pathlib import Path

import pandas as pd  # type: ignore

from src.utils import load_csv

logger = logging.getLogger(__name__)


def load_raw_data(raw_data_path: Path) -> pd.DataFrame:
    """
    Load raw data from a CSV file path.

    Contract:
      - Input: raw_data_path (Path) must exist and be a readable CSV.
      - Output: non-empty pandas.DataFrame with raw data.
    """
    logger.info("Attempting to load raw data from %s", raw_data_path)

    if not isinstance(raw_data_path, Path):
        raise TypeError(f"raw_data_path must be a Path, got {type(raw_data_path)}")

    if not raw_data_path.exists():
        msg = (
            f"Raw data file not found at: {raw_data_path}\n"
            "Expected a CSV at this location. "
            "Place the raw dataset there or update config.yaml (paths.raw_data)."
        )
        logger.error(msg)
        raise FileNotFoundError(msg)

    if raw_data_path.suffix.lower() != ".csv":
        msg = f"Raw data path must be a .csv file, got: {raw_data_path.name}"
        logger.error(msg)
        raise ValueError(msg)

    try:
        df = load_csv(raw_data_path)
    except Exception:
        logger.exception("Failed to read raw data from %s", raw_data_path)
        raise

    if df is None or df.empty:
        msg = f"Loaded raw data is empty: {raw_data_path}"
        logger.error(msg)
        raise ValueError(msg)

    logger.info("Raw data loaded successfully. shape=%s", df.shape)
    return df
