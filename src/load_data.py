"""
Module: Data Loader
-------------------
Role: Ingest raw data from sources (CSV, SQL, API).
Input: Path to file or connection string.
Output: pandas.DataFrame (Raw).
"""
"""
Educational Goal:
- Why this module exists in an MLOps system: Decouples data retrieval from processing.
- Responsibility (separation of concerns): Fetches raw data from source.
- Pipeline contract (inputs and outputs): Takes a path; returns a raw DataFrame.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""


import pandas as pd
from pathlib import Path
import logging
from src.utils import load_csv, save_csv

logging.basicConfig(
    filename="mlops.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

logger = logging.getLogger(__name__)

def load_raw_data(raw_data_path: Path) -> pd.DataFrame:
    logger.info(f"Attempting to load raw data from {raw_data_path}")

    try:
        if not raw_data_path.exists():
            logger.warning("Raw data file not found. Creating dummy dataset.")

            dummy_df = pd.DataFrame({
                "num_feature": [1.0, 2.5, 3.2, 4.8, 5.1, 0.5, 1.2, 3.3, 4.1, 2.9],
                "cat_feature": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"],
                "target": [10, 20, 15, 30, 25, 12, 28, 22, 14, 27]
            })

            save_csv(dummy_df, raw_data_path)
            logger.info("Dummy dataset created and saved.")

        data = load_csv(raw_data_path)
        logger.info("Raw data loaded successfully.")

        return data

    except Exception as e:
        logger.error(f"Error occurred while loading raw data: {e}")
        raise