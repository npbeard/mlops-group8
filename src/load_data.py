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
from src.utils import load_csv, save_csv

def load_raw_data(raw_data_path: Path) -> pd.DataFrame:
    """
    Inputs:
    - raw_data_path: Path to the raw data file.
    Outputs:
    - pd.DataFrame: The loaded raw dataset.
    Why this contract matters for reliable ML delivery:
    - Guarantees that the pipeline starts with a consistent data structure.
    """
    print(f"Loading raw data from {raw_data_path}") # TODO: replace with logging later
    
    if not raw_data_path.exists():
        print("!!! LOUD WARNING: Raw data file not found !!!")
        print("Creating a DUMMY dataset for scaffolding purposes only.")
        dummy_df = pd.DataFrame({
            "num_feature": [1.0, 2.5, 3.2, 4.8, 5.1, 0.5, 1.2, 3.3, 4.1, 2.9],
            "cat_feature": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"],
            "target": [10, 20, 15, 30, 25, 12, 28, 22, 14, 27]
        })
        save_csv(dummy_df, raw_data_path)
        print("Note: Update your SETTINGS in main.py once you replace this dummy data.")

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend the baseline
    # Why: You might need to merge multiple files or fetch data from a SQL database.
    # Examples:
    # 1. pd.concat([df1, df2])
    # 2. sql_engine.connect()
    return load_csv(raw_data_path)
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------