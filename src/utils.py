"""
Educational Goal:
- Why this module exists in an MLOps system: Centralizes I/O logic to ensure consistent data handling across the pipeline.
- Responsibility (separation of concerns): Handles reading/writing files and model serialization.
- Pipeline contract (inputs and outputs): Takes file paths and objects; returns loaded data or saves to disk.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""
import pandas as pd
import joblib
from pathlib import Path

def load_csv(filepath: Path) -> pd.DataFrame:
    """
    Inputs:
    - filepath: Path object pointing to a CSV file.
    Outputs:
    - pd.DataFrame: Loaded data.
    Why this contract matters for reliable ML delivery:
    - Provides a single point of failure and fix for data ingestion issues.
    """
    print(f"Reading CSV from {filepath}") # TODO: replace with logging later
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
    print(f"Saving CSV to {filepath}") # TODO: replace with logging later
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
    print(f"Saving model to {filepath}") # TODO: replace with logging later
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
    print(f"Loading model from {filepath}") # TODO: replace with logging later
    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Add error handling for missing files
    return joblib.load(filepath)
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------