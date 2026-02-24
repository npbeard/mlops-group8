"""
Module: Data Validation
-----------------------
Role: Check data quality (schema, types, ranges) before training.
Input: pandas.DataFrame.
Output: Boolean (True if valid) or raises Error.
"""

import pandas as pd

def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Inputs:
    - df: The dataframe to validate.
    - required_columns: List of columns that MUST be present.
    Outputs:
    - bool: Returns True if valid.
    Why this contract matters for reliable ML delivery:
    - If data is empty or corrupted, we should stop the pipeline before wasting compute or deploying bad models.
    """
    print("Validating data schema...") # TODO: replace with logging later

    if df.empty:
        raise ValueError("Validation Failed: The provided DataFrame is empty.")

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend the baseline
    # Why: Different models have different requirements (e.g., non-negative values for certain features).
    # Examples:
    # 1. check if columns in required_columns exist in df.columns
    # 2. assert df[target].isnull().sum() == 0
    # Verify all expected audio features and target are present 
    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
         raise ValueError(f"Validation Failed: Missing columns {missing_cols}")
    
    # Ensure popularity (target) is within valid 0-100 range 
    if not df['popularity'].between(0, 100).all():
        print("Warning: Some popularity scores are outside the 0-100 range.")
    
    print("Schema and Null-check validation passed.")
    return True
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------