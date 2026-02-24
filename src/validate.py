"""
Module: Data Validation
-----------------------
Role: Check data quality (schema, types, ranges) before training.
Input: pandas.DataFrame.
Output: Boolean (True if valid) or raises Error.
"""

import pandas as pd

def validate_dataframe(df: pd.DataFrame, required_columns: list, target_column: str, allow_feature_nulls: bool = True) -> bool:
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
    
    # Check required columns
    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Validation Failed: Missing columns {missing_cols}")

    # Null checks
    if target_column not in df.columns:
        raise ValueError(f"Validation Failed: target_column '{target_column}' not found in df.")

    if df[target_column].isnull().any():
        n_null = int(df[target_column].isnull().sum())
        raise ValueError(f"Validation Failed: Target column '{target_column}' contains {n_null} null values.")

    feature_cols = [c for c in required_columns if c != target_column]
    feature_null_cols = [c for c in feature_cols if df[c].isnull().any()]

    if feature_null_cols:
        if allow_feature_nulls:
            # Pipeline imputes; warn but don't fail
            null_counts = {c: int(df[c].isnull().sum()) for c in feature_null_cols}
            print(f"Warning: Nulls found in feature columns (will be imputed in Pipeline): {null_counts}")
        else:
            raise ValueError(f"Validation Failed: Nulls found in feature columns {feature_null_cols}")

    # Basic target sanity check (if numeric)
    if pd.api.types.is_numeric_dtype(df[target_column]):
        if not df[target_column].between(0, 100).all():
            print(f"Warning: Some '{target_column}' values are outside the 0-100 range.")
    
    print("Schema and Null-check validation passed.")
    return True
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------