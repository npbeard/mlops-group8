"""
Module: Data Cleaning
---------------------
Role: Preprocessing, missing value imputation, and feature engineering.
Input: pandas.DataFrame (Raw).
Output: pandas.DataFrame (Processed/Clean).
"""

import pandas as pd
import logging
logger = logging.getLogger(__name__)

def clean_dataframe(df_raw: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Inputs:
    - df_raw: The raw input DataFrame.
    - target_column: The name of the label/target.
    Outputs:
    - pd.DataFrame: Cleaned data ready for validation.
    Why this contract matters for reliable ML delivery:
    - Prevents downstream errors caused by missing values or duplicate rows.
    """
    logger.info("Cleaning dataframe...")
    df_clean = df_raw.copy()

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # 1. Remove duplicate tracks to prevent leakage (tracks appearing in both train/test)
    df_clean = df_clean.drop_duplicates()

    # 2. Filter out "Non-Musical" content (Speechiness > 0.66 usually indicates spoken word)
    # This helps the model focus on actual songs as identified in Group 8 EDA.
    if "speechiness" not in df_clean.columns:
        raise KeyError("Expected column 'speechiness' not found in input dataframe.")
    df_clean = df_clean[df_clean["speechiness"] < 0.66]

    # 3. Handle 0-length tracks if any exist
    if "duration_ms" not in df_clean.columns:
        raise KeyError("Expected column 'duration_ms' not found in input dataframe.")
    df_clean = df_clean[df_clean["duration_ms"] > 0]

    return df_clean
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------