"""
Module: Data Cleaning
---------------------
Role: Preprocessing, missing value imputation, and feature engineering.
Input: pandas.DataFrame (Raw).
Output: pandas.DataFrame (Processed/Clean).
"""

import pandas as pd

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
    print("Cleaning dataframe...") # TODO: replace with logging later
    df_clean = df_raw.copy()

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # 1. Remove duplicate tracks to prevent leakage (tracks appearing in both train/test)
    df_clean = df_raw.drop_duplicates(subset=['track_id'])

    # 2. Filter out "Non-Musical" content (Speechiness > 0.66 usually indicates spoken word)
    # This helps the model focus on actual songs as identified in Group 8 EDA.
    df_clean = df_clean[df_clean['speechiness'] < 0.66]

    # 3. Handle 0-length tracks if any exist
    df_clean = df_clean[df_clean['duration_ms'] > 0]

    # 4. Handle missing values in audio features
    # Spotify data is usually clean, but we fill numeric NaNs with the median just in case.
    numeric_cols = df_clean.select_dtypes(include=['number']).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())

    return df_clean
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------