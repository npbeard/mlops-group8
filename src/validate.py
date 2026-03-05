"""
Module: Data Validation
-----------------------
Role: Check data quality (schema, types, ranges) before training.
Input: pandas.DataFrame.
Output: Boolean (True if valid) or raises Error.
"""

import logging
import pandas as pd  # type: ignore

logger = logging.getLogger(__name__)


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list,
    target_column: str,
    allow_feature_nulls: bool = True
) -> bool:
    """
    Inputs:
    - df: The dataframe to validate.
    - required_columns: List of columns that MUST be present.
    Outputs:
    - bool: Returns True if valid.
    Why this contract matters for reliable ML delivery:
    - If data is empty or corrupted, we should stop the pipeline
    before wasting compute or deploying bad models.
    """
    logger.info("Validating data schema...")

    if df.empty:
        raise ValueError("Validation Failed: The provided DataFrame is empty.")

    if (missing_cols := [c for c in required_columns if c not in df.columns]):
        raise ValueError(f"Validation Failed: Missing columns {missing_cols}")

    # Null checks
    if target_column not in df.columns:
        raise ValueError(
            (
                f"Validation Failed: target_column '{target_column}' "
                "not found in df."
            )
        )

    if df[target_column].isnull().any():
        n_null = int(df[target_column].isnull().sum())
        raise ValueError(
            (
                f"Validation Failed: Target column '{target_column}' "
                f"contains {n_null} null values."
            )
        )

    feature_cols = [c for c in required_columns if c != target_column]
    feature_null_cols = [c for c in feature_cols if df[c].isnull().any()]

    if feature_null_cols:
        if allow_feature_nulls:
            # Pipeline imputes; warn but don't fail
            null_counts = {
                c: int(df[c].isnull().sum())
                for c in feature_null_cols
            }
            logger.warning(
                (
                    "Nulls found in feature columns "
                    "(will be imputed in Pipeline): %s"
                ),
                null_counts,
            )
        else:
            raise ValueError(
                (
                    f"Validation Failed: Nulls found in feature columns "
                    f"{feature_null_cols}"
                )
            )

    # Basic target sanity check (if numeric)
    if (
        pd.api.types.is_numeric_dtype(df[target_column])
        and not df[target_column].between(0, 100).all()
    ):
        logger.warning(
            "Some '%s' values are outside the 0-100 range.", target_column)

    logger.info("Schema and null-check validation passed.")
    return True
