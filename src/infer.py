"""
Module: Inference
-----------------
Role: Make predictions on new, unseen data.
Input: Trained Model + New Data.
Output: Predictions (Array or DataFrame).
"""
import logging
import pandas as pd  # type: ignore
from typing import Any

logger = logging.getLogger(__name__)


def run_inference(model: Any, X_infer: pd.DataFrame) -> pd.DataFrame:
    """
    Inputs:
    - model: The fitted Pipeline / estimator implementing .predict()
    - X_infer: New features to predict on.
    Outputs:
    - pd.DataFrame: A DataFrame with a single column "prediction".
    """
    logger.info("Running inference on new data...")

    if not hasattr(model, "predict"):
        raise TypeError("Model artifact must implement a .predict() method")

    if not isinstance(X_infer, pd.DataFrame):
        raise TypeError(f"X_infer must be a pandas.DataFrame,"
                        f"got {type(X_infer)}")

    if X_infer.empty:
        raise ValueError("X_infer is empty; cannot run inference")

    preds = model.predict(X_infer)

    df_preds = pd.DataFrame({"prediction": preds})
    logger.info("Inference complete. Generated %d predictions.", len(df_preds))
    return df_preds
