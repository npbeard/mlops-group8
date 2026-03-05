"""
Module: Evaluation
------------------
Role: Generate metrics and plots for model performance.
Input: Trained Model + Test Data.
Output: Metrics dictionary and plots saved to `reports/`.
"""
import logging
from typing import Any, Dict

import numpy as np
import pandas as pd  # type: ignore
from sklearn.metrics import f1_score  # type: ignore
from sklearn.metrics import mean_absolute_error  # type: ignore
from sklearn.metrics import mean_squared_error  # type: ignore

logger = logging.getLogger(__name__)


def evaluate_model(
    model: Any,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    problem_type: str,
) -> Dict[str, float]:
    logger.info("Evaluating model performance...")

    # contract checks
    if not hasattr(model, "predict"):
        raise TypeError("Model artifact must implement a .predict() method")

    if not isinstance(X_eval, pd.DataFrame):
        raise TypeError(
            f"X_eval must be a pandas.DataFrame, got {type(X_eval)}")

    if not isinstance(y_eval, (pd.Series, pd.DataFrame, np.ndarray, list)):
        raise TypeError(f"y_eval must be array-like, got {type(y_eval)}")

    if X_eval.empty:
        raise ValueError("X_eval is empty; cannot evaluate")

    if len(X_eval) != len(y_eval):
        raise ValueError(
            f"X_eval and y_eval length mismatch:"
            f"{len(X_eval)} vs {len(y_eval)}"
        )

    if problem_type not in {"regression", "classification"}:
        raise ValueError(
            f"problem_type must be 'regression' or 'classification',"
            f"got: {problem_type}"
        )

    preds = model.predict(X_eval)

    # optional sanity check
    if np.any(pd.isna(preds)):
        raise ValueError("Predictions contain NaN; cannot evaluate")

    # metrics
    if problem_type == "regression":
        rmse = float(np.sqrt(mean_squared_error(y_eval, preds)))
        mae = float(mean_absolute_error(y_eval, preds))
        logger.info("Metrics: rmse=%.6f mae=%.6f", rmse, mae)
        return {"rmse": rmse, "mae": mae}

    f1 = float(f1_score(y_eval, preds, average="weighted"))
    logger.info("Metrics: f1_weighted=%.6f", f1)
    return {"f1_weighted": f1}
