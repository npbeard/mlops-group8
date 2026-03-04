"""
Module: Evaluation
------------------
Role: Generate metrics and plots for model performance.
Input: Trained Model + Test Data.
Output: Metrics dictionary and plots saved to `reports/`.
"""
import logging

import numpy as np
import pandas as pd  # type: ignore
from sklearn.metrics import (  # pyright: ignore[reportMissingModuleSource]
    f1_score, mean_absolute_error, mean_squared_error)

logger = logging.getLogger(__name__)


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    problem_type: str
) -> float:
    """
    Inputs:
    - model: Fitted Pipeline.
    - X_test, y_test: Evaluation data.
    - problem_type: 'regression' or 'classification'.
    Outputs:
    - float: The calculated metric (RMSE or F1).
    Why this contract matters for reliable ML delivery:
    - Consistent evaluation allows us to
    compare different model versions fairly.
    """
    logger.info("Evaluating model performance...")

    preds = model.predict(X_test)

    # ---------------------
    # START STUDENT CODE
    # ---------------------
    # TODO_STUDENT:
    # Paste your notebook logic here to replace or extend the baseline
    # Why: Business needs vary
    # (e.g., minimizing False Negatives in fraud detection).
    # Examples:
    # 1. classification_report(y_test, preds)
    # 2. mean_absolute_percentage_error(y_test, preds)
    if problem_type == "regression":
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        logger.info("Metric (RMSE): %.6f", rmse)
        logger.info("Additional Metric (MAE): %.6f", mae)
        return {"rmse": rmse, "mae": mae}

    # classification
    f1 = float(f1_score(y_test, preds, average="weighted"))
    logger.info("Metric (F1 Weighted): %.6f", f1)
    return {"f1_weighted": f1}
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    # no fallback metric; the appropriate value is returned above
    # for regression or classification, so this return is unnecessary.
