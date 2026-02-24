"""
Module: Evaluation
------------------
Role: Generate metrics and plots for model performance.
Input: Trained Model + Test Data.
Output: Metrics dictionary and plots saved to `reports/`.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, f1_score

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, problem_type: str) -> float:
    """
    Inputs:
    - model: Fitted Pipeline.
    - X_test, y_test: Evaluation data.
    - problem_type: 'regression' or 'classification'.
    Outputs:
    - float: The calculated metric (RMSE or F1).
    Why this contract matters for reliable ML delivery:
    - Consistent evaluation allows us to compare different model versions fairly.
    """
    print("Evaluating model performance...") # TODO: replace with logging later
    
    preds = model.predict(X_test)
    
    if problem_type == "regression":
        metric = np.sqrt(mean_squared_error(y_test, preds))
        print(f"Metric (RMSE): {metric}")
    else:
        metric = f1_score(y_test, preds, average='weighted')
        print(f"Metric (F1 Weighted): {metric}")

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend the baseline
    # Why: Business needs vary (e.g., minimizing False Negatives in fraud detection).
    # Examples:
    # 1. classification_report(y_test, preds)
    # 2. mean_absolute_percentage_error(y_test, preds)
    if problem_type == "regression":
        from sklearn.metrics import mean_absolute_error
        mae = mean_absolute_error(y_test, preds)
        print(f"Additional Metric (MAE): {mae}")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return float(metric)