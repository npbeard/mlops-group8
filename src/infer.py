"""
Module: Inference
-----------------
Role: Make predictions on new, unseen data.
Input: Trained Model + New Data.
Output: Predictions (Array or DataFrame).
"""

import pandas as pd
import numpy as np

def run_inference(model, X_infer: pd.DataFrame) -> pd.DataFrame:
    """
    Inputs:
    - model: The fitted Pipeline.
    - X_infer: New features to predict on.
    Outputs:
    - pd.DataFrame: A DataFrame with a single column "prediction".
    Why this contract matters for reliable ML delivery:
    - Separation of training and inference ensures that production code is lightweight and focused.
    """
    print("Running inference on new data...") # TODO: replace with logging later
    
    preds = model.predict(X_infer)

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend the baseline
    # Why: You might need to post-process predictions (e.g., applying a threshold).
    # Examples:
    # 1. np.exp(preds) if you trained on log-target
    # 2. (preds > 0.8).astype(int)
    preds = np.clip(preds, 0, 100)

    # Rounding to the nearest integer as Spotify uses whole numbers
    preds = np.round(preds).astype(int)
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return pd.DataFrame({"prediction": preds}, index=X_infer.index)