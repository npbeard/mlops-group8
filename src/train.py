"""
Module: Model Training
----------------------
Role: Split data, train model, and save the artifact.
Input: pandas.DataFrame (Processed).
Output: Serialized model file (e.g., .pkl) in `models/`.
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def train_model(X_train: pd.DataFrame, y_train: pd.Series, preprocessor, problem_type: str, train_config: dict | None = None):
    """
    Inputs:
    - X_train, y_train: Training features and target.
    - preprocessor: The ColumnTransformer recipe.
    - problem_type: 'regression' or 'classification'.
    Outputs:
    - Pipeline: A fitted scikit-learn Pipeline object.
    Why this contract matters for reliable ML delivery:
    - Using a Pipeline ensures that preprocessing and modeling are bundled into a single atomic object.
    """
    print(f"Training {problem_type} model...") # TODO: replace with logging later

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend the baseline
    # Why: Choosing the right algorithm (XGBoost, RandomForest) depends on data size and complexity.
    # Examples:
    # 1. estimator = RandomForestClassifier(n_estimators=100)
    # 2. Add hyperparameter tuning (GridSearchCV)
    train_config = train_config or {}

    seed = train_config.get("seed", 42)
    rf_n_estimators = train_config.get("rf_n_estimators", 100)
    rf_max_depth = train_config.get("rf_max_depth", 10)
    baseline_model = train_config.get("baseline_model", "rf")  # optional

    # Choose model
    if baseline_model == "ridge" and problem_type == "regression":
        estimator = Ridge(alpha=float(train_config.get("ridge_alpha", 1.0)))
    elif baseline_model == "logreg" and problem_type != "regression":
        estimator = LogisticRegression(max_iter=int(train_config.get("logreg_max_iter", 500)))
    else:
        # Default: Random Forest
        if problem_type == "regression":
            estimator = RandomForestRegressor(
                n_estimators=int(rf_n_estimators),
                max_depth=int(rf_max_depth) if rf_max_depth is not None else None,
                random_state=int(seed),
                n_jobs=-1,
            )
        else:
            estimator = RandomForestClassifier(
                n_estimators=int(rf_n_estimators),
                max_depth=int(rf_max_depth) if rf_max_depth is not None else None,
                random_state=int(seed),
                n_jobs=-1,
            )
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    model_pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", estimator)
    ])

    model_pipeline.fit(X_train, y_train)
    return model_pipeline