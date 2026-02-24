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

def train_model(X_train: pd.DataFrame, y_train: pd.Series, preprocessor, problem_type: str):
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

    if problem_type == "regression":
        estimator = Ridge()
    else:
        estimator = LogisticRegression(max_iter=500)

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend the baseline
    # Why: Choosing the right algorithm (XGBoost, RandomForest) depends on data size and complexity.
    # Examples:
    # 1. estimator = RandomForestClassifier(n_estimators=100)
    # 2. Add hyperparameter tuning (GridSearchCV)
    if problem_type == "regression":
        # Improved Ridge (Good for interpretability)
        # estimator = Ridge(alpha=1.0) 

        # Random Forest (Better for handling 'sound archetypes' and non-linearity)
        estimator = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        
    else:
        # For classification (e.g., predicting 'is_hit'), RF is also superior
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    model_pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", estimator)
    ])

    model_pipeline.fit(X_train, y_train)
    return model_pipeline