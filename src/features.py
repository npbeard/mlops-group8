from typing import Optional, List
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, StandardScaler

def get_feature_preprocessor(
    quantile_bin_cols: Optional[List[str]] = None,
    categorical_onehot_cols: Optional[List[str]] = None,
    numeric_passthrough_cols: Optional[List[str]] = None,
    n_bins: int = 3
):
    """
    Inputs:
    - quantile_bin_cols: Numeric columns to discretize.
    - categorical_onehot_cols: Categorical columns to encode.
    - numeric_passthrough_cols: Numeric columns to keep as-is.
    Outputs:
    - ColumnTransformer: A scikit-learn transformation recipe.
    Why this contract matters for reliable ML delivery:
    - Encapsulating logic in a ColumnTransformer prevents data leakage by ensuring the same rules apply to test/live data.
    """
    print("Building feature preprocessor recipe...") # TODO: replace with logging later

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend the baseline
    # Why: Different datasets need different encoding (e.g., Target Encoding for high-cardinality features).
    # Examples:
    # 1. SimpleImputer() for missing values
    # 2. FunctionTransformer() for custom log transforms
    transformers = []

    if quantile_bin_cols:
        # Good for non-linear features like 'tempo' or 'duration_ms' 
        transformers.append(
            (
                "quantile_bins", 
                KBinsDiscretizer(
                    n_bins=n_bins, 
                    encode="ordinal", 
                    strategy='quantile',
                    quantile_method='linear'
                ), 
                quantile_bin_cols
            )
        )

    if categorical_onehot_cols:
        # For 'key' or 'mode' if treated as categories 
        transformers.append(("cat_ohe", OneHotEncoder(handle_unknown="ignore"), categorical_onehot_cols))

    if numeric_passthrough_cols:
        # Standardize features like 'danceability' and 'loudness'
        transformers.append(("num_pass", StandardScaler(), numeric_passthrough_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    # return ColumnTransformer(transformers=transformers, remainder="drop")