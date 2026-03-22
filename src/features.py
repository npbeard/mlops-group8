import logging
from inspect import signature
from typing import List, Optional

from sklearn.compose import ColumnTransformer  # type: ignore
from sklearn.preprocessing import (KBinsDiscretizer,  # type: ignore
                                   OneHotEncoder, StandardScaler)

logger = logging.getLogger(__name__)


def _build_kbins_discretizer(n_bins: int) -> KBinsDiscretizer:
    kwargs = {
        "n_bins": n_bins,
        "encode": "onehot-dense",
        "strategy": "quantile",
    }
    if "quantile_method" in signature(KBinsDiscretizer).parameters:
        kwargs["quantile_method"] = "linear"
    return KBinsDiscretizer(**kwargs)


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
    - Encapsulating logic in a ColumnTransformer prevents data
    leakage by ensuring the same rules apply to test/live data.
    """
    logger.info("Building feature preprocessor recipe...")

    transformers = []

    if quantile_bin_cols:
        # Good for non-linear features like 'tempo' or 'duration_ms'
        transformers.append(
            (
                "quantile_bins",
                _build_kbins_discretizer(n_bins),
                quantile_bin_cols
            )
        )

    if categorical_onehot_cols:
        # For 'key' or 'mode' if treated as categories
        transformers.append(
            (
                "cat_ohe",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_onehot_cols
            )
        )

    if numeric_passthrough_cols:
        # Standardize features like
        # 'danceability' and 'loudness'
        transformers.append(
            ("num_pass", StandardScaler(), numeric_passthrough_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")
