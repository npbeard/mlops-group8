import pytest
import pandas as pd
from src.features import get_feature_preprocessor

def test_get_feature_preprocessor_minimal():
    # construct dummy DataFrame
    df = pd.DataFrame({
        "num_feature1": [1, 2, 3],
        "cat_feature1": ["A", "B", "C"]
    })

    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=["num_feature1"],
        categorical_onehot_cols=["cat_feature1"],
        numeric_passthrough_cols=None,
        n_bins=2
    )

    from sklearn.compose import ColumnTransformer
    assert isinstance(preprocessor, ColumnTransformer)
    assert len(preprocessor.transformers) > 0