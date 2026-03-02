import pytest
import pandas as pd
from src.validate import validate_dataframe

def test_validate_dataframe_basic():
    df = pd.DataFrame({
        "num_feature": [1,2],
        "target": [10,20]
    })
    required_columns = ["num_feature", "target"]
    target_column = "target"

    result = validate_dataframe(df, required_columns, target_column)
    assert result is True