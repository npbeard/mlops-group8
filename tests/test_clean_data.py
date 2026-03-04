import pandas as pd  # type: ignore
import pytest
from src.clean_data import clean_dataframe


def test_clean_dataframe_basic():
    # construct dummy DataFrame
    df = pd.DataFrame({
        "num_feature": [1, 2, 3, 3],
        "cat_feature": ["A", "B", "C", "C"],
        "speechiness": [0.1, 0.2, 0.5, 0.7],
        "duration_ms": [100, 200, 300, 400],
        "target": [10, 20, 30, 40]
    })

    # call clean_dataframe
    df_clean = clean_dataframe(df, target_column="target")

    assert isinstance(df_clean, pd.DataFrame)
    assert "speechiness" in df_clean.columns
    assert "duration_ms" in df_clean.columns
    assert len(df_clean) <= len(df)

def test_clean_dataframe_raises_if_speechiness_missing():
    df = pd.DataFrame({
        "duration_ms": [1000, 2000],
        "popularity": [10, 20],
    })
    with pytest.raises(KeyError):
        clean_dataframe(df, target_column="popularity")


def test_clean_dataframe_raises_if_duration_missing():
    df = pd.DataFrame({
        "speechiness": [0.1, 0.2],
        "popularity": [10, 20],
    })
    with pytest.raises(KeyError):
        clean_dataframe(df, target_column="popularity")