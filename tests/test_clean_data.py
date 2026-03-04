import pandas as pd  # type: ignore

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
