import pandas as pd  # type: ignore
import pytest  # type: ignore
from src.validate import validate_dataframe


def test_validate_dataframe_basic():
    df = pd.DataFrame({
        "num_feature": [1, 2],
        "target": [10, 20]
    })
    required_columns = ["num_feature", "target"]
    target_column = "target"

    result = validate_dataframe(df, required_columns, target_column)
    assert result is True


def test_validate_raises_on_empty_df():
    df = pd.DataFrame()
    with pytest.raises(ValueError):
        validate_dataframe(df, required_columns=["a"], target_column="a")


def test_validate_raises_on_missing_required_columns():
    df = pd.DataFrame({"a": [1], "popularity": [10]})
    with pytest.raises(ValueError):
        validate_dataframe(df,
                           required_columns=["a", "missing", "popularity"],
                           target_column="popularity")


def test_validate_raises_on_null_target():
    df = pd.DataFrame({"a": [1, 2], "popularity": [10, None]})
    with pytest.raises(ValueError):
        validate_dataframe(df,
                           required_columns=["a", "popularity"],
                           target_column="popularity")


def test_validate_raises_on_feature_nulls_when_disallowed():
    df = pd.DataFrame({"a": [1, None], "popularity": [10, 20]})
    with pytest.raises(ValueError):
        validate_dataframe(df,
                           required_columns=["a", "popularity"],
                           target_column="popularity",
                           allow_feature_nulls=False)


def test_validate_warns_on_feature_nulls_when_allowed(caplog):
    df = pd.DataFrame({"x": [1, None], "popularity": [10, 20]})
    required = ["x", "popularity"]

    validate_dataframe(df,
                       required,
                       target_column="popularity",
                       allow_feature_nulls=True)

    assert any("Nulls found in feature columns"
               in r.message for r in caplog.records)


def test_validate_raises_when_target_missing():
    df = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(ValueError):
        validate_dataframe(df,
                           required_columns=["x", "popularity"],
                           target_column="popularity")


def test_validate_warns_on_out_of_range_target(caplog):
    df = pd.DataFrame({"x": [1, 2], "popularity": [-1, 101]})
    validate_dataframe(df,
                       required_columns=["x", "popularity"],
                       target_column="popularity",
                       allow_feature_nulls=True)

    assert any("outside the 0-100 range" in r.message for r in caplog.records)


def test_validate_raises_when_target_missing_even_if_not_required():
    df = pd.DataFrame({"x": [1, 2]})

    # required_columns does NOT include 'popularity',
    # so we bypass the missing_cols check
    with pytest.raises(ValueError):
        validate_dataframe(df,
                           required_columns=["x"],
                           target_column="popularity")
