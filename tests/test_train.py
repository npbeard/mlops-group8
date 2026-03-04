import pandas as pd  # type: ignore
from sklearn.compose import ColumnTransformer  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
import pytest
from src.train import train_model
from src.features import get_feature_preprocessor
from sklearn.datasets import make_classification


def test_train_model_minimal():
    X_train = pd.DataFrame({"num_feature": [1, 2, 3]})
    y_train = pd.Series([10, 20, 30])
    preprocessor = ColumnTransformer([
        ("num_pass", StandardScaler(), ["num_feature"])])

    model = train_model(
        X_train, y_train, preprocessor, problem_type="regression")

    assert hasattr(model, "predict")

def test_train_model_raises_on_invalid_problem_type():
    X = pd.DataFrame({"x": [1, 2, 3]})
    y = pd.Series([1, 2, 3])
    pre = get_feature_preprocessor([], [], ["x"], n_bins=3)

    with pytest.raises(ValueError):
        train_model(X, y, pre, "clustering", train_config={})

def test_train_model_classification_runs():
    X_np, y_np = make_classification(
        n_samples=20,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        random_state=0,
    )
    X = pd.DataFrame(X_np, columns=["x1", "x2"])
    y = pd.Series(y_np)

    pre = get_feature_preprocessor([], [], ["x1", "x2"], n_bins=3)

    model = train_model(X, y, pre, "classification", train_config={})
    assert hasattr(model, "predict")

def test_train_model_uses_defaults_when_config_missing():
    X = pd.DataFrame({"x": [1, 2, 3, 4]})
    y = pd.Series([10, 20, 15, 25])
    pre = get_feature_preprocessor([], [], ["x"], n_bins=3)

    model = train_model(X, y, pre, "regression", train_config={})
    assert hasattr(model, "predict")

def test_train_model_uses_default_config_when_none():
    X = pd.DataFrame({
        "speechiness": [0.1, 0.2, 0.3, 0.4],
        "duration_ms": [1000, 2000, 1500, 1200],
    })
    y = pd.Series([10, 20, 15, 25])

    pre = get_feature_preprocessor(
        quantile_bin_cols=["duration_ms"],
        categorical_onehot_cols=[],
        numeric_passthrough_cols=["speechiness"],
        n_bins=3,
    )

    model = train_model(X, y, pre, "regression", train_config=None)
    assert hasattr(model, "predict")

def test_train_model_defaults_branch_none_config():
    X = pd.DataFrame({"speechiness": [0.1, 0.2], "duration_ms": [1000, 2000]})
    y = pd.Series([10, 20])

    pre = get_feature_preprocessor(["duration_ms"], [], ["speechiness"], n_bins=2)

    model = train_model(X, y, pre, "regression", train_config=None)
    assert hasattr(model, "predict")

def test_train_model_none_config_hits_defaults():
    X = pd.DataFrame({"speechiness": [0.1, 0.2], "duration_ms": [1000, 2000]})
    y = pd.Series([10, 20])
    pre = get_feature_preprocessor(["duration_ms"], [], ["speechiness"], n_bins=2)

    model = train_model(X, y, pre, "regression", train_config=None)
    assert hasattr(model, "predict")

def test_train_model_ridge_branch_runs():
    X = pd.DataFrame({"speechiness": [0.1, 0.2, 0.3], "duration_ms": [1000, 2000, 1500]})
    y = pd.Series([10.0, 20.0, 15.0])

    pre = get_feature_preprocessor(["duration_ms"], [], ["speechiness"], n_bins=2)

    model = train_model(
        X, y, pre,
        problem_type="regression",
        train_config={"baseline_model": "ridge", "ridge_alpha": 1.0},
    )
    assert hasattr(model, "predict")


def test_train_model_logreg_branch_runs():
    X = pd.DataFrame({"speechiness": [0.1, 0.2, 0.3, 0.4], "duration_ms": [1000, 2000, 1500, 1200]})
    y = pd.Series([0, 1, 0, 1])

    pre = get_feature_preprocessor(["duration_ms"], [], ["speechiness"], n_bins=2)

    model = train_model(
        X, y, pre,
        problem_type="classification",
        train_config={"baseline_model": "logreg", "logreg_max_iter": 200},
    )
    assert hasattr(model, "predict")