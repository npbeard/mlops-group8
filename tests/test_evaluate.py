import numpy as np
import pandas as pd  # type: ignore
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression  # type: ignore

from src.evaluate import evaluate_model


class DummyModel:
    def predict(self, X):
        return [0] * len(X)


class NaNPredictModel:
    def predict(self, X):
        return np.array([np.nan] * len(X))


def test_evaluate_regression_returns_rmse_mae():
    X = pd.DataFrame({"x": [1, 2, 3]})
    y = pd.Series([1.0, 2.0, 3.0])
    model = LinearRegression().fit(X, y)

    metrics = evaluate_model(model, X, y, "regression")
    assert "rmse" in metrics
    assert "mae" in metrics


def test_evaluate_classification_returns_f1():
    X = pd.DataFrame({"x": [0, 1, 2, 3]})
    y = pd.Series([0, 0, 1, 1])
    model = LogisticRegression(max_iter=200).fit(X, y)

    metrics = evaluate_model(model, X, y, "classification")
    assert "f1_weighted" in metrics


def test_evaluate_raises_for_invalid_problem_type():
    X = pd.DataFrame({"x": [1, 2, 3]})
    y = pd.Series([1.0, 2.0, 3.0])
    model = LinearRegression().fit(X, y)

    with pytest.raises(ValueError):
        evaluate_model(model, X, y, "clustering")


def test_evaluate_requires_predict():
    X = pd.DataFrame({"x": [1, 2, 3]})
    y = pd.Series([1.0, 2.0, 3.0])

    class NoPredict:
        pass

    with pytest.raises(TypeError):
        evaluate_model(NoPredict(), X, y, "regression")


def test_evaluate_raises_when_X_not_dataframe():
    y = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(TypeError):
        evaluate_model(DummyModel(), [1, 2, 3], y, "regression")  # type: ignore


def test_evaluate_raises_when_X_empty():
    X = pd.DataFrame({"x": []})
    y = pd.Series([], dtype=float)
    with pytest.raises(ValueError):
        evaluate_model(DummyModel(), X, y, "regression")


def test_evaluate_raises_when_length_mismatch():
    X = pd.DataFrame({"x": [1, 2, 3]})
    y = pd.Series([1.0, 2.0])  # mismatch
    with pytest.raises(ValueError):
        evaluate_model(DummyModel(), X, y, "regression")


def test_evaluate_raises_when_y_not_array_like():
    X = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(TypeError):
        evaluate_model(DummyModel(), X, y_eval=123, problem_type="regression")  # not array-like


def test_evaluate_raises_on_nan_predictions():
    X = pd.DataFrame({"x": [1, 2, 3]})
    y = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        evaluate_model(NaNPredictModel(), X, y, "regression")