import pandas as pd  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore

from src.evaluate import evaluate_model


def test_evaluate_model_regression_basic():
    # constuct dummy data for regression
    X_test = pd.DataFrame({"feature": [1, 2, 3]})
    y_test = pd.Series([10, 20, 30])

    # simple linear regression model
    model = LinearRegression().fit(X_test, y_test)

    metrics = evaluate_model(model, X_test, y_test, problem_type="regression")

    assert isinstance(metrics, dict)
    assert "rmse" in metrics
    assert "mae" in metrics
