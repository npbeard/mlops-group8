import pandas as pd
from sklearn.linear_model import LinearRegression
import pytest

from src.infer import run_inference


def test_run_inference_returns_dataframe():
    X_infer = pd.DataFrame({"num_feature": [1, 2, 3]})
    y = [10, 20, 30]  # length matches X_infer

    model = LinearRegression().fit(X_infer, y)

    df_preds = run_inference(model, X_infer)

    assert isinstance(df_preds, pd.DataFrame)
    assert "prediction" in df_preds.columns
    assert len(df_preds) == len(X_infer)


def test_run_inference_requires_predict():
    X_infer = pd.DataFrame({"num_feature": [1, 2, 3]})

    class NoPredict:
        pass

    with pytest.raises(TypeError):
        run_inference(NoPredict(), X_infer)

def test_run_inference_raises_on_empty_df():
    X = pd.DataFrame({"x": []})
    model = LinearRegression()
    X_fit = pd.DataFrame({"x": [1, 2]})
    y_fit = [1, 2]
    model.fit(X_fit, y_fit)

    with pytest.raises(ValueError):
        run_inference(model, X)


def test_run_inference_raises_on_non_dataframe():
    X_fit = pd.DataFrame({"x": [1, 2]})
    y_fit = [1, 2]
    model = LinearRegression().fit(X_fit, y_fit)

    with pytest.raises(TypeError):
        run_inference(model, [1, 2, 3])  # not a DataFrame