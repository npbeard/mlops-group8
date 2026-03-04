import pandas as pd  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore

from src.infer import run_inference


def test_run_inference_returns_dataframe():
    X_infer = pd.DataFrame({"num_feature": [1, 2, 3]})
    model = LinearRegression().fit(X_infer, [10, 2030])
    preds = run_inference(model, X_infer)

    assert isinstance(preds, pd.DataFrame)
    assert "prediction" in preds.columns
    assert len(preds) == len(X_infer)
