import pytest
import pandas as pd
from sklearn.linear_model import LinearRegression
from src.infer import run_inference

def test_run_inference_returns_dataframe():
    X_infer = pd.DataFrame({"num_feature": [1,2,3]})
    model = LinearRegression().fit(X_infer, [10,20,30])
    preds = run_inference(model, X_infer)
    
    assert isinstance(preds, pd.DataFrame)
    assert "prediction" in preds.columns
    assert len(preds) == len(X_infer)