import pytest
import pandas as pd
from pathlib import Path
from src.utils import load_csv, save_csv, save_json, load_model, save_model
from sklearn.linear_model import LinearRegression

def test_save_and_load_csv(tmp_path):
    df = pd.DataFrame({"col":[1,2,3]})
    file_path = tmp_path / "test.csv"
    save_csv(df, file_path)
    df2 = load_csv(file_path)
    pd.testing.assert_frame_equal(df, df2)

def test_save_and_load_model(tmp_path):
    model = LinearRegression()
    file_path = tmp_path / "model.joblib"
    save_model(model, file_path)
    loaded = load_model(file_path)
    assert hasattr(loaded, "predict")

def test_save_json(tmp_path):
    obj = {"a": 1}
    file_path = tmp_path / "test.json"
    save_json(obj, file_path)
    assert file_path.exists()