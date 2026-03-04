import pandas as pd  # type: ignore
from sklearn.compose import ColumnTransformer  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from src.train import train_model


def test_train_model_minimal():
    X_train = pd.DataFrame({"num_feature": [1, 2, 3]})
    y_train = pd.Series([10, 20, 30])
    preprocessor = ColumnTransformer([
        ("num_pass", StandardScaler(), ["num_feature"])])

    model = train_model(
        X_train, y_train, preprocessor, problem_type="regression")

    assert hasattr(model, "predict")
