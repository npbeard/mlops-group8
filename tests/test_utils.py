import pandas as pd  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from pathlib import Path
import pytest
import logging
import json
import joblib  # type: ignore
from src.utils import load_csv, load_model, save_csv, save_json, save_model, setup_logging


def test_save_and_load_csv(tmp_path):
    df = pd.DataFrame({"col": [1, 2, 3]})
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

def test_load_model_raises_if_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nope.joblib")

def test_setup_logging_idempotent():
    setup_logging(level="INFO", log_file=None)
    setup_logging(level="INFO", log_file=None)  # should not add duplicate handlers

def test_save_json_writes_file(tmp_path):
    p = tmp_path / "out.json"
    save_json({"a": 1}, p)
    assert p.exists()

def test_setup_logging_creates_file_handler(tmp_path, monkeypatch):
    # Reset root logger handlers so we can test the handler creation branch
    root = logging.getLogger()
    old_handlers = root.handlers[:]
    root.handlers.clear()

    log_path = tmp_path / "logs" / "test.log"
    setup_logging(level="INFO", log_file=str(log_path))

    assert log_path.exists()  # file handler should have created the file's parent dir

    # Restore handlers so we don't affect other tests
    root.handlers = old_handlers


def test_setup_logging_returns_if_handlers_exist(monkeypatch):
    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(logging.StreamHandler())

    n_before = len(root.handlers)
    setup_logging(level="INFO", log_file=None)
    n_after = len(root.handlers)

    assert n_after == n_before

def test_save_json_writes_content(tmp_path):
    p = tmp_path / "out.json"
    save_json({"a": 1}, p)

    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)

    assert obj == {"a": 1}

def test_save_json_roundtrip(tmp_path):
    p = tmp_path / "out.json"
    save_json({"a": 1}, p)

    with open(p, "r", encoding="utf-8") as f:
        assert json.load(f) == {"a": 1}

def test_load_model_loads_roundtrip(tmp_path):
    # Create a tiny object and save it using joblib directly
    p = tmp_path / "model.joblib"
    obj = {"hello": "world"}
    joblib.dump(obj, p)

    loaded = load_model(p)
    assert loaded == obj


def test_load_model_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "missing_model.joblib")

def test_load_csv_raises_if_not_path():
    with pytest.raises(TypeError):
        load_csv("not_a_path")  # type: ignore

def test_load_csv_raises_if_missing(tmp_path):
    missing = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        load_csv(missing)