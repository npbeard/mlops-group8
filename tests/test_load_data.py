import pytest  # type: ignore
from src.load_data import load_raw_data
import pandas as pd  # type: ignore
import src.load_data as load_data_mod


def test_load_raw_data_raises_if_missing(tmp_path):
    test_file = tmp_path / "test.csv"

    with pytest.raises(FileNotFoundError):
        load_raw_data(test_file)


def test_load_raw_data_reads_csv(tmp_path):
    test_file = tmp_path / "test.csv"
    df_in = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df_in.to_csv(test_file, index=False)

    df_out = load_raw_data(test_file)

    assert df_out.shape == (2, 2)
    assert list(df_out.columns) == ["a", "b"]


def test_load_raw_data_rejects_non_csv(tmp_path):
    p = tmp_path / "data.txt"
    p.write_text("not a csv")
    with pytest.raises(ValueError):
        load_raw_data(p)


def test_load_raw_data_raises_on_empty_csv(tmp_path):
    p = tmp_path / "empty.csv"
    pd.DataFrame().to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_raw_data(p)


def test_load_raw_data_logs_and_raises_on_read_error(tmp_path, monkeypatch):
    p = tmp_path / "data.csv"
    p.write_text("a,b\n1,2\n")  # exists

    def boom(_):
        raise RuntimeError("read failed")
    monkeypatch.setattr(load_data_mod, "load_csv", boom)
    with pytest.raises(RuntimeError):
        load_data_mod.load_raw_data(p)


def test_load_raw_data_raises_on_load_csv_error(tmp_path, monkeypatch):
    p = tmp_path / "data.csv"
    p.write_text("a,b\n1,2\n")

    def boom(_):
        raise RuntimeError("boom")

    monkeypatch.setattr(load_data_mod, "load_csv", boom)

    with pytest.raises(RuntimeError):
        load_data_mod.load_raw_data(p)


def test_load_raw_data_raises_on_load_csv_exception(tmp_path, monkeypatch):
    p = tmp_path / "data.csv"
    p.write_text("a,b\n1,2\n")  # exists and is .csv, so we reach load_csv

    def boom(_):
        raise RuntimeError("boom")

    monkeypatch.setattr(load_data_mod, "load_csv", boom)

    with pytest.raises(RuntimeError):
        load_data_mod.load_raw_data(p)


def test_load_raw_data_raises_on_empty_dataframe(tmp_path, monkeypatch):
    p = tmp_path / "data.csv"
    p.write_text("a,b\n1,2\n")

    monkeypatch.setattr(load_data_mod, "load_csv", lambda _: pd.DataFrame())

    with pytest.raises(ValueError):
        load_data_mod.load_raw_data(p)


def test_load_raw_data_requires_path_type():
    with pytest.raises(TypeError):
        load_raw_data("not_a_path")  # type: ignore
