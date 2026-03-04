from src.load_data import load_raw_data


def test_load_raw_data_creates_dummy(tmp_path):
    test_file = tmp_path / "test.csv"

    df = load_raw_data(test_file)

    assert df is not None
    assert not df.empty
