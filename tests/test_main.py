# import  `pandas` is a popular Python library used for
# data manipulation and analysis. In the provided
# code snippet, `pandas` is being used to create
# a DataFrame from a dictionary and to save that
# DataFrame to a CSV file. Additionally, `pandas`
# is commonly used for tasks such as data
# cleaning, transformation, and exploration in
# data science and machine learning projects.

import pandas as pd  # type: ignore
from pathlib import Path
import src.utils as utils_mod
import src.main as main_mod


def test_main_pipeline_smoke(tmp_path, monkeypatch):
    # create temp folder structure
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"

    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    models_dir.mkdir()
    reports_dir.mkdir()

    # create tiny raw dataset with required columns
    df_raw = pd.DataFrame({
        "duration_ms": [100000, 200000, 150000, 120000],
        "tempo": [120, 130, 110, 125],
        "key": [0, 1, 2, 3],
        "mode": [1, 0, 1, 0],
        "acousticness": [0.1, 0.2, 0.3, 0.4],
        "danceability": [0.5, 0.6, 0.7, 0.8],
        "energy": [0.7, 0.8, 0.6, 0.9],
        "instrumentalness": [0.0, 0.1, 0.0, 0.2],
        "liveness": [0.1, 0.2, 0.1, 0.3],
        "loudness": [-5.0, -6.0, -4.0, -7.0],
        "speechiness": [0.05, 0.04, 0.06, 0.03],
        "valence": [0.4, 0.5, 0.6, 0.3],
        "popularity": [10, 20, 15, 25],
    })
    raw_path = raw_dir / "SpotifyAudioFeaturesApril2019.csv"
    df_raw.to_csv(raw_path, index=False)

    # temporary config
    settings = {
        "project": {
            "name": "Test Project",
            "problem_type": "regression",
            "target_column": "popularity",
        },
        "paths": {
            "raw_data": str(raw_path),
            "clean_data": str(processed_dir / "clean.csv"),
            "model_path": str(models_dir / "model.joblib"),
            "report_path": str(reports_dir / "predictions.csv"),
        },
        "train": {
            "test_size": 0.25,
            "val_size": 0.25,
            "seed": 42,
            "rf_n_estimators": 10,   # keep small for test speed
            "rf_max_depth": 3,
        },
        "features": {
            "quantile_bin": ["duration_ms", "tempo"],
            "categorical_onehot": ["key", "mode"],
            "numeric_passthrough": [
                "acousticness", "danceability", "energy", "instrumentalness",
                "liveness", "loudness", "speechiness", "valence"
            ],
            "n_bins": 3,
        },
        "logging": {"level": "INFO", "file": None},
    }

    # Redirect JSON outputs to tmp reports/
    real_save_json = utils_mod.save_json

    def save_json_to_tmp(obj, filepath: Path):
        reports_dir.mkdir(parents=True, exist_ok=True)
        forced = reports_dir / filepath.name  # metrics.json / run_config.json
        return real_save_json(obj, forced)

    monkeypatch.setattr(main_mod, "save_json", save_json_to_tmp)

    # Monkeypatch SETTINGS used inside main.py
    monkeypatch.setattr(main_mod, "SETTINGS", settings)

    # run pipeline
    main_mod.main()

    # artifact checks (all inside tmp_path)
    assert Path(settings["paths"]["clean_data"]).exists()
    assert Path(settings["paths"]["model_path"]).exists()
    assert Path(settings["paths"]["report_path"]).exists()

    assert (reports_dir / "metrics.json").exists()
    assert (reports_dir / "run_config.json").exists()


def test_main_logs_and_raises_on_failure(monkeypatch):
    # Make pipeline fail early
    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(main_mod.load_data, "load_raw_data", boom)

    try:
        main_mod.main()
        assert False, "Expected RuntimeError"
    except RuntimeError as e:
        assert "boom" in str(e)
