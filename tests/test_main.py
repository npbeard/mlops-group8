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
import types
import yaml  # type: ignore
import pytest  # type: ignore
import runpy
import sys
import src.utils as utils_mod
import src.main as main_mod
import src.clean_data as clean_data_mod
import src.evaluate as evaluate_mod
import src.features as features_mod
import src.infer as infer_mod
import src.load_data as load_data_mod
import src.logger as logger_mod
import src.train as train_mod
import src.validate as validate_mod


def _base_settings(tmp_path, *, wandb_enabled=False):
    return {
        "project": {
            "name": "Test Project",
            "problem_type": "regression",
            "target_column": "popularity",
        },
        "paths": {
            "raw_data": str(tmp_path / "data" / "raw" / "SpotifyAudioFeaturesApril2019.csv"),
            "clean_data": str(tmp_path / "data" / "processed" / "clean.csv"),
            "model_path": str(tmp_path / "models" / "model.joblib"),
            "report_path": str(tmp_path / "reports" / "predictions.csv"),
            "metrics_path": str(tmp_path / "reports" / "metrics.json"),
            "run_config_path": str(tmp_path / "reports" / "run_config.json"),
            "log_file": str(tmp_path / "logs" / "pipeline.log"),
            "artifact_cache_dir": str(tmp_path / ".artifacts"),
        },
        "train": {
            "test_size": 0.25,
            "val_size": 0.25,
            "seed": 42,
            "rf_n_estimators": 10,
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
        "logging": {"level": "INFO"},
        "wandb": {
            "enabled": wandb_enabled,
            "entity": "group8",
            "project": "spotify-tests",
            "model_artifact_name": "spotify-popularity-pipeline",
            "production_alias": "prod",
            "log_processed_data": False,
            "log_predictions": False,
        },
        "inference": {"source": "local"},
        "api": {"host": "0.0.0.0", "port": 8000},
    }


def _tiny_df():
    return pd.DataFrame({
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


class DummyModel:
    def predict(self, X):
        return [1] * len(X)


class FakeArtifact:
    def __init__(self, name, type, description):
        self.name = name
        self.type = type
        self.description = description
        self.files = []

    def add_file(self, filepath):
        self.files.append(filepath)


class FakeWandb:
    def __init__(self):
        self.logged = []
        self.artifacts = []
        self.finished = []
        self.run = None

    def init(self, project, config, job_type):
        self.run = types.SimpleNamespace(name="fake-run")
        self.init_args = {
            "project": project,
            "config": config,
            "job_type": job_type,
        }
        return self.run

    def log(self, payload):
        self.logged.append(payload)

    def Artifact(self, name, type, description):
        return FakeArtifact(name, type, description)

    def log_artifact(self, artifact):
        self.artifacts.append(artifact)

    def finish(self, exit_code=None):
        self.finished.append(exit_code)
        self.run = None


class BrokenWandb:
    __file__ = "/tmp/fake-wandb.py"


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
    settings = _base_settings(tmp_path)
    settings["paths"]["raw_data"] = str(raw_path)
    settings["paths"]["clean_data"] = str(processed_dir / "clean.csv")
    settings["paths"]["model_path"] = str(models_dir / "model.joblib")
    settings["paths"]["report_path"] = str(reports_dir / "predictions.csv")

    # Redirect JSON outputs to tmp reports/
    real_save_json = utils_mod.save_json

    def save_json_to_tmp(obj, filepath: Path):
        reports_dir.mkdir(parents=True, exist_ok=True)
        forced = reports_dir / filepath.name  # metrics.json / run_config.json
        return real_save_json(obj, forced)

    monkeypatch.setattr(main_mod, "save_json", save_json_to_tmp)

    # run pipeline
    main_mod.main(settings)

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

    with pytest.raises(RuntimeError, match="boom"):
        main_mod.main(_base_settings(Path.cwd()))


def test_load_config_reads_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    expected = {"project": {"name": "Loaded Project"}}
    config_path.write_text(yaml.safe_dump(expected), encoding="utf-8")

    assert main_mod.load_config(config_path) == expected


def test_resolve_repo_path_handles_relative_and_absolute(tmp_path):
    absolute = tmp_path / "absolute.txt"
    assert main_mod.resolve_repo_path(tmp_path, "relative.txt") == tmp_path / "relative.txt"
    assert main_mod.resolve_repo_path(tmp_path, str(absolute)) == absolute


def test_wandb_helpers_handle_missing_or_empty_config():
    assert main_mod._wandb_is_enabled({}) is False
    assert main_mod._wandb_is_enabled({"wandb": {"enabled": True}}) is True
    assert main_mod._wandb_get_str({}, "project", default="fallback") == "fallback"
    assert main_mod._wandb_get_str({"wandb": {"project": None}}, "project", default="fallback") == "fallback"
    assert main_mod._wandb_get_bool({}, "enabled", default=True) is True
    assert main_mod._wandb_get_bool({"wandb": {"enabled": 0}}, "enabled", default=True) is False


def test_load_wandb_module_returns_none_when_missing(monkeypatch):
    def boom(name):
        raise ImportError

    monkeypatch.setattr(main_mod.importlib, "import_module", boom)
    assert main_mod._load_wandb_module(Path.cwd()) is None


def test_load_wandb_module_restores_existing_module_on_import_error(monkeypatch):
    sentinel = object()

    def boom(name):
        raise ImportError

    monkeypatch.setitem(main_mod.sys.modules, "wandb", sentinel)
    monkeypatch.setattr(main_mod.importlib, "import_module", boom)

    assert main_mod._load_wandb_module(Path.cwd()) is None
    assert main_mod.sys.modules["wandb"] is sentinel


def test_load_wandb_module_raises_for_incomplete_package(monkeypatch):
    monkeypatch.setattr(
        main_mod.importlib,
        "import_module",
        lambda name: BrokenWandb(),
    )

    with pytest.raises(ImportError, match="not a usable Weights & Biases package"):
        main_mod._load_wandb_module(Path.cwd())


def test_load_wandb_module_recovers_from_local_wandb_namespace(tmp_path, monkeypatch):
    class NamespaceWandb:
        __file__ = None
        __path__ = [str(tmp_path / "wandb")]

    fake_wandb = FakeWandb()
    calls = {"count": 0}

    def fake_import(name):
        calls["count"] += 1
        if calls["count"] == 1:
            return NamespaceWandb()
        return fake_wandb

    monkeypatch.setattr(main_mod.importlib, "import_module", fake_import)
    monkeypatch.setattr(main_mod.sys, "path", [str(tmp_path), "/site-packages"])
    monkeypatch.setattr(main_mod.sys, "modules", {})

    assert main_mod._load_wandb_module(tmp_path) is fake_wandb


def test_load_wandb_module_returns_valid_module_without_recovery(monkeypatch):
    fake_wandb = FakeWandb()
    monkeypatch.setattr(
        main_mod.importlib,
        "import_module",
        lambda name: fake_wandb,
    )

    assert main_mod._load_wandb_module(Path.cwd()) is fake_wandb


def test_main_loads_config_and_dotenv_when_config_not_passed(tmp_path, monkeypatch):
    settings = _base_settings(tmp_path)
    df = _tiny_df()
    dotenv_calls = []
    configured = []

    monkeypatch.setattr(main_mod, "load_config", lambda path: settings)
    monkeypatch.setattr(main_mod, "load_dotenv", lambda **kwargs: dotenv_calls.append(kwargs))
    monkeypatch.setattr(main_mod, "configure_logging", lambda **kwargs: configured.append(kwargs))
    monkeypatch.setattr(main_mod.load_data, "load_raw_data", lambda path: df)
    monkeypatch.setattr(main_mod.clean_data, "clean_dataframe", lambda frame, target: frame)
    monkeypatch.setattr(main_mod.validate, "validate_dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_mod.features, "get_feature_preprocessor", lambda **kwargs: object())
    monkeypatch.setattr(main_mod.train, "train_model", lambda *args, **kwargs: DummyModel())
    monkeypatch.setattr(main_mod.evaluate, "evaluate_model", lambda *args, **kwargs: {"rmse": 1.0})
    monkeypatch.setattr(main_mod.infer, "run_inference", lambda *args, **kwargs: pd.DataFrame({"prediction": [1]}))
    monkeypatch.setattr(main_mod, "save_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_mod, "save_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_mod, "save_json", lambda *args, **kwargs: None)

    assert main_mod.main() == 0
    assert dotenv_calls
    assert configured[0]["log_level"] == "INFO"


def test_main_raises_when_wandb_enabled_but_package_missing(tmp_path, monkeypatch):
    settings = _base_settings(tmp_path, wandb_enabled=True)
    monkeypatch.setattr(main_mod, "_load_wandb_module", lambda project_root: None)
    monkeypatch.setattr(main_mod, "configure_logging", lambda **kwargs: None)
    monkeypatch.setattr(main_mod, "load_dotenv", None)

    with pytest.raises(ImportError, match="wandb is enabled"):
        main_mod.main(settings)


def test_main_raises_when_wandb_project_missing(tmp_path, monkeypatch):
    settings = _base_settings(tmp_path, wandb_enabled=True)
    settings["wandb"]["project"] = "   "
    monkeypatch.setattr(main_mod, "_load_wandb_module", lambda project_root: FakeWandb())
    monkeypatch.setattr(main_mod, "configure_logging", lambda **kwargs: None)
    monkeypatch.setattr(main_mod, "load_dotenv", None)

    with pytest.raises(ValueError, match="wandb.project must be set"):
        main_mod.main(settings)


def test_main_wandb_logs_artifacts_and_predictions(tmp_path, monkeypatch):
    settings = _base_settings(tmp_path, wandb_enabled=True)
    settings["wandb"]["log_processed_data"] = True
    settings["wandb"]["log_predictions"] = True
    df = _tiny_df()
    fake_wandb = FakeWandb()

    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setattr(main_mod, "_load_wandb_module", lambda project_root: fake_wandb)
    monkeypatch.setattr(main_mod, "load_dotenv", None)
    monkeypatch.setattr(main_mod, "configure_logging", lambda **kwargs: None)
    monkeypatch.setattr(main_mod.load_data, "load_raw_data", lambda path: df)
    monkeypatch.setattr(main_mod.clean_data, "clean_dataframe", lambda frame, target: frame)
    monkeypatch.setattr(main_mod.validate, "validate_dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_mod.features, "get_feature_preprocessor", lambda **kwargs: object())
    monkeypatch.setattr(main_mod.train, "train_model", lambda *args, **kwargs: DummyModel())
    monkeypatch.setattr(
        main_mod,
        "save_model",
        lambda model, path: path.write_text("model", encoding="utf-8"),
    )
    monkeypatch.setattr(
        main_mod,
        "save_csv",
        lambda frame, path: path.write_text(frame.to_csv(index=False), encoding="utf-8"),
    )
    monkeypatch.setattr(
        main_mod,
        "save_json",
        lambda obj, path: path.write_text("{}", encoding="utf-8"),
    )
    monkeypatch.setattr(
        main_mod.evaluate,
        "evaluate_model",
        lambda *args, **kwargs: {"rmse": 1.0, "mae": 0.5},
    )
    monkeypatch.setattr(
        main_mod.infer,
        "run_inference",
        lambda *args, **kwargs: pd.DataFrame({"prediction": [1]}),
    )

    assert main_mod.main(settings) == 0
    assert fake_wandb.init_args["project"] == "spotify-tests"
    assert len(fake_wandb.logged) >= 4
    assert any("data/raw_rows" in payload for payload in fake_wandb.logged)
    assert any("metrics/val/rmse" in payload for payload in fake_wandb.logged)
    assert len(fake_wandb.artifacts) == 3
    assert fake_wandb.finished == [None]


def test_main_finishes_wandb_with_error_code_on_failure(tmp_path, monkeypatch):
    settings = _base_settings(tmp_path, wandb_enabled=True)
    fake_wandb = FakeWandb()

    monkeypatch.setenv("WANDB_API_KEY", "token")
    monkeypatch.setattr(main_mod, "_load_wandb_module", lambda project_root: fake_wandb)
    monkeypatch.setattr(main_mod, "load_dotenv", None)
    monkeypatch.setattr(main_mod, "configure_logging", lambda **kwargs: None)
    monkeypatch.setattr(main_mod.load_data, "load_raw_data", lambda path: _tiny_df())
    monkeypatch.setattr(main_mod.clean_data, "clean_dataframe", lambda frame, target: frame)
    monkeypatch.setattr(main_mod.validate, "validate_dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_mod.features, "get_feature_preprocessor", lambda **kwargs: object())
    monkeypatch.setattr(main_mod.train, "train_model", lambda *args, **kwargs: DummyModel())
    monkeypatch.setattr(
        main_mod.evaluate,
        "evaluate_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("eval failed")),
    )
    monkeypatch.setattr(main_mod, "save_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_mod, "save_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_mod, "save_json", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="eval failed"):
        main_mod.main(settings)

    assert fake_wandb.finished == [1]


def test_module_entrypoint_calls_main(monkeypatch, tmp_path):
    settings = _base_settings(tmp_path)
    df = _tiny_df()
    configured = []

    monkeypatch.setattr(yaml, "safe_load", lambda stream: settings)
    monkeypatch.setattr(logger_mod, "configure_logging", lambda **kwargs: configured.append(kwargs))
    monkeypatch.setattr(load_data_mod, "load_raw_data", lambda path: df)
    monkeypatch.setattr(clean_data_mod, "clean_dataframe", lambda frame, target: frame)
    monkeypatch.setattr(validate_mod, "validate_dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(features_mod, "get_feature_preprocessor", lambda **kwargs: object())
    monkeypatch.setattr(train_mod, "train_model", lambda *args, **kwargs: DummyModel())
    monkeypatch.setattr(evaluate_mod, "evaluate_model", lambda *args, **kwargs: {"rmse": 1.0})
    monkeypatch.setattr(infer_mod, "run_inference", lambda *args, **kwargs: pd.DataFrame({"prediction": [1]}))
    monkeypatch.setattr(utils_mod, "save_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr(utils_mod, "save_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(utils_mod, "save_json", lambda *args, **kwargs: None)

    sys.modules.pop("__main__", None)
    sys.modules.pop("src.main", None)
    runpy.run_module("src.main", run_name="__main__")

    assert configured
