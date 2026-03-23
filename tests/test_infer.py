from pathlib import Path

import pandas as pd  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
import pytest  # type: ignore

import src.infer as infer_mod
from src.infer import _build_wandb_artifact_reference, load_inference_model
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


def test_build_wandb_artifact_reference_uses_prod_alias():
    config = {
        "wandb": {
            "entity": "group8",
            "project": "spotify-sound-archetypes",
            "model_artifact_name": "spotify-popularity-pipeline",
            "production_alias": "prod",
        }
    }

    assert _build_wandb_artifact_reference(config) == (
        "group8/spotify-sound-archetypes/spotify-popularity-pipeline:prod"
    )


def test_load_inference_model_from_local_path(tmp_path):
    X_train = pd.DataFrame({"x": [1, 2, 3]})
    y_train = [1, 2, 3]
    model = LinearRegression().fit(X_train, y_train)
    model_path = tmp_path / "models" / "model.joblib"
    model_path.parent.mkdir(parents=True)

    import joblib  # type: ignore

    joblib.dump(model, model_path)

    loaded_model, metadata = load_inference_model(
        {
            "paths": {
                "model_path": str(model_path),
                "artifact_cache_dir": str(tmp_path / ".artifacts"),
            },
            "inference": {"source": "local"},
        },
        project_root=tmp_path,
    )

    assert hasattr(loaded_model, "predict")
    assert metadata["source"] == "local"


def test_load_inference_model_from_wandb_artifact(tmp_path):
    model_dir = tmp_path / "downloaded-artifact"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "model.joblib"
    X_train = pd.DataFrame({"x": [1, 2, 3]})
    y_train = [1, 2, 3]
    model = LinearRegression().fit(X_train, y_train)

    import joblib  # type: ignore

    joblib.dump(model, model_path)

    class FakeArtifact:
        def download(self, root):
            return str(model_dir)

    class FakeApi:
        def artifact(self, reference):
            self.reference = reference
            return FakeArtifact()

    class FakeWandb:
        def Api(self):
            return FakeApi()

    loaded_model, metadata = load_inference_model(
        {
            "paths": {
                "model_path": "models/model.joblib",
                "artifact_cache_dir": str(tmp_path / ".artifacts"),
            },
            "wandb": {
                "entity": "group8",
                "project": "spotify",
                "model_artifact_name": "popularity-model",
                "production_alias": "prod",
            },
            "inference": {"source": "wandb"},
        },
        project_root=Path(tmp_path),
        wandb_module=FakeWandb(),
    )

    assert hasattr(loaded_model, "predict")
    assert metadata["source"] == "wandb"
    assert metadata["reference"] == "group8/spotify/popularity-model:prod"


def test_get_inference_source_defaults_to_local_for_non_dict_config():
    assert infer_mod._get_inference_source({"inference": "wandb"}) == "local"


def test_load_wandb_module_raises_when_package_missing(monkeypatch):
    def boom(name):
        raise ImportError("missing")

    monkeypatch.setattr(infer_mod.importlib, "import_module", boom)

    with pytest.raises(ImportError, match="wandb is required"):
        infer_mod._load_wandb_module()


def test_load_wandb_module_raises_when_api_missing(monkeypatch):
    class BrokenWandb:
        pass

    monkeypatch.setattr(
        infer_mod.importlib, "import_module", lambda name: BrokenWandb()
    )

    with pytest.raises(ImportError, match="does not expose the Api client"):
        infer_mod._load_wandb_module()


def test_load_wandb_module_returns_valid_module(monkeypatch):
    class ValidWandb:
        class Api:
            pass

    monkeypatch.setattr(
        infer_mod.importlib, "import_module", lambda name: ValidWandb()
    )

    assert infer_mod._load_wandb_module() is not None


def test_build_wandb_artifact_reference_requires_dict_section():
    with pytest.raises(ValueError, match="wandb section is required"):
        _build_wandb_artifact_reference({"wandb": "bad"})


def test_build_wandb_artifact_reference_requires_project_and_name():
    with pytest.raises(
        ValueError, match="wandb.project and wandb.model_artifact_name"
    ):
        _build_wandb_artifact_reference(
            {
                "wandb": {
                    "entity": "group8",
                    "project": "",
                    "model_artifact_name": "",
                }
            }
        )


def test_build_wandb_artifact_reference_without_entity():
    assert _build_wandb_artifact_reference(
        {
            "wandb": {
                "project": "spotify",
                "model_artifact_name": "popularity-model",
                "production_alias": "prod",
            }
        }
    ) == "spotify/popularity-model:prod"


def test_find_downloaded_model_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="No .joblib model file"):
        infer_mod._find_downloaded_model(tmp_path)


def test_load_inference_model_rejects_unsupported_source(tmp_path):
    with pytest.raises(ValueError, match="Unsupported inference.source"):
        load_inference_model(
            {
                "paths": {"model_path": "models/model.joblib"},
                "inference": {"source": "registry"},
            },
            project_root=tmp_path,
        )
