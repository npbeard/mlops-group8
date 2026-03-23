from pathlib import Path

from fastapi.testclient import TestClient  # type: ignore

from src.api import create_app


class DummyModel:
    def predict(self, frame):
        return [42.0] * len(frame)


def _config(tmp_path: Path) -> dict:
    return {
        "project": {
            "name": "Spotify Sound Archetypes",
            "problem_type": "regression",
            "target_column": "popularity",
        },
        "paths": {
            "raw_data": str(
                tmp_path / "data" / "raw" / "SpotifyAudioFeaturesApril2019.csv"
            ),
            "clean_data": str(tmp_path / "data" / "processed" / "clean.csv"),
            "model_path": str(tmp_path / "models" / "model.joblib"),
            "report_path": str(tmp_path / "reports" / "predictions.csv"),
            "metrics_path": str(tmp_path / "reports" / "metrics.json"),
            "run_config_path": str(tmp_path / "reports" / "run_config.json"),
            "log_file": str(tmp_path / "logs" / "pipeline.log"),
            "artifact_cache_dir": str(tmp_path / ".artifacts"),
        },
        "logging": {"level": "INFO"},
        "wandb": {
            "entity": "group8",
            "project": "spotify-tests",
            "model_artifact_name": "spotify-popularity-pipeline",
            "production_alias": "prod",
        },
        "inference": {"source": "local"},
        "api": {"host": "0.0.0.0", "port": 8000},
    }


def _payload() -> dict:
    return {
        "instances": [
            {
                "acousticness": 0.12,
                "danceability": 0.65,
                "duration_ms": 200000,
                "energy": 0.71,
                "instrumentalness": 0.0,
                "key": 5,
                "liveness": 0.11,
                "loudness": -5.1,
                "mode": 1,
                "speechiness": 0.05,
                "tempo": 120.5,
                "valence": 0.44,
            }
        ]
    }


def test_health_returns_ok(tmp_path):
    app = create_app(
        config=_config(tmp_path),
        project_root=tmp_path,
        model_loader=lambda cfg, project_root: (
            DummyModel(),
            {"source": "wandb", "reference": "entity/project/model:prod"},
        ),
    )

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_predict_returns_predictions(tmp_path):
    app = create_app(
        config=_config(tmp_path),
        project_root=tmp_path,
        model_loader=lambda cfg, project_root: (
            DummyModel(),
            {"source": "wandb", "reference": "entity/project/model:prod"},
        ),
    )

    with TestClient(app) as client:
        response = client.post("/predict", json=_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["predictions"] == [42.0]
    assert body["model_source"] == "wandb"


def test_predict_rejects_extra_fields(tmp_path):
    app = create_app(
        config=_config(tmp_path),
        project_root=tmp_path,
        model_loader=lambda cfg, project_root: (
            DummyModel(),
            {"source": "wandb", "reference": "entity/project/model:prod"},
        ),
    )
    payload = _payload()
    payload["instances"][0]["unexpected"] = "nope"

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422
