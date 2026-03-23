"""
FastAPI serving layer for Spotify popularity predictions.

This module contains request/response validation and delegates actual model
loading and scoring to serving utilities in src.infer.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore
from fastapi import FastAPI, HTTPException  # type: ignore
from pydantic import BaseModel, ConfigDict, Field

from src.infer import load_inference_model, run_inference
from src.logger import configure_logging
from src.main import load_config, resolve_repo_path

logger = logging.getLogger(__name__)


class TrackFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    acousticness: float = Field(..., ge=0, le=1)
    danceability: float = Field(..., ge=0, le=1)
    duration_ms: int = Field(..., gt=0)
    energy: float = Field(..., ge=0, le=1)
    instrumentalness: float = Field(..., ge=0, le=1)
    key: int = Field(..., ge=0, le=11)
    liveness: float = Field(..., ge=0, le=1)
    loudness: float
    mode: int = Field(..., ge=0, le=1)
    speechiness: float = Field(..., ge=0, le=1)
    tempo: float = Field(..., gt=0)
    valence: float = Field(..., ge=0, le=1)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instances: list[TrackFeatures] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    predictions: list[float]
    model_source: str
    model_reference: str


def create_app(
    *,
    config: dict[str, Any] | None = None,
    project_root: Path | None = None,
    model_loader=load_inference_model,
) -> FastAPI:
    resolved_project_root = project_root or Path(__file__).resolve().parents[1]
    cfg = config or load_config(resolved_project_root / "config.yaml")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log_level = str(cfg.get("logging", {}).get("level", "INFO"))
        log_file = resolve_repo_path(
            resolved_project_root,
            str(cfg.get("paths", {}).get("log_file", "logs/pipeline.log")),
        )
        configure_logging(log_level=log_level, log_file=log_file)

        model, metadata = model_loader(cfg, project_root=resolved_project_root)
        app.state.model = model
        app.state.model_metadata = metadata
        logger.info(
            "API startup complete | model_source=%s | model_reference=%s",
            metadata["source"],
            metadata["reference"],
        )
        yield

    app = FastAPI(
        title="Spotify Popularity Prediction API",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        metadata = getattr(app.state, "model_metadata", {})
        return {
            "status": "ok",
            "model_loaded": hasattr(app.state, "model"),
            "model_source": metadata.get("source", "unknown"),
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        try:
            feature_frame = pd.DataFrame(
                [instance.model_dump() for instance in request.instances]
            )
            predictions = run_inference(app.state.model, feature_frame)
        except Exception as exc:  # pragma: no cover - exercised via HTTP layer
            logger.exception("Prediction request failed.")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        metadata = app.state.model_metadata
        return PredictResponse(
            predictions=predictions["prediction"].astype(float).tolist(),
            model_source=metadata["source"],
            model_reference=metadata["reference"],
        )

    return app


app = create_app()
