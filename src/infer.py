"""
Module: Inference
-----------------
Role: Make predictions on new, unseen data.
Input: Trained Model + New Data.
Output: Predictions (Array or DataFrame).
"""
import importlib
import logging
from pathlib import Path
from typing import Any
from typing import cast

import pandas as pd  # type: ignore

from src.utils import load_model

logger = logging.getLogger(__name__)


def resolve_project_path(project_root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else project_root / path


def _get_inference_source(config: dict[str, Any]) -> str:
    inference_cfg = config.get("inference", {})
    if not isinstance(inference_cfg, dict):
        return "local"
    return str(inference_cfg.get("source", "local")).strip().lower() or "local"


def _load_wandb_module():
    try:
        wandb_module = importlib.import_module("wandb")
    except ImportError as exc:  # pragma: no cover - exercised in tests
        raise ImportError(
            "wandb is required when inference.source is set to 'wandb'."
        ) from exc

    if not hasattr(wandb_module, "Api"):
        raise ImportError(
            "Imported wandb module does not expose the Api client."
        )

    return wandb_module


def _build_wandb_artifact_reference(config: dict[str, Any]) -> str:
    wandb_cfg = config.get("wandb", {})
    if not isinstance(wandb_cfg, dict):
        raise ValueError(
            "config.yaml: wandb section is required for W&B inference"
        )

    entity = str(wandb_cfg.get("entity", "")).strip()
    project = str(wandb_cfg.get("project", "")).strip()
    artifact_name = str(wandb_cfg.get("model_artifact_name", "")).strip()
    alias = str(wandb_cfg.get("production_alias", "prod")).strip() or "prod"

    if not project or not artifact_name:
        raise ValueError(
            "config.yaml: wandb.model_artifact_name are required "
            "for W&B-backed inference"
        )

    if entity:
        return f"{entity}/{project}/{artifact_name}:{alias}"
    return f"{project}/{artifact_name}:{alias}"


def _find_downloaded_model(artifact_dir: Path) -> Path:
    if not (model_candidates := sorted(artifact_dir.rglob("*.joblib"))):
        raise FileNotFoundError(
            f"No .joblib model file was found in downloaded artifact: "
            f"{artifact_dir}"
        )
    else:
        return model_candidates[0]


def load_inference_model(
    config: dict[str, Any],
    *,
    project_root: Path | None = None,
    wandb_module: Any | None = None,
) -> tuple[Any, dict[str, str]]:
    """
    Load the model used by the serving layer.
    In production, inference is expected to use the promoted W&B artifact
    aliased 'prod'. Local model loading is still supported for offline
    development.
    """
    project_root = project_root or Path(__file__).resolve().parents[1]
    source = _get_inference_source(config)

    if source == "local":
        local_model_path = resolve_project_path(
            project_root,
            str(config["paths"]["model_path"]),
        )
        model = load_model(local_model_path)
        return model, {
            "source": "local",
            "reference": str(local_model_path),
        }

    if source != "wandb":
        raise ValueError(
            f"Unsupported inference.source '{source}'. "
            f"Expected 'local' or 'wandb'."
        )

    active_wandb = wandb_module or _load_wandb_module()
    artifact_reference = _build_wandb_artifact_reference(config)
    cache_dir = resolve_project_path(
        project_root,
        str(config["paths"].get("artifact_cache_dir", ".artifacts")),
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    artifact = active_wandb.Api().artifact(artifact_reference)
    artifact_dir = Path(cast(str, artifact.download(root=str(cache_dir))))
    model_path = _find_downloaded_model(artifact_dir)
    model = load_model(model_path)
    logger.info(
        "Loaded inference model from W&B artifact alias | reference=%s",
        artifact_reference,
    )
    return model, {
        "source": "wandb",
        "reference": artifact_reference,
    }


def run_inference(model: Any, X_infer: pd.DataFrame) -> pd.DataFrame:
    """
    Inputs:
    - model: The fitted Pipeline / estimator implementing .predict()
    - X_infer: New features to predict on.
    Outputs:
    - pd.DataFrame: A DataFrame with a single column "prediction".
    """
    logger.info("Running inference on new data...")

    if not hasattr(model, "predict"):
        raise TypeError("Model artifact must implement a .predict() method")

    if not isinstance(X_infer, pd.DataFrame):
        raise TypeError(f"X_infer must be a pandas.DataFrame,"
                        f"got {type(X_infer)}")

    if X_infer.empty:
        raise ValueError("X_infer is empty; cannot run inference")

    preds = model.predict(X_infer)

    df_preds = pd.DataFrame({"prediction": preds})
    logger.info("Inference complete. Generated %d predictions.", len(df_preds))
    return df_preds
