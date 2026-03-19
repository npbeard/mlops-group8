"""
Module: Main Pipeline
---------------------
Role: Orchestrate the entire flow
(Load -> Clean -> Validate -> Train -> Evaluate).
Usage: python -m src.main
"""
import logging
import os
import importlib
import sys
from pathlib import Path
from typing import Any

import yaml  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

import src.clean_data as clean_data
import src.evaluate as evaluate
import src.features as features
import src.infer as infer
import src.load_data as load_data
import src.train as train
import src.validate as validate
from src.logger import configure_logging
from src.utils import save_csv, save_json, save_model

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency at runtime
    load_dotenv = None

def load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_repo_path(project_root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else project_root / path


def _wandb_is_enabled(cfg: dict[str, Any]) -> bool:
    wandb_cfg = cfg.get("wandb")
    return isinstance(wandb_cfg, dict) and bool(wandb_cfg.get("enabled", False))


def _wandb_get_str(cfg: dict[str, Any], key: str, default: str = "") -> str:
    wandb_cfg = cfg.get("wandb")
    if not isinstance(wandb_cfg, dict):
        return default
    value = wandb_cfg.get(key, default)
    return str(value).strip() if value is not None else default


def _wandb_get_bool(cfg: dict[str, Any], key: str, default: bool = False) -> bool:
    wandb_cfg = cfg.get("wandb")
    if not isinstance(wandb_cfg, dict):
        return default
    return bool(wandb_cfg.get(key, default))


def _load_wandb_module(project_root: Path):
    existing_module = sys.modules.pop("wandb", None)
    try:
        wandb_module = importlib.import_module("wandb")
    except ImportError:
        if existing_module is not None:
            sys.modules["wandb"] = existing_module
        return None

    required_attrs = ["init", "Artifact", "log", "log_artifact", "finish"]
    missing_attrs = [
        attr for attr in required_attrs if not hasattr(wandb_module, attr)
    ]
    if missing_attrs:
        module_locations = [
            str(path)
            for path in getattr(wandb_module, "__path__", [])
        ]
        if any(
            Path(path).resolve() == (project_root / "wandb").resolve()
            for path in module_locations
        ):
            original_sys_path = sys.path[:]
            try:
                sys.modules.pop("wandb", None)
                sys.path = [
                    path for path in sys.path
                    if Path(path or ".").resolve() != project_root.resolve()
                ]
                wandb_module = importlib.import_module("wandb")
                missing_attrs = [
                    attr for attr in required_attrs if not hasattr(wandb_module, attr)
                ]
            finally:
                sys.path = original_sys_path

        if not missing_attrs:
            return wandb_module

        module_path = getattr(wandb_module, "__file__", "<unknown>")
        raise ImportError(
            "Imported 'wandb' is not a usable Weights & Biases package. "
            f"module={module_path} missing={missing_attrs}. "
            "Reinstall wandb in the active environment."
        )

    return wandb_module


def main(config: dict[str, Any] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[1]
    cfg = config or load_config(project_root / "config.yaml")

    if load_dotenv is not None:
        load_dotenv(dotenv_path=project_root / ".env", override=False)

    log_level = str(cfg.get("logging", {}).get("level", "INFO"))
    log_file_cfg = str(cfg.get("paths", {}).get("log_file", "logs/pipeline.log"))
    log_file_path = resolve_repo_path(project_root, log_file_cfg)
    configure_logging(log_level=log_level, log_file=log_file_path)

    wandb_run = None
    wandb_module = None

    try:
        logger.info(
            "--- Starting MLOps Pipeline: %s ---",
            cfg["project"]["name"],
        )

        # 1. Infrastructure Setup
        for folder_key in ["clean_data", "model_path", "report_path", "log_file"]:
            resolve_repo_path(project_root, cfg["paths"][folder_key]).parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        if _wandb_is_enabled(cfg):
            os.environ.setdefault("WANDB_DIR", str(project_root / ".wandb"))
            wandb_module = _load_wandb_module(project_root)
            if wandb_module is None:
                raise ImportError(
                    "wandb is enabled in config.yaml but the package is not installed."
                )

            wandb_project = _wandb_get_str(cfg, "project")
            if not wandb_project:
                raise ValueError(
                    "config.yaml: wandb.project must be set when wandb.enabled is true"
                )

            if not os.getenv("WANDB_API_KEY") and os.getenv("WANDB_MODE", "") != "offline":
                logger.warning(
                    "WANDB_API_KEY is not set. W&B may prompt for authentication."
                )

            wandb_run = wandb_module.init(
                project=wandb_project,
                config=cfg,
                job_type="factory-pipeline",
            )
            logger.info(
                "Initialized W&B run | name=%s | project=%s",
                wandb_run.name,
                wandb_project,
            )
        else:
            logger.info("W&B disabled, continuing without experiment tracking")

        # 2. Load
        raw_data_path = resolve_repo_path(project_root, cfg["paths"]["raw_data"])
        processed_data_path = resolve_repo_path(project_root, cfg["paths"]["clean_data"])
        model_artifact_path = resolve_repo_path(project_root, cfg["paths"]["model_path"])
        predictions_artifact_path = resolve_repo_path(
            project_root, cfg["paths"]["report_path"]
        )
        metrics_path = resolve_repo_path(project_root, cfg["paths"]["metrics_path"])
        run_config_path = resolve_repo_path(project_root, cfg["paths"]["run_config_path"])

        df_raw = load_data.load_raw_data(raw_data_path)
        if wandb_run is not None:
            wandb_module.log({
                "data/raw_rows": int(df_raw.shape[0]),
                "data/raw_cols": int(df_raw.shape[1]),
            })

        # 3. Clean
        df_clean = clean_data.clean_dataframe(
            df_raw, cfg["project"]["target_column"]
        )
        save_csv(df_clean, processed_data_path)
        if wandb_run is not None:
            wandb_module.log({
                "data/clean_rows": int(df_clean.shape[0]),
                "data/clean_cols": int(df_clean.shape[1]),
            })

        # 4. Validate
        required_cols = (
            cfg["features"]["quantile_bin"]
            + cfg["features"]["categorical_onehot"]
            + cfg["features"]["numeric_passthrough"]
            + [cfg["project"]["target_column"]]
        )
        validate.validate_dataframe(
            df_clean,
            required_cols,
            target_column=cfg["project"]["target_column"],
            allow_feature_nulls=True,
        )

        # 5. Split (Train / Val / Test)
        target = cfg["project"]["target_column"]
        X = df_clean.drop(columns=[target])
        y = df_clean[target]

        test_size = cfg["train"]["test_size"]
        val_size = cfg["train"].get("val_size", 0.2)
        seed = cfg["train"]["seed"]

        strat = y if cfg["project"]["problem_type"] == "classification" else None

        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=seed,
            stratify=strat,
        )

        strat_trainval = (
            y_trainval
            if cfg["project"]["problem_type"] == "classification"
            else None
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval,
            y_trainval,
            test_size=val_size,
            random_state=seed,
            stratify=strat_trainval,
        )

        logger.info(
            "Split sizes:"
            "train=%d (%.1f%%), val=%d (%.1f%%), test=%d (%.1f%%)",
            len(X_train), 100 * len(X_train) / len(X),
            len(X_val), 100 * len(X_val) / len(X),
            len(X_test), 100 * len(X_test) / len(X),
        )

        # 6. Feature Engineering
        preprocessor = features.get_feature_preprocessor(
            quantile_bin_cols=cfg["features"]["quantile_bin"],
            categorical_onehot_cols=cfg["features"]["categorical_onehot"],
            numeric_passthrough_cols=cfg["features"]["numeric_passthrough"],
            n_bins=cfg["features"]["n_bins"],
        )

        # 7. Train
        model_pipeline = train.train_model(
            X_train,
            y_train,
            preprocessor,
            cfg["project"]["problem_type"],
            train_config=cfg.get("train", {}),
        )

        # 8. Evaluate (Val and Test)
        val_metrics = evaluate.evaluate_model(
            model_pipeline,
            X_val,
            y_val,
            cfg["project"]["problem_type"],
        )
        if wandb_run is not None:
            wandb_module.log(
                {f"metrics/val/{k}": float(v) for k, v in val_metrics.items()}
            )

        final_preprocessor = features.get_feature_preprocessor(
            quantile_bin_cols=cfg["features"]["quantile_bin"],
            categorical_onehot_cols=cfg["features"]["categorical_onehot"],
            numeric_passthrough_cols=cfg["features"]["numeric_passthrough"],
            n_bins=cfg["features"]["n_bins"],
        )

        final_model_pipeline = train.train_model(
            X_trainval,
            y_trainval,
            final_preprocessor,
            cfg["project"]["problem_type"],
            train_config=cfg.get("train", {}),
        )

        save_model(final_model_pipeline, model_artifact_path)

        test_metrics = evaluate.evaluate_model(
            final_model_pipeline,
            X_test,
            y_test,
            cfg["project"]["problem_type"],
        )
        if wandb_run is not None:
            wandb_module.log(
                {f"metrics/test/{k}": float(v) for k, v in test_metrics.items()}
            )

        metrics = {"val": val_metrics, "test": test_metrics}
        save_json(metrics, metrics_path)
        save_json(cfg, run_config_path)

        if wandb_run is not None:
            model_artifact_name = _wandb_get_str(
                cfg, "model_artifact_name", default="spotify-model"
            )
            model_artifact = wandb_module.Artifact(
                name=model_artifact_name,
                type="model",
                description="Scikit-learn pipeline (preprocessing + estimator)",
            )
            model_artifact.add_file(str(model_artifact_path))
            wandb_module.log_artifact(model_artifact)

            if _wandb_get_bool(cfg, "log_processed_data", default=False):
                data_artifact = wandb_module.Artifact(
                    name=f"{model_artifact_name}-processed-data",
                    type="dataset",
                    description="Processed training dataset written by the factory pipeline",
                )
                data_artifact.add_file(str(processed_data_path))
                wandb_module.log_artifact(data_artifact)

        # 9. Inference
        df_preds = infer.run_inference(final_model_pipeline, X_test)
        save_csv(df_preds, predictions_artifact_path)

        if wandb_run is not None and _wandb_get_bool(
            cfg, "log_predictions", default=False
        ):
            model_artifact_name = _wandb_get_str(
                cfg, "model_artifact_name", default="spotify-model"
            )
            pred_artifact = wandb_module.Artifact(
                name=f"{model_artifact_name}-predictions",
                type="predictions",
                description="Inference outputs written by the factory pipeline",
            )
            pred_artifact.add_file(str(predictions_artifact_path))
            wandb_module.log_artifact(pred_artifact)

        logger.info("--- Pipeline Completed Successfully ---")
        return 0

    except Exception:
        logger.exception("Pipeline failed due to an unhandled exception.")
        if wandb_run is not None:
            wandb_module.finish(exit_code=1)
        raise
    finally:
        if (
            wandb_run is not None
            and wandb_module is not None
            and getattr(wandb_module, "run", None) is not None
        ):
            wandb_module.finish()


if __name__ == "__main__":
    main()
