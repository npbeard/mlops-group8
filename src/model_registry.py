"""
Helpers for working with W&B model artifacts used in production serving.
"""

import importlib
from typing import Any


def load_wandb_public_api():
    try:
        wandb_module = importlib.import_module("wandb")
    except ImportError as exc:  # pragma: no cover - exercised in tests
        raise ImportError(
            "wandb is required to manage model registry aliases."
        ) from exc

    if not hasattr(wandb_module, "Api"):
        raise ImportError(
            "Imported wandb module does not expose the Api client."
        )

    return wandb_module


def build_model_artifact_reference(
    config: dict[str, Any],
    *,
    alias: str | None = None,
) -> str:
    wandb_cfg = config.get("wandb", {})
    if not isinstance(wandb_cfg, dict):
        raise ValueError(
            (
                "config.yaml: wandb section is required for model registry "
                "operations"
            )
        )

    entity = str(wandb_cfg.get("entity", "")).strip()
    project = str(wandb_cfg.get("project", "")).strip()
    artifact_name = str(wandb_cfg.get("model_artifact_name", "")).strip()
    resolved_alias = (
        alias
        or str(wandb_cfg.get("production_alias", "prod")).strip()
        or "prod"
    )

    if not project or not artifact_name:
        raise ValueError(
            (
                "config.yaml: wandb.project and wandb.model_artifact_name are "
                "required for model registry operations"
            )
        )

    if entity:
        return f"{entity}/{project}/{artifact_name}:{resolved_alias}"
    return f"{project}/{artifact_name}:{resolved_alias}"


def promote_model_artifact(
    config: dict[str, Any],
    *,
    source_alias: str = "latest",
    target_alias: str | None = None,
    wandb_module: Any | None = None,
) -> dict[str, Any]:
    """
    Attach the target alias to a candidate artifact version.

    This supports a simple promotion flow:
    1. Train and inspect the artifact version behind `latest` or a version id.
    2. Promote the approved version to `prod`.
    """
    wandb_cfg = config.get("wandb", {})
    default_target_alias = "prod"
    if isinstance(wandb_cfg, dict):
        default_target_alias = (
            str(wandb_cfg.get("production_alias", "prod")).strip() or "prod"
        )

    resolved_target_alias = (
        (target_alias or "").strip() or default_target_alias
    )
    resolved_source_alias = source_alias.strip() or "latest"

    active_wandb = wandb_module or load_wandb_public_api()
    source_reference = build_model_artifact_reference(
        config,
        alias=resolved_source_alias,
    )
    promoted_reference = build_model_artifact_reference(
        config,
        alias=resolved_target_alias,
    )

    artifact = active_wandb.Api().artifact(source_reference)
    current_aliases = list(getattr(artifact, "aliases", []))
    if resolved_target_alias not in current_aliases:
        current_aliases.append(resolved_target_alias)
        artifact.aliases = current_aliases

    save = getattr(
        artifact,
        "save",
        None,
    )
    if not callable(save):
        raise AttributeError(
            "W&B artifact object does not expose a save() method."
        )

    save()

    return {
        "source_reference": source_reference,
        "promoted_reference": promoted_reference,
        "artifact_version": str(getattr(artifact, "version", "")),
        "aliases": list(getattr(artifact, "aliases", [])),
    }
