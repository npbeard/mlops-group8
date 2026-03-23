"""
Helpers for smoke-testing the deployed FastAPI service.
"""

import json
from json import JSONDecodeError
from datetime import datetime, timezone
from typing import Any, Callable
from urllib import error, request

DEFAULT_PREDICT_PAYLOAD = {
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

RequestFn = Callable[[str, str, dict[str, Any] | None, float], tuple[int, Any]]


def normalize_base_url(base_url: str) -> str:
    if not (cleaned := base_url.strip().rstrip("/")):
        raise ValueError("Base deployment URL cannot be empty.")
    return cleaned


def build_endpoint_url(base_url: str, path: str) -> str:
    normalized_base_url = normalize_base_url(base_url)
    normalized_path = path if path.startswith("/") else f"/{path}"
    return f"{normalized_base_url}{normalized_path}"


def request_json(
    url: str,
    method: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> tuple[int, Any]:
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = request.Request(
        url, data=body, headers=headers, method=method.upper()
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            raw_body = response.read().decode("utf-8")
            parsed = _parse_response_body(raw_body)
            return int(getattr(response, "status", 200)), parsed
    # pragma: no cover - exercised via integration
    except error.HTTPError as exc:
        raw_body = exc.read().decode("utf-8")
        parsed = _parse_response_body(raw_body)
        return exc.code, parsed
    except TimeoutError as exc:
        raise TimeoutError(
            f"Request to {url} timed out after {timeout:.1f}s. "
            "Check whether the Render service is live and responding."
        ) from exc
    except error.URLError as exc:
        raise ConnectionError(
            f"Could not reach {url}. Reason: {exc.reason}"
        ) from exc


def _parse_response_body(raw_body: str) -> dict[str, Any]:
    if not raw_body:
        return {}

    try:
        parsed = json.loads(raw_body)
    except JSONDecodeError:
        snippet = raw_body.strip()
        return {
            "raw_text": snippet[:500],
            "is_json": False,
        }

    if isinstance(parsed, dict):
        return parsed

    return {
        "json_value": parsed,
        "is_json": True,
    }


def verify_deployment(
    base_url: str,
    *,
    payload: dict[str, Any] | None = None,
    expected_source: str = "wandb",
    expected_alias: str = "prod",
    timeout: float = 30.0,
    request_fn: RequestFn = request_json,
) -> dict[str, Any]:
    health_url = build_endpoint_url(base_url, "/health")
    predict_url = build_endpoint_url(base_url, "/predict")
    predict_payload = payload or DEFAULT_PREDICT_PAYLOAD

    health_status, health_body = request_fn(health_url, "GET", None, timeout)
    if health_status != 200:
        raise ValueError(f"/health returned status {health_status}")
    if health_body.get("status") != "ok":
        raise ValueError(f"/health did not report ok status: {health_body}")
    if (
        health_body.get("model_loaded") is not True
    ):
        raise ValueError(
            f"/health indicates the model is not loaded: {health_body}"
        )
    if expected_source and health_body.get("model_source") != expected_source:
        raise ValueError(
            "Health endpoint reported an unexpected model source: "
            f"{health_body.get('model_source')!r}"
        )

    predict_status, predict_body = request_fn(
        predict_url,
        "POST",
        predict_payload,
        timeout,
    )
    if predict_status != 200:
        raise ValueError(f"/predict returned status {predict_status}")

    predictions = predict_body.get(
        "predictions"
    )
    if not isinstance(predictions, list) or not predictions:
        raise ValueError(
            f"/predict returned no usable predictions: {predict_body}"
        )
    if expected_source and predict_body.get("model_source") != expected_source:
        raise ValueError(
            "Predict endpoint reported an unexpected model source: "
            f"{predict_body.get('model_source')!r}"
        )

    model_reference = str(predict_body.get("model_reference", "")).strip()
    if expected_alias and not model_reference.endswith(f":{expected_alias}"):
        raise ValueError(
            "Predict endpoint did not confirm the expected promoted alias: "
            f"{model_reference!r}"
        )

    return {
        "base_url": normalize_base_url(base_url),
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_source": expected_source,
        "expected_alias": expected_alias,
        "health": {
            "url": health_url,
            "status_code": health_status,
            "body": health_body,
        },
        "predict": {
            "url": predict_url,
            "status_code": predict_status,
            "prediction_count": len(predictions),
            "model_reference": model_reference,
        },
    }
