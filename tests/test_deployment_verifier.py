import io

import pytest  # type: ignore
from urllib import error

from src.deployment_verifier import build_endpoint_url
from src.deployment_verifier import normalize_base_url
from src.deployment_verifier import request_json
from src.deployment_verifier import verify_deployment


def test_normalize_base_url_rejects_empty_value():
    with pytest.raises(ValueError, match="cannot be empty"):
        normalize_base_url("   ")


def test_build_endpoint_url_trims_trailing_slash():
    assert build_endpoint_url("https://example.com/", "/health") == (
        "https://example.com/health"
    )


def test_verify_deployment_confirms_prod_alias():
    def fake_request(url, method, payload, timeout):
        if url.endswith("/health"):
            return 200, {
                "status": "ok",
                "model_loaded": True,
                "model_source": "wandb",
            }
        if url.endswith("/predict"):
            return 200, {
                "predictions": [42.0],
                "model_source": "wandb",
                "model_reference": "group8/project/model:prod",
            }
        raise AssertionError(f"Unexpected URL: {url}")

    result = verify_deployment(
        "https://service.onrender.com/",
        request_fn=fake_request,
    )

    assert result["base_url"] == "https://service.onrender.com"
    assert result["predict"]["prediction_count"] == 1
    assert result["predict"]["model_reference"].endswith(":prod")


def test_verify_deployment_rejects_wrong_alias():
    def fake_request(url, method, payload, timeout):
        if url.endswith("/health"):
            return 200, {
                "status": "ok",
                "model_loaded": True,
                "model_source": "wandb",
            }
        if url.endswith("/predict"):
            return 200, {
                "predictions": [42.0],
                "model_source": "wandb",
                "model_reference": "group8/project/model:latest",
            }
        raise AssertionError(f"Unexpected URL: {url}")

    with pytest.raises(ValueError, match="expected promoted alias"):
        verify_deployment(
            "https://service.onrender.com",
            request_fn=fake_request,
        )


def test_request_json_handles_non_json_http_error(monkeypatch):
    def fake_urlopen(req, timeout):
        raise error.HTTPError(
            req.full_url,
            404,
            "Not Found",
            hdrs=None,
            fp=io.BytesIO(b"<html>not json</html>"),
        )

    monkeypatch.setattr("src.deployment_verifier.request.urlopen", fake_urlopen)

    status, body = request_json("https://service.onrender.com/health", "GET")

    assert status == 404
    assert body["is_json"] is False
    assert "not json" in body["raw_text"]


def test_request_json_raises_clear_timeout(monkeypatch):
    def fake_urlopen(req, timeout):
        raise TimeoutError("slow")

    monkeypatch.setattr("src.deployment_verifier.request.urlopen", fake_urlopen)

    with pytest.raises(TimeoutError, match="timed out"):
        request_json("https://service.onrender.com/health", "GET", timeout=3.0)
