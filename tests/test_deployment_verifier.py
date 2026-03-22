import io

import pytest  # type: ignore
from urllib import error

from src.deployment_verifier import build_endpoint_url
from src.deployment_verifier import normalize_base_url
from src.deployment_verifier import _parse_response_body
from src.deployment_verifier import request_json
from src.deployment_verifier import verify_deployment


def test_normalize_base_url_rejects_empty_value():
    with pytest.raises(ValueError, match="cannot be empty"):
        normalize_base_url("   ")


def test_build_endpoint_url_trims_trailing_slash():
    assert build_endpoint_url("https://example.com/", "/health") == (
        "https://example.com/health"
    )


def test_request_json_posts_json_payload(monkeypatch):
    class FakeResponse:
        status = 201

        def read(self):
            return b'{"ok": true}'

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    seen = {}

    def fake_urlopen(req, timeout):
        seen["content_type"] = req.headers.get("Content-type")
        seen["body"] = req.data
        return FakeResponse()

    monkeypatch.setattr("src.deployment_verifier.request.urlopen", fake_urlopen)

    status, body = request_json(
        "https://service.onrender.com/predict",
        "POST",
        payload={"hello": "world"},
    )

    assert status == 201
    assert body == {"ok": True}
    assert seen["content_type"] == "application/json"
    assert seen["body"] is not None


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


def test_request_json_handles_json_http_error(monkeypatch):
    def fake_urlopen(req, timeout):
        raise error.HTTPError(
            req.full_url,
            503,
            "Unavailable",
            hdrs=None,
            fp=io.BytesIO(b'{"detail":"down"}'),
        )

    monkeypatch.setattr("src.deployment_verifier.request.urlopen", fake_urlopen)

    status, body = request_json("https://service.onrender.com/health", "GET")

    assert status == 503
    assert body == {"detail": "down"}


def test_request_json_raises_clear_timeout(monkeypatch):
    def fake_urlopen(req, timeout):
        raise TimeoutError("slow")

    monkeypatch.setattr("src.deployment_verifier.request.urlopen", fake_urlopen)

    with pytest.raises(TimeoutError, match="timed out"):
        request_json("https://service.onrender.com/health", "GET", timeout=3.0)


def test_request_json_raises_clear_connection_error(monkeypatch):
    def fake_urlopen(req, timeout):
        raise error.URLError("dns failed")

    monkeypatch.setattr("src.deployment_verifier.request.urlopen", fake_urlopen)

    with pytest.raises(ConnectionError, match="Could not reach"):
        request_json("https://service.onrender.com/health", "GET")


def test_parse_response_body_handles_empty_and_scalar_json():
    assert _parse_response_body("") == {}
    assert _parse_response_body("[1, 2]") == {
        "json_value": [1, 2],
        "is_json": True,
    }


def test_verify_deployment_rejects_bad_health_and_predict_states():
    def health_status_bad(url, method, payload, timeout):
        return 500, {}

    with pytest.raises(ValueError, match="/health returned status 500"):
        verify_deployment("https://service.onrender.com", request_fn=health_status_bad)

    def health_body_bad(url, method, payload, timeout):
        return 200, {"status": "bad", "model_loaded": True, "model_source": "wandb"}

    with pytest.raises(ValueError, match="did not report ok status"):
        verify_deployment("https://service.onrender.com", request_fn=health_body_bad)

    def model_not_loaded(url, method, payload, timeout):
        return 200, {"status": "ok", "model_loaded": False, "model_source": "wandb"}

    with pytest.raises(ValueError, match="did not confirm a loaded model"):
        verify_deployment("https://service.onrender.com", request_fn=model_not_loaded)

    def health_source_bad(url, method, payload, timeout):
        return 200, {"status": "ok", "model_loaded": True, "model_source": "local"}

    with pytest.raises(ValueError, match="unexpected model source"):
        verify_deployment("https://service.onrender.com", request_fn=health_source_bad)


def test_verify_deployment_rejects_bad_predict_states():
    def predict_status_bad(url, method, payload, timeout):
        if url.endswith("/health"):
            return 200, {
                "status": "ok",
                "model_loaded": True,
                "model_source": "wandb",
            }
        return 500, {}

    with pytest.raises(ValueError, match="/predict returned status 500"):
        verify_deployment("https://service.onrender.com", request_fn=predict_status_bad)

    def no_predictions(url, method, payload, timeout):
        if url.endswith("/health"):
            return 200, {
                "status": "ok",
                "model_loaded": True,
                "model_source": "wandb",
            }
        return 200, {
            "predictions": [],
            "model_source": "wandb",
            "model_reference": "group8/project/model:prod",
        }

    with pytest.raises(ValueError, match="no usable predictions"):
        verify_deployment("https://service.onrender.com", request_fn=no_predictions)

    def predict_source_bad(url, method, payload, timeout):
        if url.endswith("/health"):
            return 200, {
                "status": "ok",
                "model_loaded": True,
                "model_source": "wandb",
            }
        return 200, {
            "predictions": [1.0],
            "model_source": "local",
            "model_reference": "group8/project/model:prod",
        }

    with pytest.raises(ValueError, match="unexpected model source"):
        verify_deployment("https://service.onrender.com", request_fn=predict_source_bad)
