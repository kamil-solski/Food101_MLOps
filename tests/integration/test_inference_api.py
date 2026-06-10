"""
Integration tests for the FastAPI inference API.

Strategy: patch `resolve_and_load` before `inference_api.main` is imported so
the module-level `SESSION, META = resolve_and_load()` call uses our mock.
The TestClient then exercises real route logic against the mocked ONNX session.

Run with: pytest tests/integration/ -m integration
"""
import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MOCK_META = {
    "model_name": "test_model",
    "version": "1",
    "alias": "champion",
    "image_size": 16,
    "top_k": 2,
    "classes": ["apple", "banana"],
    "output_is_logits": True,
    "preprocess": {"scale_to_unit": True},
    "input_name": "input",
}


@pytest.fixture(scope="module")
def inference_client():
    """
    FastAPI TestClient backed by a mocked ONNX session.

    The mock session returns [apple_logit=1.0, banana_logit=2.0], so banana
    should always be the top prediction after softmax.
    """
    mock_session = MagicMock()
    mock_session.run.return_value = [np.array([1.0, 2.0], dtype=np.float32)]
    input_mock = MagicMock()
    input_mock.name = "input"
    mock_session.get_inputs.return_value = [input_mock]

    # Remove any cached inference_api.main so the patched resolve_and_load is
    # picked up when main.py is (re-)imported inside the patch context.
    sys.modules.pop("inference_api.main", None)

    with patch(
        "inference_api.inference.loader.resolve_and_load",
        return_value=(mock_session, _MOCK_META),
    ):
        from inference_api.main import app
        from fastapi.testclient import TestClient

        yield TestClient(app), mock_session


@pytest.fixture
def jpeg_bytes():
    """Minimal valid JPEG image (16×16 red square)."""
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color=(200, 50, 50)).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def png_bytes():
    """Minimal valid PNG image."""
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color=(50, 200, 50)).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# /health endpoint
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestHealthEndpoint:
    def test_returns_200(self, inference_client):
        client, _ = inference_client
        response = client.get("/health")
        assert response.status_code == 200

    def test_body_has_required_fields(self, inference_client):
        client, _ = inference_client
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["model_name"] == _MOCK_META["model_name"]
        assert body["model_version"] == _MOCK_META["version"]
        assert body["alias"] == _MOCK_META["alias"]


# ---------------------------------------------------------------------------
# /predict endpoint
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestPredictEndpoint:
    def test_jpeg_returns_200(self, inference_client, jpeg_bytes):
        client, _ = inference_client
        resp = client.post(
            "/predict",
            files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
        )
        assert resp.status_code == 200

    def test_png_returns_200(self, inference_client, png_bytes):
        client, _ = inference_client
        resp = client.post(
            "/predict",
            files={"file": ("photo.png", png_bytes, "image/png")},
        )
        assert resp.status_code == 200

    def test_response_structure(self, inference_client, jpeg_bytes):
        client, _ = inference_client
        body = client.post(
            "/predict",
            files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
        ).json()

        assert "request_id" in body
        assert "predictions" in body
        assert "latency_ms" in body
        assert "served_alias" in body
        assert "model_version" in body

    def test_predictions_count_respects_top_k(self, inference_client, jpeg_bytes):
        client, _ = inference_client
        preds = client.post(
            "/predict",
            files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
        ).json()["predictions"]

        assert len(preds) <= _MOCK_META["top_k"]

    def test_prediction_fields(self, inference_client, jpeg_bytes):
        client, _ = inference_client
        preds = client.post(
            "/predict",
            files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
        ).json()["predictions"]

        for pred in preds:
            assert "class" in pred
            assert "prob" in pred
            assert "id" in pred
            assert pred["class"] in _MOCK_META["classes"]
            assert 0.0 <= pred["prob"] <= 1.0

    def test_top_prediction_is_banana(self, inference_client, jpeg_bytes):
        """Mock logits [1.0, 2.0] → banana (index 1) has higher softmax prob."""
        client, _ = inference_client
        preds = client.post(
            "/predict",
            files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
        ).json()["predictions"]

        assert preds[0]["class"] == "banana"

    def test_unsupported_file_type_returns_415(self, inference_client):
        client, _ = inference_client
        resp = client.post(
            "/predict",
            files={"file": ("doc.pdf", b"%PDF fake content", "application/pdf")},
        )
        assert resp.status_code == 415

    def test_headers_contain_model_info(self, inference_client, jpeg_bytes):
        client, _ = inference_client
        resp = client.post(
            "/predict",
            files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
        )
        assert resp.headers.get("x-model-alias") == _MOCK_META["alias"]
        assert resp.headers.get("x-model-version") == _MOCK_META["version"]
