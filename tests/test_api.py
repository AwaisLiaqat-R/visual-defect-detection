"""
Integration and Unit Tests for FastAPI Defect Detection Endpoints.
"""
import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from api.app import app
import api.app as api_module
from src.infer import DefectInferenceEngine

@pytest.fixture(scope="module")
def client():
    """Initializes FastAPI test client and ensures engine instance is loaded."""
    if api_module.engine is None:
        api_module.engine = DefectInferenceEngine()
    with TestClient(app) as test_client:
        yield test_client

def create_test_image_bytes(color=(200, 200, 200), format="PNG"):
    """Helper creating in-memory dummy test image bytes."""
    img = Image.new("RGB", (100, 100), color=color)
    buf = io.BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    return buf.getvalue()

def test_root_endpoint(client):
    """Verifies service landing page responds with 200 OK and valid links."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Visual Defect Detection API"
    assert "docs_url" in data

def test_health_endpoint(client):
    """Verifies health check endpoint returns 200 OK and system status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "model_loaded" in data
    assert "operating_threshold" in data

def test_predict_endpoint_valid_image(client):
    """Verifies single image inference on a valid PNG image upload."""
    img_bytes = create_test_image_bytes()
    response = client.post(
        "/predict",
        files={"file": ("sample.png", img_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] in ["normal", "defective"]
    assert data["predicted_label"] in [0, 1]
    assert 0.0 <= data["confidence"] <= 1.0
    assert "probabilities" in data
    assert "normal" in data["probabilities"]
    assert "defective" in data["probabilities"]
    assert "latency_ms" in data
    assert data["image_info"]["width"] == 100
    assert data["image_info"]["height"] == 100

def test_predict_endpoint_invalid_mime_type(client):
    """Verifies 415 error is returned when uploading a non-image file (e.g. text/plain)."""
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"plain text content", "text/plain")}
    )
    assert response.status_code == 415
    assert "Unsupported file type" in response.json()["detail"]

def test_predict_endpoint_corrupted_image(client):
    """Verifies 400 error is returned when uploading damaged image bytes."""
    corrupted_bytes = b"NOT_A_VALID_IMAGE_BYTE_SEQUENCE_12345"
    response = client.post(
        "/predict",
        files={"file": ("broken.png", corrupted_bytes, "image/png")}
    )
    assert response.status_code == 400
    assert "Corrupted or invalid image" in response.json()["detail"]

def test_predict_endpoint_with_custom_threshold(client):
    """Verifies inference respects optional threshold query parameter."""
    img_bytes = create_test_image_bytes()
    response = client.post(
        "/predict?threshold=0.85",
        files={"file": ("sample.png", img_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["threshold_used"] == 0.85

def test_batch_predict_endpoint(client):
    """Verifies multi-image batch prediction endpoint."""
    img1 = create_test_image_bytes()
    img2 = create_test_image_bytes(color=(50, 50, 50))
    
    response = client.post(
        "/batch-predict",
        files=[
            ("files", ("part1.png", img1, "image/png")),
            ("files", ("part2.png", img2, "image/png"))
        ]
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_images"] == 2
    assert len(data["predictions"]) == 2
    assert "defect_rate" in data
    assert "total_batch_latency_ms" in data

def test_metrics_endpoint(client):
    """Verifies telemetry metrics endpoint tracks request count and latency."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_inferences" in data
    assert "defect_rate_percentage" in data
    assert "average_latency_ms" in data
