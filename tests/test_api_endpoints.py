import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import api


def png_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (12, 12), (20, 40, 60)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DB_PATH", str(tmp_path / "api-test.sqlite3"))
    monkeypatch.setattr(api, "API_KEY", "test-secret")
    monkeypatch.setattr(api, "MAX_TOP_K", 10)

    def fake_extract_vectors(images):
        clip = np.zeros((len(images), 512), dtype=np.float32)
        dino = np.zeros((len(images), 384), dtype=np.float32)
        clip[:, 0] = 1.0
        dino[:, 0] = 1.0
        return clip, dino

    monkeypatch.setattr(api, "extract_vectors", fake_extract_vectors)
    monkeypatch.setattr(api, "get_reranker", lambda: None)
    with TestClient(api.app) as test_client:
        yield test_client


def test_ingest_search_and_feedback_round_trip(client):
    headers = {"X-API-Key": "test-secret"}
    image = png_bytes()
    files = {"image": ("item.png", image, "image/png")}
    metadata = {
        "listing_id": "listing-123",
        "title": "Test jacket",
        "price": "2500",
        "listing_url": "https://example.test/item/123",
        "image_url": "https://example.test/image/123.jpg",
    }

    response = client.post(
        "/ingest",
        headers=headers,
        files=files,
        data={"metadata": json.dumps(metadata)},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "indexed"
    row_id = response.json()["id"]

    duplicate = client.post(
        "/ingest",
        headers=headers,
        files=files,
        data={"metadata": __import__("json").dumps(metadata)},
    )
    assert duplicate.status_code == 200
    assert duplicate.json()["status"] == "duplicate"
    assert duplicate.json()["id"] == row_id

    search = client.post(
        "/search",
        headers=headers,
        files={"reference": ("query.png", image, "image/png")},
        data={"top_k": 5, "clip_weight": 0.5, "dino_weight": 0.5},
    )
    assert search.status_code == 200
    payload = search.json()
    assert payload["count"] == 1
    assert payload["results"][0]["id"] == row_id
    assert payload["results"][0]["title"] == "Test jacket"
    assert payload["reranker_used"] is False

    feedback = client.post(
        "/feedback",
        headers=headers,
        files={"reference": ("query.png", image, "image/png")},
        data={"listing_row_id": row_id, "label": 3},
    )
    assert feedback.status_code == 200
    assert feedback.json()["status"] == "saved"
    assert client.get("/feedback/stats", headers=headers).json() == {
        "total": 1,
        "labels": {"3": 1},
    }


def test_protected_endpoint_rejects_missing_api_key(client):
    response = client.get("/stats")
    assert response.status_code == 401


@pytest.mark.parametrize(
    "weights",
    [
        {"clip_weight": -1, "dino_weight": 2},
        {"clip_weight": 0, "dino_weight": 0},
        {"clip_weight": 1e309, "dino_weight": 1},
    ],
)
def test_search_rejects_invalid_weights(client, weights):
    response = client.post(
        "/search",
        headers={"X-API-Key": "test-secret"},
        files={"reference": ("query.png", png_bytes(), "image/png")},
        data={"top_k": 5, **weights},
    )
    assert response.status_code == 400


def test_search_rejects_top_k_out_of_range(client):
    response = client.post(
        "/search",
        headers={"X-API-Key": "test-secret"},
        files={"reference": ("query.png", png_bytes(), "image/png")},
        data={"top_k": 11},
    )
    assert response.status_code == 400
