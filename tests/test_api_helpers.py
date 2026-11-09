import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from PIL import Image
from fastapi import HTTPException

import api


def test_vector_blob_round_trip():
    original = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    restored = api.row_vector(api.vector_blob(original), 3)
    np.testing.assert_allclose(restored, original)


def test_decode_valid_image():
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (12, 34, 56)).save(buffer, format="PNG")
    decoded = api.decode_image(buffer.getvalue())
    assert decoded.mode == "RGB"
    assert decoded.size == (8, 8)


def test_decode_rejects_invalid_image():
    with pytest.raises(HTTPException) as error:
        api.decode_image(b"not an image")
    assert error.value.status_code == 400


def test_decode_rejects_images_over_pixel_limit(monkeypatch):
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (12, 34, 56)).save(buffer, format="PNG")
    monkeypatch.setattr(api, "MAX_IMAGE_PIXELS", 63)
    with pytest.raises(HTTPException) as error:
        api.decode_image(buffer.getvalue())
    assert error.value.status_code == 413


def test_database_initializes(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DB_PATH", str(tmp_path / "catalog.sqlite3"))
    api.init_db()
    with api.connect_db() as db:
        names = {
            row["name"]
            for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert "listings" in names


@pytest.mark.parametrize(
    "clip_weight,dino_weight,expected",
    [(0.5, 0.5, (0.5, 0.5)), (2.0, 1.0, (2 / 3, 1 / 3)), (0.0, 4.0, (0.0, 1.0))],
)
def test_normalize_weights(clip_weight, dino_weight, expected):
    assert api.normalize_weights(clip_weight, dino_weight) == pytest.approx(expected)


@pytest.mark.parametrize(
    "clip_weight,dino_weight",
    [(-1.0, 1.0), (0.0, 0.0), (float("nan"), 1.0), (1.0, float("inf")), (float("-inf"), 1.0)],
)
def test_normalize_weights_rejects_invalid_values(clip_weight, dino_weight):
    with pytest.raises(HTTPException) as error:
        api.normalize_weights(clip_weight, dino_weight)
    assert error.value.status_code == 400


def test_migration_normalizes_valid_embeddings():
    from migrate_colab_index import normalize_embeddings

    values = np.zeros((2, 512), dtype=np.float32)
    values[0, 0] = 3.0
    values[1, 1] = 4.0
    normalized = normalize_embeddings(values, "CLIP", 512)
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), [1.0, 1.0])


@pytest.mark.parametrize(
    "values,expected_dim",
    [
        (np.ones((2, 511), dtype=np.float32), 512),
        (np.ones((2, 385), dtype=np.float32), 384),
        (np.array([[np.nan] * 512], dtype=np.float32), 512),
        (np.zeros((1, 384), dtype=np.float32), 384),
        (np.array([[np.inf] * 384], dtype=np.float32), 384),
    ],
)
def test_migration_rejects_invalid_embeddings(values, expected_dim):
    from migrate_colab_index import normalize_embeddings

    with pytest.raises(ValueError):
        normalize_embeddings(values, "embedding", expected_dim)


def test_migration_uses_requested_database_even_when_api_is_loaded(tmp_path, monkeypatch):
    from migrate_colab_index import import_index

    source = tmp_path / "legacy"
    source.mkdir()
    listings = [
        {"listing_id": "a", "title": "Item A", "image_url": "https://example.test/a.jpg"},
        {"listing_id": "b", "title": "Item B", "image_url": "https://example.test/b.jpg"},
    ]
    (source / "listings.json").write_text(json.dumps(listings), encoding="utf-8")

    clip = np.zeros((2, 512), dtype=np.float32)
    clip[:, 0] = 1.0
    dino = np.zeros((2, 384), dtype=np.float32)
    dino[:, 0] = 1.0
    np.save(source / "clip_vectors.npy", clip)
    np.save(source / "dino_vectors.npy", dino)

    requested_db = tmp_path / "requested.sqlite3"
    monkeypatch.setattr(api, "DB_PATH", str(tmp_path / "stale.sqlite3"))

    result = import_index(source, requested_db)

    assert result["inserted"] == 2
    assert result["skipped"] == 0
    assert api.DB_PATH == str(requested_db)
    with api.connect_db() as db:
        rows = db.execute("SELECT listing_id, clip_dim, dino_dim FROM listings ORDER BY id").fetchall()
    assert [(row["listing_id"], row["clip_dim"], row["dino_dim"]) for row in rows] == [
        ("a", 512, 384),
        ("b", 512, 384),
    ]
    assert not (tmp_path / "stale.sqlite3").exists()
