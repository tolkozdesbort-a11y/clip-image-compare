import json

import numpy as np
import pytest

import api
from migrate_colab_index import import_index, normalize_embeddings


def test_normalize_embeddings_rejects_invalid_rows():
    with pytest.raises(ValueError, match="zero-length"):
        normalize_embeddings(np.zeros((1, 512), dtype=np.float32), "CLIP", 512)

    values = np.ones((2, 512), dtype=np.float32)
    normalized = normalize_embeddings(values, "CLIP", 512)
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1.0)


def test_import_index_is_idempotent_and_preserves_metadata(tmp_path, monkeypatch):
    index_dir = tmp_path / "legacy"
    index_dir.mkdir()
    db_path = tmp_path / "catalog.sqlite3"

    listings = [
        {
            "listing_index": 0,
            "listing_id": "item-1",
            "title": "Jacket",
            "price": "100",
            "listing_url": "https://example.test/1",
            "image_url": "https://example.test/1.jpg",
        },
        {
            "listing_index": 1,
            "listing_id": "item-2",
            "title": "Shoes",
            "price": "200",
            "listing_url": "https://example.test/2",
            "image_url": "https://example.test/2.jpg",
        },
    ]
    (index_dir / "listings.json").write_text(
        json.dumps(listings), encoding="utf-8"
    )
    clip = np.zeros((2, 512), dtype=np.float32)
    dino = np.zeros((2, 384), dtype=np.float32)
    clip[:, 0] = 1.0
    dino[:, 0] = 1.0
    np.save(index_dir / "clip_vectors.npy", clip)
    np.save(index_dir / "dino_vectors.npy", dino)

    monkeypatch.setattr(api, "DB_PATH", str(db_path))
    first = import_index(index_dir, db_path)
    second = import_index(index_dir, db_path)

    assert first["inserted"] == 2
    assert first["skipped"] == 0
    assert second["inserted"] == 0
    assert second["skipped"] == 2

    with api.connect_db() as db:
        rows = db.execute(
            "SELECT listing_id, title, clip_dim, dino_dim FROM listings ORDER BY id"
        ).fetchall()

    assert [(row["listing_id"], row["title"]) for row in rows] == [
        ("item-1", "Jacket"),
        ("item-2", "Shoes"),
    ]
    assert all(row["clip_dim"] == 512 and row["dino_dim"] == 384 for row in rows)


def test_import_index_rejects_row_count_mismatch_before_database_write(tmp_path, monkeypatch):
    index_dir = tmp_path / "broken"
    index_dir.mkdir()
    (index_dir / "listings.json").write_text("[]", encoding="utf-8")
    np.save(index_dir / "clip_vectors.npy", np.ones((1, 512), dtype=np.float32))
    np.save(index_dir / "dino_vectors.npy", np.ones((1, 384), dtype=np.float32))
    db_path = tmp_path / "must-not-exist.sqlite3"
    monkeypatch.setattr(api, "DB_PATH", str(db_path))

    with pytest.raises(SystemExit, match="Row count mismatch"):
        import_index(index_dir, db_path)

    assert not db_path.exists()
