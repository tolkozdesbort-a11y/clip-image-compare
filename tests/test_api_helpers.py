import io

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


def test_database_initializes(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DB_PATH", str(tmp_path / "catalog.sqlite3"))
    api.init_db()
    with api.connect_db() as db:
        names = {
            row["name"]
            for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert "listings" in names
