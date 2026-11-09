"""Safely import the legacy Colab CLIP/DINO vectors into the SQLite catalog.

Expected files: listings.json, clip_vectors.npy (N, 512), dino_vectors.npy (N, 384).
The script validates the complete input before writing anything to the database.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np


def normalize_embeddings(values, name, expected_dim):
    """Return finite, non-zero, row-normalized float32 embeddings."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != expected_dim:
        raise ValueError(
            f"{name} must have shape (N, {expected_dim}); got {values.shape}."
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
        raise ValueError(f"{name} contains zero-length or invalid vectors.")
    return values / norms


def import_index(index_dir, db_path):
    root = Path(index_dir)
    required = [root / "listings.json", root / "clip_vectors.npy", root / "dino_vectors.npy"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing index files:\n" + "\n".join(missing))

    try:
        with required[0].open("r", encoding="utf-8") as handle:
            listings = json.load(handle)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SystemExit(f"Invalid listings.json: {exc}") from exc
    if not isinstance(listings, list):
        raise SystemExit("listings.json must contain a JSON array.")

    try:
        clip_raw = np.load(required[1], mmap_mode="r", allow_pickle=False)
        dino_raw = np.load(required[2], mmap_mode="r", allow_pickle=False)
        if clip_raw.ndim != 2 or dino_raw.ndim != 2:
            raise ValueError("Embedding files must be 2D arrays.")
        if len(clip_raw) != len(dino_raw) or len(clip_raw) != len(listings):
            raise ValueError(
                f"Row count mismatch: listings={len(listings)}, "
                f"CLIP={len(clip_raw)}, DINO={len(dino_raw)}"
            )
        # Validate all vectors before opening the database for writes.
        clip = normalize_embeddings(clip_raw, "CLIP embeddings", 512)
        dino = normalize_embeddings(dino_raw, "DINO embeddings", 384)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Invalid embedding index: {exc}") from exc

    # Delay importing api until the source has passed validation.
    os.environ["VISUAL_SEARCH_DB"] = str(db_path)
    import api
    # api may already be imported by a caller (notably tests or notebooks),
    # so update the module constant as well as the environment variable.
    api.DB_PATH = str(db_path)

    api.init_db()
    inserted = skipped = 0
    seen_indices = set()
    with api.db_lock, api.connect_db() as db:
        for i, item in enumerate(listings):
            if not isinstance(item, dict):
                skipped += 1
                continue
            try:
                listing_index = int(item.get("listing_index", i))
            except (TypeError, ValueError, OverflowError):
                skipped += 1
                continue
            if listing_index < 0 or listing_index >= len(clip) or listing_index in seen_indices:
                skipped += 1
                continue
            seen_indices.add(listing_index)

            listing_id = str(item.get("listing_id") or "").strip()
            title = str(item.get("title") or "").strip()
            price = str(item.get("price") or "").strip()
            listing_url = str(item.get("listing_url") or "").strip()
            image_url = str(item.get("image_url") or "").strip()
            if not image_url:
                image_url = f"legacy-index:{listing_index}"

            exists = db.execute(
                "SELECT id FROM listings WHERE image_url = ?", (image_url,)
            ).fetchone()
            if exists:
                skipped += 1
                continue

            db.execute(
                """INSERT INTO listings
                (listing_id,title,price,listing_url,image_url,metadata_json,
                 clip_vector,dino_vector,clip_dim,dino_dim)
                VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    listing_id, title, price, listing_url, image_url,
                    json.dumps(item, ensure_ascii=False),
                    api.vector_blob(clip[listing_index]),
                    api.vector_blob(dino[listing_index]),
                    512, 384,
                ),
            )
            inserted += 1
        db.commit()

    result = {
        "database": str(db_path),
        "legacy_images": len(listings),
        "inserted": inserted,
        "skipped": skipped,
        "clip_dimensions": 512,
        "dino_dimensions": 384,
    }
    print(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index-dir",
        default=os.getenv("LEGACY_INDEX_DIR", "/content/drive/MyDrive/visual_search_combined"),
    )
    parser.add_argument(
        "--db",
        default=os.getenv("VISUAL_SEARCH_DB", "visual_search.sqlite3"),
    )
    args = parser.parse_args()
    import_index(args.index_dir, args.db)


if __name__ == "__main__":
    main()
