"""Import the existing Colab CLIP/DINO vector index into the SQLite catalog.

Expected files in --index-dir:
  listings.json
  clip_vectors.npy  (N, 512)
  dino_vectors.npy  (N, 384)

No model download or image re-embedding is needed. The vectors are normalized
before insertion. Existing image_url rows are skipped, never overwritten.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "--index-dir",
    default=os.getenv("LEGACY_INDEX_DIR", "/content/drive/MyDrive/visual_search_combined"),
)
parser.add_argument(
    "--db",
    default=os.getenv("VISUAL_SEARCH_DB", "visual_search.sqlite3"),
)
args = parser.parse_args()

os.environ["VISUAL_SEARCH_DB"] = args.db
import api  # noqa: E402

root = Path(args.index_dir)
required = [root / "listings.json", root / "clip_vectors.npy", root / "dino_vectors.npy"]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit("Missing index files:\n" + "\n".join(missing))

with (root / "listings.json").open("r", encoding="utf-8") as handle:
    listings = json.load(handle)
if not isinstance(listings, list):
    raise SystemExit("listings.json must contain a JSON array.")

clip = np.load(root / "clip_vectors.npy", mmap_mode="r")
dino = np.load(root / "dino_vectors.npy", mmap_mode="r")
if clip.ndim != 2 or dino.ndim != 2:
    raise SystemExit("Embedding files must be 2D arrays.")
if len(clip) != len(dino) or len(clip) != len(listings):
    raise SystemExit(
        f"Row count mismatch: listings={len(listings)}, CLIP={len(clip)}, DINO={len(dino)}"
    )

api.init_db()
clip = np.asarray(clip, dtype=np.float32)
dino = np.asarray(dino, dtype=np.float32)
clip_norms = np.linalg.norm(clip, axis=1, keepdims=True)
dino_norms = np.linalg.norm(dino, axis=1, keepdims=True)
clip = clip / np.maximum(clip_norms, 1e-12)
dino = dino / np.maximum(dino_norms, 1e-12)

inserted = 0
skipped = 0
with api.db_lock, api.connect_db() as db:
    for i, item in enumerate(listings):
        if not isinstance(item, dict):
            skipped += 1
            continue
        listing_index = int(item.get("listing_index", i))
        if listing_index < 0 or listing_index >= len(clip):
            skipped += 1
            continue

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
                int(clip.shape[1]), int(dino.shape[1]),
            ),
        )
        inserted += 1
    db.commit()

print({
    "database": args.db,
    "legacy_images": len(listings),
    "inserted": inserted,
    "skipped": skipped,
    "clip_dimensions": int(clip.shape[1]),
    "dino_dimensions": int(dino.shape[1]),
})
