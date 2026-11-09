"""Persistent CLIP + DINOv2 visual catalog API.

The service stores image embeddings and listing metadata in SQLite. It deliberately
indexes only images explicitly submitted by an authorized client; it does not crawl
marketplaces or bypass access controls.
"""
import io
import json
import hashlib
import math
import joblib
import os
import sqlite3
import threading
from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

CLIP_NAME = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
DINO_NAME = os.getenv("DINO_MODEL", "facebook/dinov2-small")
DB_PATH = os.getenv("VISUAL_SEARCH_DB", "visual_search.sqlite3")
API_KEY = os.getenv("VISUAL_SEARCH_API_KEY")
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(12 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "25000000"))
MAX_TOP_K = int(os.getenv("MAX_TOP_K", "100"))
RERANKER_PATH = os.getenv("RERANKER_PATH", "reranker.joblib")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

app = FastAPI(
    title="Visual Product Search API",
    version="1.0.0",
    description="Persistent image-first catalog search using CLIP and DINOv2.",
)
origins = [x.strip() for x in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",") if x.strip()]
if origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-API-Key"],
    )

model_lock = threading.RLock()
db_lock = threading.RLock()
clip_processor = clip_model = dino_processor = dino_model = None
reranker_model = None
reranker_mtime = None


def connect_db():
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with db_lock, connect_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS listings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                listing_id TEXT,
                title TEXT NOT NULL DEFAULT '',
                price TEXT NOT NULL DEFAULT '',
                listing_url TEXT NOT NULL DEFAULT '',
                image_url TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                clip_vector BLOB NOT NULL,
                dino_vector BLOB NOT NULL,
                clip_dim INTEGER NOT NULL,
                dino_dim INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(image_url)
            )
        """)
        db.execute("CREATE INDEX IF NOT EXISTS idx_listings_listing_id ON listings(listing_id)")
        db.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                listing_row_id INTEGER NOT NULL,
                label INTEGER NOT NULL CHECK(label BETWEEN 0 AND 3),
                clip_similarity REAL NOT NULL,
                dino_similarity REAL NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        db.commit()


@app.on_event("startup")
def startup():
    init_db()


def require_key(provided: Optional[str]):
    if API_KEY and provided != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


async def read_upload(upload: UploadFile) -> bytes:
    data = await upload.read(MAX_IMAGE_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Empty image.")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds upload size limit.")
    return data


def decode_image(data: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(data)) as source:
            source.verify()
        with Image.open(io.BytesIO(data)) as source:
            # Pillow's MAX_IMAGE_PIXELS threshold emits a warning before it
            # raises an error. Enforce our configured limit explicitly so
            # oversized images are rejected rather than merely logged.
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail=f"Image exceeds the {MAX_IMAGE_PIXELS}-pixel limit.",
                )
            return ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="Invalid or unsafe image file.")


def load_models():
    global clip_processor, clip_model, dino_processor, dino_model
    with model_lock:
        if clip_model is None:
            clip_processor = CLIPProcessor.from_pretrained(CLIP_NAME)
            clip_model = CLIPModel.from_pretrained(CLIP_NAME).to(DEVICE).eval()
        if dino_model is None:
            dino_processor = AutoImageProcessor.from_pretrained(DINO_NAME)
            dino_model = AutoModel.from_pretrained(DINO_NAME).to(DEVICE).eval()
    return clip_processor, clip_model, dino_processor, dino_model


def extract_vectors(images):
    cp, cm, dp, dm = load_models()
    with model_lock, torch.inference_mode():
        ci = cp(images=images, return_tensors="pt")
        ci = {k: v.to(DEVICE) for k, v in ci.items()}
        co = cm.vision_model(pixel_values=ci["pixel_values"])
        cv = cm.visual_projection(co.pooler_output)
        cv = torch.nn.functional.normalize(cv, dim=-1)

        di = dp(images=images, return_tensors="pt")
        di = {k: v.to(DEVICE) for k, v in di.items()}
        dout = dm(**di)
        # DINOv2 CLS token is a stable global descriptor.
        dv = dout.last_hidden_state[:, 0]
        dv = torch.nn.functional.normalize(dv, dim=-1)
    return cv.cpu().numpy().astype(np.float32), dv.cpu().numpy().astype(np.float32)


def get_reranker():
    global reranker_model, reranker_mtime
    if not os.path.isfile(RERANKER_PATH):
        return None
    mtime = os.path.getmtime(RERANKER_PATH)
    if reranker_model is None or reranker_mtime != mtime:
        with model_lock:
            reranker_model = joblib.load(RERANKER_PATH)
            reranker_mtime = mtime
    return reranker_model


def normalize_weights(clip_weight: float, dino_weight: float):
    """Validate and normalize finite, non-negative search weights."""
    if not math.isfinite(clip_weight) or not math.isfinite(dino_weight):
        raise HTTPException(status_code=400, detail="Weights must be finite numbers.")
    if clip_weight < 0 or dino_weight < 0 or clip_weight + dino_weight <= 0:
        raise HTTPException(status_code=400, detail="Weights must be non-negative and not both zero.")
    # Scale before adding: two finite, very large floats can overflow their sum.
    scale = max(clip_weight, dino_weight)
    scaled_clip = clip_weight / scale
    scaled_dino = dino_weight / scale
    total = scaled_clip + scaled_dino
    return scaled_clip / total, scaled_dino / total


def vector_blob(vector):
    return np.asarray(vector, dtype=np.float32).reshape(-1).tobytes()


def row_vector(blob, dim):
    """Decode one stored embedding and reject truncated or corrupted blobs."""
    if not isinstance(dim, int) or dim <= 0:
        raise ValueError("Embedding dimension must be a positive integer.")
    expected_bytes = dim * np.dtype(np.float32).itemsize
    if blob is None or len(blob) != expected_bytes:
        raise ValueError(
            f"Invalid embedding blob length: expected {expected_bytes} bytes."
        )
    vector = np.frombuffer(blob, dtype=np.float32, count=dim).copy()
    if not np.isfinite(vector).all():
        raise ValueError("Embedding blob contains NaN or infinite values.")
    return vector


def public_listing(row, score=None):
    item = {
        "id": row["id"],
        "listing_id": row["listing_id"],
        "title": row["title"],
        "price": row["price"],
        "listing_url": row["listing_url"],
        "image_url": row["image_url"],
    }
    if score is not None:
        item["similarity"] = round(float(score), 5)
        item["similarity_percent"] = round(max(0.0, min(1.0, float(score))) * 100, 2)
    return item


@app.get("/")
def home():
    return {"status": "online", "service": "CLIP + DINOv2 Visual Search", "version": "1.0.0"}


@app.get("/health")
def health():
    with connect_db() as db:
        count = db.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
    return {
        "status": "ok",
        "device": DEVICE,
        "clip_loaded": clip_model is not None,
        "dino_loaded": dino_model is not None,
        "indexed_images": count,
    }


@app.get("/stats")
def stats(x_api_key: Optional[str] = Header(None)):
    require_key(x_api_key)
    with connect_db() as db:
        count = db.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
        unique_ids = db.execute(
            "SELECT COUNT(DISTINCT listing_id) FROM listings WHERE listing_id IS NOT NULL AND listing_id != ''"
        ).fetchone()[0]
    return {"indexed_images": count, "unique_listing_ids": unique_ids, "database": DB_PATH}


@app.post("/ingest")
async def ingest(
    image: UploadFile = File(...),
    metadata: str = Form("{}"),
    x_api_key: Optional[str] = Header(None),
):
    require_key(x_api_key)
    try:
        meta = json.loads(metadata)
        if not isinstance(meta, dict):
            raise ValueError("metadata must be an object")
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(status_code=400, detail="metadata must be a JSON object.")

    raw = await read_upload(image)
    decoded = decode_image(raw)
    try:
        clip, dino = extract_vectors([decoded])
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Embedding extraction failed.") from exc

    image_url = str(meta.get("image_url") or "").strip()
    listing_id = str(meta.get("listing_id") or "").strip()
    title = str(meta.get("title") or "").strip()
    price = str(meta.get("price") or "").strip()
    listing_url = str(meta.get("listing_url") or "").strip()
    if not image_url:
        # A stable key is required for deduplication; clients should provide source image URL.
        image_url = f"upload:{listing_id}:{hashlib.sha256(raw).hexdigest()}"

    with db_lock, connect_db() as db:
        existing = db.execute("SELECT id FROM listings WHERE image_url = ?", (image_url,)).fetchone()
        if existing:
            return {"status": "duplicate", "id": existing["id"], "image_url": image_url}
        cursor = db.execute(
            """INSERT INTO listings
            (listing_id,title,price,listing_url,image_url,metadata_json,clip_vector,dino_vector,clip_dim,dino_dim)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                listing_id, title, price, listing_url, image_url,
                json.dumps(meta, ensure_ascii=False),
                vector_blob(clip[0]), vector_blob(dino[0]),
                int(clip.shape[1]), int(dino.shape[1]),
            ),
        )
        db.commit()
        new_id = cursor.lastrowid
    return {"status": "indexed", "id": new_id, "image_url": image_url}


@app.post("/search")
async def search(
    reference: UploadFile = File(...),
    top_k: int = Form(20),
    clip_weight: float = Form(0.5),
    dino_weight: float = Form(0.5),
    x_api_key: Optional[str] = Header(None),
):
    require_key(x_api_key)
    if top_k < 1 or top_k > MAX_TOP_K:
        raise HTTPException(status_code=400, detail=f"top_k must be between 1 and {MAX_TOP_K}.")
    cw, dw = normalize_weights(clip_weight, dino_weight)

    query_image = decode_image(await read_upload(reference))
    try:
        qclip, qdino = extract_vectors([query_image])
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Query embedding failed.") from exc

    with db_lock, connect_db() as db:
        rows = db.execute("SELECT * FROM listings").fetchall()
    if not rows:
        return {"count": 0, "results": [], "message": "Catalog is empty. Ingest listings first."}

    valid_rows = []
    clip_vectors = []
    dino_vectors = []
    for row in rows:
        try:
            cv = row_vector(row["clip_vector"], row["clip_dim"])
            dv = row_vector(row["dino_vector"], row["dino_dim"])
        except (TypeError, ValueError):
            # Ignore damaged legacy rows rather than failing the entire search.
            continue
        if cv.shape != qclip[0].shape or dv.shape != qdino[0].shape:
            continue
        valid_rows.append(row)
        clip_vectors.append(cv)
        dino_vectors.append(dv)

    ranked = []
    if valid_rows:
        # Batched exact cosine search is efficient for small/medium catalogs.
        clip_matrix = np.stack(clip_vectors)
        dino_matrix = np.stack(dino_vectors)
        clip_scores = clip_matrix @ qclip[0]
        dino_scores = dino_matrix @ qdino[0]
        combined_scores = cw * clip_scores + dw * dino_scores
        ranked = [
            (float(combined_scores[i]), row, float(clip_scores[i]), float(dino_scores[i]))
            for i, row in enumerate(valid_rows)
        ]
    reranker = get_reranker()
    reranker_used = reranker is not None
    if reranker is not None and ranked:
        feature_matrix = np.asarray([[entry[2], entry[3]] for entry in ranked], dtype=np.float32)
        probabilities = reranker.predict_proba(feature_matrix)[:, 1]
        ranked = [
            (float(probabilities[i]), entry[1], entry[2], entry[3], entry[0])
            for i, entry in enumerate(ranked)
        ]
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        results = []
        for score, row, cs, ds, base_score in ranked[:top_k]:
            item = public_listing(row, base_score)
            item["reranker_score"] = round(float(score), 5)
            results.append(item)
    else:
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        results = [public_listing(row, score) for score, row, _, _ in ranked[:top_k]]
    return {
        "count": len(results),
        "model": {"clip": CLIP_NAME, "dino": DINO_NAME},
        "weights": {"clip": cw, "dino": dw},
        "reranker_used": reranker_used,
        "results": results,
    }


@app.post("/feedback")
async def submit_feedback(
    reference: UploadFile = File(...),
    listing_row_id: int = Form(...),
    label: int = Form(...),
    x_api_key: Optional[str] = Header(None),
):
    """Store explicit human relevance feedback (0=unrelated, 3=excellent match)."""
    require_key(x_api_key)
    if label not in (0, 1, 2, 3):
        raise HTTPException(status_code=400, detail="label must be 0, 1, 2 or 3.")
    with connect_db() as db:
        row = db.execute("SELECT * FROM listings WHERE id = ?", (listing_row_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Listing not found.")
    query_image = decode_image(await read_upload(reference))
    try:
        qclip, qdino = extract_vectors([query_image])
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Feedback embedding failed.") from exc
    cv = row_vector(row["clip_vector"], row["clip_dim"])
    dv = row_vector(row["dino_vector"], row["dino_dim"])
    if cv.shape != qclip[0].shape or dv.shape != qdino[0].shape:
        raise HTTPException(status_code=409, detail="Model dimensions differ from indexed catalog.")
    cs = float(np.dot(qclip[0], cv))
    ds = float(np.dot(qdino[0], dv))
    with db_lock, connect_db() as db:
        cursor = db.execute(
            "INSERT INTO feedback (listing_row_id,label,clip_similarity,dino_similarity) VALUES (?,?,?,?)",
            (listing_row_id, label, cs, ds),
        )
        db.commit()
    return {"status": "saved", "feedback_id": cursor.lastrowid, "label": label}


@app.get("/feedback/stats")
def feedback_stats(x_api_key: Optional[str] = Header(None)):
    require_key(x_api_key)
    with connect_db() as db:
        total = db.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        rows = db.execute("SELECT label, COUNT(*) AS count FROM feedback GROUP BY label").fetchall()
    return {"total": total, "labels": {str(row["label"]): row["count"] for row in rows}}


@app.post("/rank")
async def rank_images(
    reference: UploadFile = File(...),
    candidate1: UploadFile = File(...),
    candidate2: Optional[UploadFile] = File(None),
    candidate3: Optional[UploadFile] = File(None),
    x_api_key: Optional[str] = Header(None),
):
    require_key(x_api_key)
    candidates = [x for x in (candidate1, candidate2, candidate3) if x is not None]
    images = [decode_image(await read_upload(reference))]
    names = []
    for candidate in candidates:
        images.append(decode_image(await read_upload(candidate)))
        names.append(candidate.filename or "image")
    try:
        clip, dino = extract_vectors(images)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Embedding extraction failed.") from exc

    scores = []
    for i, name in enumerate(names, start=1):
        cs = float(np.dot(clip[0], clip[i]))
        ds = float(np.dot(dino[0], dino[i]))
        combined = 0.5 * cs + 0.5 * ds
        scores.append({
            "filename": name,
            "clip_similarity": round(cs, 5),
            "dino_similarity": round(ds, 5),
            "similarity": round(combined, 5),
            "similarity_percent": round(max(0.0, min(1.0, combined)) * 100, 2),
        })
    scores.sort(key=lambda x: x["similarity"], reverse=True)
    return {"count": len(scores), "results": scores}


@app.get("/listings")
def listings(limit: int = 50, offset: int = 0, x_api_key: Optional[str] = Header(None)):
    require_key(x_api_key)
    if limit < 1 or limit > 200 or offset < 0:
        raise HTTPException(status_code=400, detail="Use limit 1..200 and offset >= 0.")
    with connect_db() as db:
        rows = db.execute(
            "SELECT * FROM listings ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset)
        ).fetchall()
    return {"count": len(rows), "results": [public_listing(row) for row in rows]}


@app.delete("/listings/{listing_row_id}")
def delete_listing(listing_row_id: int, x_api_key: Optional[str] = Header(None)):
    require_key(x_api_key)
    with db_lock, connect_db() as db:
        cursor = db.execute("DELETE FROM listings WHERE id = ?", (listing_row_id,))
        db.commit()
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Listing not found.")
    return {"status": "deleted", "id": listing_row_id}
