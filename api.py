import io
import os
import threading
from typing import Optional

import torch
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(12 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "25000000"))
API_KEY = os.getenv("VISUAL_SEARCH_API_KEY")
ALLOWED_ORIGINS = [x.strip() for x in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",") if x.strip()]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

app = FastAPI(title="Visual Product Search API", version="0.2.0")
if ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key"],
    )

processor = None
model = None
model_lock = threading.Lock()


def load_model():
    global processor, model
    if model is None:
        with model_lock:
            if model is None:
                processor = CLIPProcessor.from_pretrained(MODEL_NAME)
                model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    return processor, model


def check_api_key(provided: Optional[str]):
    if API_KEY and provided != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


async def read_upload(upload: UploadFile) -> bytes:
    data = await upload.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds upload size limit.")
    if not data:
        raise HTTPException(status_code=400, detail="Empty image file.")
    return data


def decode_image(data: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(data)) as source:
            source.verify()
        with Image.open(io.BytesIO(data)) as source:
            return source.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="Invalid or unsafe image file.")


def embed_images(images):
    image_processor, clip_model = load_model()
    inputs = image_processor(images=images, return_tensors="pt")
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
    with model_lock, torch.inference_mode():
        vision = clip_model.vision_model(pixel_values=inputs["pixel_values"])
        features = clip_model.visual_projection(vision.pooler_output)
        features = torch.nn.functional.normalize(features, dim=-1)
    return features.cpu()


@app.get("/")
def home():
    return {"status": "online", "service": "CLIP Visual Search API", "version": "0.2.0"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "device": DEVICE}


@app.post("/rank")
async def rank_images(
    reference: UploadFile = File(...),
    candidate1: UploadFile = File(...),
    candidate2: Optional[UploadFile] = File(None),
    candidate3: Optional[UploadFile] = File(None),
    x_api_key: Optional[str] = Header(None),
):
    check_api_key(x_api_key)
    candidates = [item for item in (candidate1, candidate2, candidate3) if item]
    images = [decode_image(await read_upload(reference))]
    filenames = []
    for candidate in candidates:
        images.append(decode_image(await read_upload(candidate)))
        filenames.append(candidate.filename or "image")

    try:
        features = embed_images(images)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Image embedding failed.") from exc

    scores = torch.mv(features[1:], features[0]).tolist()
    results = [{
        "filename": name,
        "cosine_similarity": round(float(score), 5),
        "similarity_percent": round(max(0.0, min(1.0, float(score))) * 100, 2),
    } for name, score in zip(filenames, scores)]
    results.sort(key=lambda item: item["cosine_similarity"], reverse=True)
    return {"count": len(results), "model": MODEL_NAME, "results": results}
