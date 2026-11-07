import io
from typing import Optional

import torch
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import CLIPProcessor, CLIPModel

app = FastAPI(title="Visual Product Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = "openai/clip-vit-base-patch32"

processor = None
model = None


def load_model():
    global processor, model

    if model is None:
        processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        model = CLIPModel.from_pretrained(MODEL_NAME)
        model.eval()

    return processor, model


def read_image(data):
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )


@app.get("/")
def home():
    return {
        "status": "online",
        "service": "CLIP Visual Search API"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/rank")
async def rank_images(
    reference: UploadFile = File(...),
    candidate1: UploadFile = File(...),
    candidate2: Optional[UploadFile] = File(None),
    candidate3: Optional[UploadFile] = File(None)
):
    files = [candidate1, candidate2, candidate3]
    files = [f for f in files if f is not None]

    reference_image = read_image(await reference.read())

    images = [reference_image]
    filenames = []

    for file in files:
        images.append(read_image(await file.read()))
        filenames.append(file.filename or "image")

    image_processor, clip_model = load_model()

    inputs = image_processor(
        images=images,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = clip_model.vision_model(
            pixel_values=inputs["pixel_values"]
        )

        features = clip_model.visual_projection(
            outputs.pooler_output
        )

        features = features / features.norm(
            dim=-1,
            keepdim=True
        )

        similarities = torch.matmul(
            features[1:],
            features[0]
        )

    results = []

    for filename, similarity in zip(
        filenames,
        similarities.tolist()
    ):
        results.append({
            "filename": filename,
            "similarity_percent": round(
                max(0.0, min(1.0, similarity)) * 100,
                2
            )
        })

    results.sort(
        key=lambda item: item["similarity_percent"],
        reverse=True
    )

    return {
        "count": len(results),
        "results": results
    }
