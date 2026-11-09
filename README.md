# Visual Search

Image-first product discovery. The current baseline provides a CLIP image-ranking API and a Streamlit two-image comparison demo.

## Current status

- [x] CLIP pair comparison in Streamlit.
- [x] FastAPI endpoint to rank up to three candidates against a reference image.
- [x] API hardening: upload-size limits, image validation, optional API-key protection, explicit CORS allowlist, device selection and model inference locking.
- [ ] Persistent product/listing catalog and image metadata.
- [ ] DINOv2 global and patch embeddings with FAISS retrieval.
- [ ] Avito visible-page collector that reads only rendered cards during normal user browsing.
- [ ] Human relevance feedback and separately trained reranker.
- [ ] Outfit/stylist workflow with user-supplied size, fit, budget and style constraints.
- [ ] Deployment and end-to-end tests.

## Run locally

```bash
pip install -r requirements.txt
uvicorn api:app --host 127.0.0.1 --port 8000
```

For public deployment, set `VISUAL_SEARCH_API_KEY` and `CORS_ALLOWED_ORIGINS`. Keep secrets in environment variables, never in source code or browser-extension bundles.

## API

- `GET /` service metadata.
- `GET /health` health and model-load status.
- `POST /rank` multipart fields: `reference`, `candidate1`, optional `candidate2`, optional `candidate3`. Send `X-API-Key` when `VISUAL_SEARCH_API_KEY` is configured.

Similarity is a ranking signal, not a probability of an exact match.
