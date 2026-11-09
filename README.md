# Visual Search — image-first product discovery

A practical starting point for visual product search. The API stores CLIP and DINOv2 image embeddings with listing metadata in SQLite, then ranks catalog entries against a query photograph. A Streamlit interface provides catalog search and pair comparison.

## Implemented

- CLIP global image embeddings.
- DINOv2 global (CLS-token) image embeddings.
- Persistent SQLite catalog; each ingested image stores title, price, listing URL, source image URL and model vectors.
- Duplicate detection by image URL.
- `POST /ingest`, `POST /search`, `POST /rank`, `GET /listings`, `GET /stats`, `GET /health`.
- Configurable API-key protection, CORS allowlist, image byte/pixel limits, EXIF orientation handling, safe image decoding.
- Streamlit UI for visual catalog search and pair comparison.

## Run locally

Python 3.10+ is recommended. Install dependencies and start the API:

```bash
pip install -r requirements.txt
export VISUAL_SEARCH_API_KEY="replace-with-a-long-random-secret"
export CORS_ALLOWED_ORIGINS="http://localhost:8501"
uvicorn api:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
streamlit run app.py
```

Set `VISUAL_SEARCH_API_URL` and `VISUAL_SEARCH_API_KEY` as environment variables for Streamlit, or enter the API address/key in its sidebar.

The first request downloads CLIP and DINOv2 weights. This can take time and requires network access. CPU works but is slower; CUDA is used automatically when available.

## API

All image endpoints use multipart form data. When `VISUAL_SEARCH_API_KEY` is configured, send it in `X-API-Key`. If the variable is unset, the API runs without authentication; do not expose that configuration publicly.

### Ingest one catalog image

`POST /ingest`

Fields:
- `image`: image file
- `metadata`: JSON object encoded as a form field. Supported keys: `listing_id`, `title`, `price`, `listing_url`, `image_url`.

Example metadata:

```json
{"listing_id":"123","title":"Vintage jacket","price":"2500 ₽","listing_url":"https://example.com/item","image_url":"https://example.com/photo.jpg"}
```

### Search the catalog

`POST /search`

Fields:
- `reference`: query image
- `top_k`: number of results (1–100; default 20)
- `clip_weight` and `dino_weight`: non-negative fusion weights (default 0.5 each)

Response includes title, price, listing URL, image URL and a combined similarity ranking score. Similarity is not a probability of exact identity.

### Compare candidate images

`POST /rank`

Fields: `reference`, `candidate1`, optional `candidate2`, optional `candidate3`.

### Catalog administration

- `GET /health` — service/model/database status.
- `GET /stats` — catalog counts.
- `GET /listings?limit=50&offset=0` — listing metadata.
- `DELETE /listings/{row_id}` — delete one row.

## Data and persistence

SQLite database path defaults to `visual_search.sqlite3`. Override with `VISUAL_SEARCH_DB`. Persist this file on a durable volume in deployment. Back it up before upgrades or migrations.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `VISUAL_SEARCH_API_KEY` | unset | Shared API key; required for public deployments |
| `CORS_ALLOWED_ORIGINS` | empty | Comma-separated browser origins; empty disables CORS |
| `VISUAL_SEARCH_DB` | `visual_search.sqlite3` | SQLite database path |
| `CLIP_MODEL` | `openai/clip-vit-base-patch32` | CLIP model ID |
| `DINO_MODEL` | `facebook/dinov2-small` | DINOv2 model ID |
| `MAX_IMAGE_BYTES` | 12 MiB | Upload size limit |
| `MAX_IMAGE_PIXELS` | 25 million | Pillow pixel limit |
| `MAX_TOP_K` | 100 | Maximum search result count |

## Important limitations / next milestones

- Search currently performs exact vector comparison over the SQLite catalog; FAISS/ANN indexing is not yet wired in. For larger catalogs, add an index and rebuild/update it transactionally.
- DINOv2 currently uses its global CLS descriptor. Patch-level matching and a learned reranker are future improvements.
- There is no automatic marketplace crawler. An Avito browser collector should only read cards rendered in the user's normal browser session and must not bypass CAPTCHA, authentication, anti-bot controls or rate limits.
- The API does not yet include a user-facing feedback-labeling workflow, multi-marketplace connectors, size/fit-aware outfit generation, or virtual try-on.
- Validate performance on a held-out labeled set before describing any model/reranker as an accuracy improvement.

## Security

Read [SECURITY.md](SECURITY.md). Never commit keys, cookies, browser profiles, or private tunnel credentials.
