# Visual Search — image-first product discovery

An image-first product-search prototype. The API stores CLIP and DINOv2 global image embeddings with listing metadata in SQLite, then ranks catalog entries against a query photograph. A Streamlit interface provides search, pair comparison and human relevance feedback.

## Implemented

- CLIP global image embeddings.
- DINOv2 global CLS-token image embeddings.
- Persistent SQLite catalog with duplicate detection by image URL.
- `POST /ingest`, `POST /search`, `POST /rank`, `POST /feedback`.
- Listing, health and feedback-statistics endpoints.
- Upload size/pixel limits, safe image decoding, optional API-key authentication and explicit CORS allowlist.
- Streamlit visual search UI with relevance labels (0–3).
- Optional Logistic Regression reranker trainer; it is not trained by default.
- Manifest V3 Chrome extension source for currently visible Avito listing cards.

## Run locally

Python 3.10+ is recommended.

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

Set `VISUAL_SEARCH_API_URL` and `VISUAL_SEARCH_API_KEY` for Streamlit, or enter the API address/key in its sidebar. The first request downloads CLIP and DINOv2 weights; this requires network access. CPU works but is slower; CUDA is used automatically when available.

## Docker Compose

Copy `.env.example` to `.env`, replace the placeholder with a newly generated secret, then run:

```bash
docker compose up --build -d
```

Open the Streamlit UI at `http://localhost:8501`. The API is available locally at `http://127.0.0.1:8000`. The SQLite database is stored in a named Docker volume. Do not publish the API port directly to the internet without HTTPS, authentication and suitable network controls.

## API

Image endpoints use multipart form data. When `VISUAL_SEARCH_API_KEY` is configured, send it in `X-API-Key`. If unset, the API runs without authentication; never expose that configuration publicly.

### Ingest

`POST /ingest`

Fields:
- `image`: image file.
- `metadata`: JSON object in a form field. Supported keys: `listing_id`, `title`, `price`, `listing_url`, `image_url`.

Example metadata:

```json
{"listing_id":"123","title":"Vintage jacket","price":"2500 ₽","listing_url":"https://example.com/item","image_url":"https://example.com/photo.jpg"}
```

### Search

`POST /search`

Fields: `reference` (query image), `top_k` (1–100), `clip_weight`, `dino_weight`. Scores are ranking signals, not probabilities of exact identity.

### Compare

`POST /rank`

Fields: `reference`, `candidate1`, optional `candidate2`, optional `candidate3`.

### Human feedback

`POST /feedback`: fields `reference`, `listing_row_id`, `label` (0=unrelated, 1=weak, 2=relevant, 3=excellent).

`GET /feedback/stats` returns label counts. After collecting at least 20 labels and at least 5 examples in each binary class, install development dependencies and run:

```bash
pip install -r requirements-dev.txt
python train_reranker.py
```

The script evaluates a holdout split, then saves `reranker.joblib`. The API automatically uses that model if present. Keep it only if independent evaluation shows a gain; feedback is not automatically changing CLIP or DINOv2.

### Other endpoints

- `GET /health` — service, model and catalog status.
- `GET /stats` — catalog counts.
- `GET /listings?limit=50&offset=0` — listing metadata.
- `DELETE /listings/{row_id}` — delete one row.
- `GET /feedback/stats` — feedback counts.

## Avito Chrome extension

Source code and installation steps are in [extension/](extension/). Load that folder unpacked at `chrome://extensions`, configure the API address/key in the popup, open an Avito results page, then submit the currently visible cards. Scroll normally and repeat for more results.

The extension does not navigate result pages, crawl hidden endpoints, bypass login/CAPTCHA/anti-bot controls, or promise full marketplace coverage. Avito DOM selectors and image hosts can change and require a live browser test.

## Import the existing Colab index

The legacy Drive index can be imported without re-embedding the 1,190 existing images. Make the three files available in the runtime where the API database lives, then run:

```bash
python migrate_colab_index.py --index-dir /path/to/visual_search_combined --db visual_search.sqlite3
```

The importer expects `listings.json`, `clip_vectors.npy` and `dino_vectors.npy` with matching row counts. It normalizes vectors, skips image URLs already in SQLite and prints inserted/skipped totals. Back up the SQLite database before importing.

## Data and persistence

SQLite path defaults to `visual_search.sqlite3`; override with `VISUAL_SEARCH_DB`. Persist this file on a durable volume and back it up. The existing Colab FAISS files are not automatically imported into this SQLite database.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `VISUAL_SEARCH_API_KEY` | unset | Shared API key; set for public deployments |
| `CORS_ALLOWED_ORIGINS` | empty | Comma-separated browser origins; empty disables CORS |
| `VISUAL_SEARCH_DB` | `visual_search.sqlite3` | SQLite database path |
| `CLIP_MODEL` | `openai/clip-vit-base-patch32` | CLIP model ID |
| `DINO_MODEL` | `facebook/dinov2-small` | DINOv2 model ID |
| `RERANKER_PATH` | `reranker.joblib` | Optional trained reranker |
| `MAX_IMAGE_BYTES` | 12 MiB | Upload size limit |
| `MAX_IMAGE_PIXELS` | 25 million | Pillow pixel limit |
| `MAX_TOP_K` | 100 | Maximum search results |

## Not implemented / still needs verification

- FAISS/ANN acceleration; search currently performs exact vector comparison over SQLite rows.
- DINOv2 patch-level matching.
- Import/migration of the existing Colab index and its 1,190 image vectors.
- Live end-to-end test of Avito selectors/image fetching and deployment of the API.
- Other marketplace connectors, size/fit-aware stylist recommendations, outfit assembly and virtual try-on.
- Relevance model quality has not been established; do not assume the reranker improves results.

Read [SECURITY.md](SECURITY.md). Never commit API keys, cookies, browser profiles or private tunnel credentials.
