# TraceAI

AI-assisted missing-person investigation **support** platform for Australian contexts. It analyses
witness reports and photos, plots sightings, and ranks leads for a human investigator to review.

> **Research prototype.** Public research datasets and synthetic profiles only. No real cases, no real
> CCTV or personal data. It is an investigative support tool, not an identification system.

## What it does

| Module | Approach |
|---|---|
| Vision | InsightFace / ArcFace embeddings, cosine similarity between a profile photo and a sighting photo |
| NLP | Extracts place (Australian gazetteer), clothing and time from free text; optional spaCy NER; hedging and certainty cues feed credibility |
| Geospatial | DBSCAN (haversine) clusters of strong sightings and an inferred movement corridor |
| Lead scoring | Weighted image, description, proximity, recency, credibility. Missing signals are dropped and weights renormalised. Identity evidence (face or description) scales the result, so a well-located report about someone else does not rank like a match. Impossible journeys and sightings that predate the last-known time score 0 |
| API / UI | FastAPI backend, Streamlit dashboard with a folium map |

## Run locally

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt insightface   # Windows; use .venv/bin on Linux/macOS
.venv/Scripts/python -m spacy download en_core_web_sm

# Seed synthetic data (first run downloads LFW, ~230 MB, and the face model)
TRACEAI_FACE_MODEL=buffalo_sc .venv/Scripts/python -m scripts.seed --reset

# API and dashboard
TRACEAI_FACE_MODEL=buffalo_sc .venv/Scripts/python -m uvicorn traceai.api.main:app --port 8010
TRACEAI_API_URL=http://localhost:8010 .venv/Scripts/python -m streamlit run dashboard/app.py
```

- The API defaults to port 8000 in the Dockerfile; use another port if something else holds it and set
  `TRACEAI_API_URL` to match.
- `TRACEAI_FACE_MODEL`: `buffalo_l` (default, most accurate, ~280 MB) or `buffalo_sc` (~15 MB).
- `TRACEAI_FACE_BACKEND=none` disables image scoring.
- `DATABASE_URL` defaults to SQLite in `data/`. Docker Compose uses PostgreSQL.

## Docker

```bash
docker compose up --build   # API :8000, dashboard :8501, PostgreSQL
```

Not yet tested (no Docker on the development machine).

## Tests

```bash
.venv/Scripts/python -m pytest -q tests
```

## Limitations

- The gazetteer is about 50 well-known places with approximate coordinates, not the full OpenStreetMap or
  ASGS extracts. `gazetteer.resolve` and `nearest` are the interface a full version would keep.
- Market-1501 is not used (it needs a Kaggle login). Cross-camera sightings are simulated by augmenting
  LFW images.
- Lead scores are heuristic and uncalibrated. Weights are in `traceai/scoring.py`.
- No authentication: do not expose it publicly.

## Data

Faces: [LFW](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset) via scikit-learn. All profile
details (names, ages, clothing, last-known places) are invented.
