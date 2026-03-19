# Spotify Popularity Prediction Pipeline

**Author:** Group 8  
**Course:** MLOps: Master in Business Analytics and Data Science  
**Status:** Production-oriented final submission

-----

## Business Case

Music platforms need to decide which tracks deserve promotion in playlists, search surfaces, and recommendation modules. Relying only on historical engagement creates lag, reinforces incumbents, and slows down discovery. This project predicts track popularity from audio features so product, recommendation, and analytics teams can rank songs earlier and more consistently.

The business value is operational as much as predictive:
- Faster experimentation on ranking rules
- More reproducible model retraining and evaluation
- Safer deployment through CI, release discipline, and typed API contracts
- Better traceability through logs, W&B artifacts, and versioned serving

### Objective

Develop a supervised machine learning model to predict Spotify track popularity on a 0 to 100 scale using measurable audio features such as energy, danceability, valence, tempo, acousticness, and loudness. The objective is to identify which audio characteristics are associated with higher popularity and to turn that analysis into a reproducible ML product.

### Users

Primary users of this solution are:
- Product teams
- Recommendation system engineers
- Music analytics teams

These stakeholders can use predicted popularity scores to support ranking, playlist ordering, and discovery experiments.

### What Success Means

- Business KPI: Improve recommendation or playlist engagement by supporting better track ranking decisions, with a target uplift in listening time or completion rate.
- Technical KPI: Achieve RMSE and MAE that outperform a simple baseline and remain stable across validation and test splits.
- Operational KPI: Ensure predictions can be reproduced through a single entrypoint, tracked in W&B, served through a validated API contract, and deployed through controlled CI/CD workflows.
- Acceptance criteria: The pipeline must train, evaluate, register, and serve the model consistently with no manual notebook steps required in production.

### AI vs Non-AI Approach

Traditional ranking approaches rely heavily on historical engagement signals or manual curation. A machine learning approach can surface promising tracks earlier by learning from intrinsic audio properties, while still allowing product teams to combine predictions with editorial or behavioral signals.

### Risks and Mitigations

- Popularity bias toward already well-known tracks. Mitigation: monitor prediction distributions and compare against baseline ranking logic.
- Data drift as music trends evolve. Mitigation: retrain periodically and track performance over time in W&B.
- Overfitting to historical patterns. Mitigation: use validation splits, held-out test evaluation, and reproducible retraining.
- Misuse in ranking decisions. Mitigation: use predictions as one signal among several, not as the sole ranking mechanism.

## What This Repository Delivers

- Modular end-to-end ML pipeline in `src/`
- Single orchestration entrypoint via `python -m src.main`
- Centralized non-secret runtime settings in `config.yaml`
- Secret handling through `.env` and `.env.example`
- Dual-destination logging to console and local file
- W&B experiment tracking plus managed model-artifact inference
- FastAPI service with `/health` and `/predict`
- Dockerized serving setup with strict `.dockerignore`
- PR CI workflow and release-gated deploy workflow
- Tests for core modules and API contract behavior

## Repository Structure

```text
.
├── .github/workflows/
│   ├── ci.yml
│   ├── deploy.yml
│   └── retrain.yml
├── config.yaml
├── conda-lock.yml
├── Dockerfile
├── environment.yml
├── README.md
├── data/
├── logs/
├── models/
├── notebooks/
├── reports/
├── scripts/
│   └── call_api.py
├── src/
│   ├── api.py
│   ├── clean_data.py
│   ├── evaluate.py
│   ├── features.py
│   ├── infer.py
│   ├── load_data.py
│   ├── logger.py
│   ├── main.py
│   ├── train.py
│   ├── utils.py
│   └── validate.py
└── tests/
```

## Setup

### 1. Create the environment

```bash
conda env create -f environment.yml
conda activate spotify-archetypes-env
```

For a reproducible lock-based environment:

```bash
conda-lock install -n spotify-archetypes-env conda-lock.yml
```

### 2. Configure secrets

Copy `.env.example` into `.env` and fill in your W&B API key:

```dotenv
WANDB_API_KEY=...
```

Secrets must never be committed. `.env` is ignored by git and excluded from Docker builds.

Keep non-secret W&B settings such as `entity`, `project`, artifact name, and production alias in `config.yaml`.

### 3. Place the raw dataset

The default expected path is `data/raw/SpotifyAudioFeaturesApril2019.csv`.

## Running the Training Pipeline

```bash
python -m src.main
```

Outputs:
- Clean dataset in `data/processed/clean.csv`
- Model artifact in `models/model.joblib`
- Predictions in `reports/predictions.csv`
- Metrics in `reports/metrics.json`
- Run config snapshot in `reports/run_config.json`
- Local logs in `logs/pipeline.log`

When W&B is enabled, `src.main` also logs:
- Run metadata
- Validation and test metrics
- Model artifact
- Optional processed-data and prediction artifacts

## Model Registry and Production Inference

Training logs a managed W&B model artifact. Production inference is configured to load the artifact aliased `prod`, not an unmanaged local file. That behavior is controlled in `config.yaml`:

```yaml
wandb:
  entity: "your-wandb-entity"
  project: "spotify-sound-archetypes"
  model_artifact_name: "spotify-popularity-pipeline"
  production_alias: "prod"

inference:
  source: "wandb"
```

Recommended promotion flow:
1. Train and inspect the candidate artifact in W&B.
2. Promote the approved artifact to alias `prod`.
3. Deploy only after publishing a GitHub Release from `main`.

For offline local development, you can temporarily switch `inference.source` to `local`.

## API Usage

Run locally:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Prediction request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {
        "acousticness": 0.12,
        "danceability": 0.65,
        "duration_ms": 200000,
        "energy": 0.71,
        "instrumentalness": 0.0,
        "key": 5,
        "liveness": 0.11,
        "loudness": -5.1,
        "mode": 1,
        "speechiness": 0.05,
        "tempo": 120.5,
        "valence": 0.44
      }
    ]
  }'
```

You can also call the service from Python:

```bash
python scripts/call_api.py --url https://your-render-service.onrender.com/predict
```

## Docker and Deployment

Build the serving image:

```bash
docker build -t spotify-archetypes-api .
```

Run it:

```bash
docker run --rm -p 8000:8000 --env-file .env spotify-archetypes-api
```

Deployment discipline:
- `ci.yml` validates pull requests to `main`
- `deploy.yml` runs only when a GitHub Release is published
- The deploy workflow expects a `RENDER_DEPLOY_HOOK_URL` repository secret

## Testing

```bash
pytest --cov=src --cov-report=term-missing --cov-fail-under=80
```

## Monitoring and Observability

- Console logs for local and CI visibility
- Persistent local log file at `logs/pipeline.log`
- W&B run metrics and artifacts
- Render service logs after deployment

## Simple Model Card

### Intended use
Estimate expected Spotify track popularity from audio features for ranking support and exploratory prioritization.

### Inputs
Acousticness, danceability, duration, energy, instrumentalness, key, liveness, loudness, mode, speechiness, tempo, and valence.

### Output
A continuous popularity prediction.

### Risks
- Popularity reflects historical platform dynamics, not intrinsic quality.
- Trends drift over time, so retraining and monitoring are required.
- Predictions should support, not replace, editorial and ranking guardrails.

### Performance
Validation and test metrics are written to `reports/metrics.json` and tracked in W&B during pipeline runs.
