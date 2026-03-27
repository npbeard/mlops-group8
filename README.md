# Spotify Popularity Prediction Pipeline

**Author:** Group 8  
**Course:** MLOps: Master in Business Analytics and Data Science  
**Status:** Final project repo

-----

## Business Case

Music platforms always have to decide which songs should get more visibility in playlists, recommendation sections, and discovery pages. If they only react to past engagement, they often notice promising tracks too late and keep favoring songs that are already doing well. In this project, we try to predict Spotify track popularity from audio features so that those decisions can be supported earlier and in a more consistent way.

For us, the value of the project is not only the model itself. A big part of the assignment was showing how that model could be handled in a more realistic MLOps workflow, with testing, tracking, deployment, and reproducibility.

- Faster experimentation on ranking ideas
- More reproducible retraining and evaluation
- Safer deployment through CI and release discipline
- Better traceability through logs, W&B artifacts, and a live API

### Objective

Our objective is to build a supervised machine learning model that predicts Spotify track popularity on a 0 to 100 scale using measurable audio features such as energy, danceability, valence, tempo, acousticness, and loudness. We also wanted to move the work beyond notebook experimentation and package it as a reproducible ML project.

### Users

The main users we imagine for this solution are:
- Product teams
- Recommendation system engineers
- Music analytics teams

These stakeholders could use predicted popularity scores to support ranking decisions, playlist ordering, and discovery experiments.

### What Success Means

- Business KPI: improve recommendation or playlist engagement by supporting better ranking decisions, ideally increasing listening time or completion rate.
- Technical KPI: achieve RMSE and MAE that beat a simple baseline and stay stable across validation and test splits.
- Operational KPI: make sure predictions can be reproduced through one entrypoint, tracked in W&B, served through a validated API contract, and deployed through controlled CI/CD workflows.
- Acceptance criteria: the pipeline should train, evaluate, register, and serve the model without depending on manual notebook steps for production use.

### AI vs Non-AI Approach

A more traditional approach would rely mostly on historical engagement signals or manual curation. A machine learning approach can help surface promising tracks earlier by learning from the audio properties of the song itself, while still being combined with business rules, editorial decisions, or behavioral data.

### Risks and Mitigations

- Popularity bias toward already well-known tracks. Mitigation: monitor prediction distributions and compare them with simpler ranking logic.
- Data drift as music trends evolve. Mitigation: retrain periodically and track performance over time in W&B.
- Overfitting to historical patterns. Mitigation: use validation splits, held-out test evaluation, and reproducible retraining.
- Misuse in ranking decisions. Mitigation: treat predictions as one signal among several, not as the only ranking mechanism.

## What This Repository Includes

- Modular end-to-end ML pipeline in `src/`
- Single orchestration entrypoint via `python -m src.main`
- Centralized non-secret runtime settings in `config.yaml`
- Secret handling through `.env` and `.env.example`
- Logging to both console and local file
- W&B experiment tracking plus managed model-artifact inference
- FastAPI service with `/health` and `/predict`
- Dockerized serving setup with a strict `.dockerignore`
- PR CI workflow and release-gated deploy workflow
- Tests for core modules and API behavior

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
│   ├── call_api.py
│   ├── promote_model.py
│   └── verify_deployment.py
├── src/
│   ├── api.py
│   ├── clean_data.py
│   ├── deployment_verifier.py
│   ├── evaluate.py
│   ├── features.py
│   ├── infer.py
│   ├── load_data.py
│   ├── logger.py
│   ├── main.py
│   ├── model_registry.py
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

Secrets should never be committed. `.env` is ignored by git and excluded from Docker builds.

Non-secret W&B settings such as `entity`, `project`, artifact name, and production alias stay in `config.yaml`.

### 3. Place the raw dataset

The default expected path is `data/raw/SpotifyAudioFeaturesApril2019.csv`.

## Running The Training Pipeline

```bash
python -m src.main
```

For reproducible tracked experiments with alternate configs:

```bash
python -m src.main --config configs/experiments/rf_candidate.yaml
python -m src.main --config configs/experiments/ridge_candidate.yaml
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

## Simple Model Selection Experiment

To satisfy the final-assignment model-registry requirement, we ran two simple tracked candidates in W&B and chose the best one using validation RMSE.

Compared runs:
- `final-rf-candidate`: validation RMSE `18.2369`, validation MAE `14.9154`, test RMSE `18.1460`, test MAE `14.8233`
- `final-ridge-candidate`: validation RMSE `18.8577`, validation MAE `15.5427`, test RMSE `18.7476`, test MAE `15.4518`

Decision rule:
- Lower validation RMSE wins
- Winner: `final-rf-candidate`

W&B run links:
- RF candidate: `https://wandb.ai/nicopbeard-ie-university/spotify-sound-archetypes/runs/y5fhbs7z`
- Ridge candidate: `https://wandb.ai/nicopbeard-ie-university/spotify-sound-archetypes/runs/z96ptosr`

Promotion result:
- Promoted artifact family: `spotify-popularity-pipeline`
- Promoted artifact version: `v4`
- Production alias attached: `prod`

Promotion command used:

```bash
python scripts/promote_model.py --source latest --target prod
```

## Model Registry And Production Inference

Training logs a managed W&B model artifact. Production inference is configured to load the artifact aliased `prod`, instead of depending on an unmanaged local file. That behavior is controlled in `config.yaml`:

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

Promote the approved candidate with:

```bash
python scripts/promote_model.py --source latest --target prod
```

The script prints a short JSON summary with the source reference, promoted reference, artifact version, and aliases applied to that artifact. It uses `WANDB_API_KEY` from the current environment or from the project `.env` file.

For offline local development, `inference.source` can be temporarily switched to `local`.

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
python scripts/call_api.py --url https://mlops-group8-1.onrender.com/predict
```

## Docker And Deployment

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

## Deployment Verification

Current live service URL:

```text
https://mlops-group8-1.onrender.com
```

After Render finishes deploying, you can verify that the live service is using the promoted W&B artifact alias instead of a local model file:

```bash
python scripts/verify_deployment.py \
  --base-url https://mlops-group8-1.onrender.com \
  --expect-source wandb \
  --expect-alias prod
```

The verification script checks:
- `/health` returns `200`, reports `status=ok`, and confirms the model is loaded
- `/predict` returns `200` with at least one prediction
- `/predict` reports `model_source=wandb`
- `/predict` reports a `model_reference` ending in `:prod`

If you want to validate a custom JSON request body, you can pass `--payload-file path/to/request.json`.

## Testing

```bash
pytest --cov=src --cov-report=term-missing --cov-fail-under=100
```

## Monitoring And Observability

- Console logs for local and CI visibility
- Persistent local log file at `logs/pipeline.log`
- W&B run metrics and artifacts
- Render service logs after deployment

## Simple Model Card

### Intended use
Estimate Spotify track popularity from audio features for ranking support and exploratory prioritization.

### Inputs
Acousticness, danceability, duration, energy, instrumentalness, key, liveness, loudness, mode, speechiness, tempo, and valence.

### Output
A continuous popularity prediction score.

### Risks
- Popularity reflects historical platform dynamics, not intrinsic quality.
- Trends drift over time, so retraining and monitoring are necessary.
- Predictions should support, not replace, editorial and ranking guardrails.

### Performance
Validation and test metrics are written to `reports/metrics.json` and tracked in W&B during pipeline runs.
