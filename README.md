# Spotify Popularity Prediction Pipeline #

**Author:** Group 8  
**Course:** MLOps: Master in Business Analytics and Data Science  
**Status:** On Development

-----

## Business Case

# 1. Client & Industry
The hypothetical client is a music streaming platform (e.g., Spotify) operating in the digital music streaming industry. The platform manages millions of tracks and must prioritize which songs appear in playlists, recommendations, and discovery feeds.

# 2. Problem Statement
Music platforms must continuously decide which tracks to promote or prioritize. Current ranking often relies heavily on historical engagement metrics or manual curation, which can delay discovery of emerging hits.

# 3. Objective
Develop a supervised machine learning model to predict Spotify track popularity (0–100) using measurable audio features such as energy, danceability, valence, tempo, acousticness, and loudness. The objective is to identify which audio characteristics are most strongly associated with higher popularity and to build a predictive system that supports data-driven ranking and promotion decisions.

# 4. Users
Primary users:
- Product teams
- Recommendation system engineers
- Music analytics teams
They use predicted popularity scores to improve playlist ordering and recommendation quality.

# 5. Success KPI
**Business KPI (The "Why"):**  
  Improve recommendation and playlist engagement metrics by enabling more accurate ranking of tracks, with a target uplift of 3–5% in average listening time or track completion rate.

**Technical Metric (The "How"):**  
  Achieve an RMSE that outperforms a baseline model (e.g., mean predictor or simple linear regression) and maintain stable MAE performance across validation splits.

**Acceptance Criteria:**  
  The model must outperform a defined baseline on RMSE and MAE, demonstrate stable performance across validation data, and produce reproducible predictions through the end-to-end pipeline executed via `src.main`.

# 6. AI vs Non-AI Approach
Traditional approaches rely on historical engagement or manual curation. A machine learning model can identify promising tracks earlier using intrinsic audio properties.

# 7. Estimated Costs
A typical implementation could involve:
- **Team:** 1 Data Scientist and 1 ML Engineer
- **Timeline:** Approximately 4–6 weeks for initial development and deployment
- **Infrastructure:** Cloud compute for training and batch inference, with costs depending on catalog size and retraining frequency.

# 8. Risks & Mitigations
- Popularity bias toward already well-known tracks | Use diverse training data and monitor prediction distributions |
- Data drift as music trends evolve | Implement periodic retraining and performance monitoring |
- Overfitting to historical patterns | Apply validation splits, baselines, and regularization |
- Misuse in recommendation ranking | Combine predictions with other engagement signals rather than relying solely on model output |

# 9. The Data
- Source: SpotifyAudioFeaturesApril2019 (Kaggle)
- Target Variable:  `popularity` (numeric score from 0 to 100), representing the relative popularity of a track on Spotify.
- Sensitive Info: The dataset does not contain personally identifiable information (PII). It consists solely of track-level audio features and metadata.

# 10. Repository Structure
This project follows a strict separation between "Sandbox" (Notebooks) and "Production" (Src).

```text
.
├── LICENSE
├── README.md                # Project definition
├── config.yaml              # Global configuration (paths, params)
├── environment.yml          # Dependencies (Conda/Pip)
├── mlops.log                # Pipeline / run logs
├── pytest.ini               # Pytest configuration
│
├── data/                    # Local data storage
│   ├── inference/           # Inputs/outputs for inference runs
│   ├── processed/           # Clean/processed datasets
│   │   └── clean.csv
│   └── raw/                 # Original source data
│       └── SpotifyAudioFeaturesApril2019.csv
│
├── models/                  # Serialized model artifacts
│   └── model.joblib
│
├── notebooks/               # Experimental sandbox
│   ├── Final_Assignment.ipynb
│   └── sandbox_pipeline_step_by_step.ipynb
│
├── reports/                 # Generated metrics, predictions, and configs
│   ├── metrics.json
│   ├── predictions.csv
│   └── run_config.json
│
├── src/                     # Production code (The "Factory")
│   ├── __init__.py
│   ├── clean_data.py
│   ├── evaluate.py
│   ├── features.py
│   ├── infer.py
│   ├── load_data.py
│   ├── main.py
│   ├── train.py
│   ├── utils.py
│   └── validate.py
│
└── tests/                   # Automated test suite
    ├── test_clean_data.py
    ├── test_evaluate.py
    ├── test_features.py
    ├── test_infer.py
    ├── test_load_data.py
    ├── test_main.py
    ├── test_train.py
    ├── test_utils.py
    └── test_validate.py
```

## 11. Execution Model

The full machine learning pipeline will eventually be executable through:

`python -m src.main`

To run tests/coverage:

`pytest --cov=src --cov-report=term-missing`
