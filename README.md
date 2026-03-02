# Spotify Popularity Prediction Pipeline

**Author:** Group 8  
**Course:** MLOps: Master in Business Analytics and Data Science  
**Status:** Session 1 (Initialization)

---

## 1. Business Objective

* **The Goal:**  
  Develop a supervised machine learning model to predict Spotify track popularity (0–100) using measurable audio features such as energy, danceability, valence, tempo, acousticness, and loudness.

  The objective is to identify which audio characteristics are most strongly associated with higher popularity and to build a predictive system that supports data-driven ranking and promotion decisions.

* **The User:**  
  The primary users of this model are music streaming product teams, recommendation system engineers, and music analytics teams.

  They would use predicted popularity scores to prioritize tracks within recommendation systems, optimize playlist ordering, and better understand the relationship between audio features and user engagement.

---

## 2. Success Metrics

* **Business KPI (The "Why"):**  
  Improve recommendation and playlist engagement metrics by enabling more accurate ranking of tracks, with a target uplift of 3–5% in average listening time or track completion rate.

* **Technical Metric (The "How"):**  
  Achieve an RMSE that outperforms a baseline model (e.g., mean predictor or simple linear regression) and maintain stable MAE performance across validation splits.

* **Acceptance Criteria:**  
  The model must outperform a defined baseline on RMSE and MAE, demonstrate stable performance across validation data, and produce reproducible predictions through the end-to-end pipeline executed via `src.main`.

---

## 3. The Data

* **Source:** SpotifyAudioFeaturesApril2019 (Kaggle)

* **Target Variable:**  
  `popularity` (numeric score from 0 to 100), representing the relative popularity of a track on Spotify.

* **Sensitive Info:**  
  The dataset does not contain personally identifiable information (PII). It consists solely of track-level audio features and metadata.

---

## 4. Repository Structure

This project follows a strict separation between "Sandbox" (Notebooks) and "Production" (Src).

```text
.
├── README.md                # Project definition
├── environment.yml          # Dependencies (Conda/Pip)
├── config.yaml              # Global configuration (paths, params)
├── .env                     # Secrets placeholder
│
├── notebooks/               # Experimental sandbox
│   └── Final_Assignment.ipynb
│
├── src/                     # Production code (The "Factory")
│   ├── __init__.py
│   ├── load_data.py
│   ├── clean_data.py
│   ├── features.py
│   ├── validate.py
│   ├── train.py
│   ├── evaluate.py
│   ├── infer.py
│   └── main.py
│
├── data/                    # Local storage (IGNORED by Git)
│   ├── raw/
│   └── processed/
│
├── models/                  # Serialized model artifacts (IGNORED by Git)
│
├── reports/                 # Generated metrics, predictions, and configs
│
└── tests/                   # Automated test suite
```

## 5. Execution Model

The full machine learning pipeline will eventually be executable through:

`python -m src.main`



