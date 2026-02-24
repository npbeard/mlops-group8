# Spotify Track Popularity Prediction Pipeline

**Author:** Group 8  
**Course:** MLOps: Master in Business Analytics and Data Science  
**Status:** Session 1 (Initialization)

---

## 1. Business Objective

* **The Goal:**  
  Predict the popularity score of Spotify tracks using audio features to better understand the key characteristics of successful songs. This model can help music producers, record labels, and streaming platforms identify high-potential tracks and optimize production and promotion strategies.

* **The User:**  
  Music analysts, producers, and platform analysts consume the output as a structured predictions file (`reports/predictions.csv`) that estimates expected popularity based on track audio features. This enables data-driven decision-making for music production, recommendation systems, and content strategy.

---

## 2. Success Metrics

* **Business KPI (The "Why"):**  
  Enable data-driven identification of high-potential songs, improving decision-making efficiency for music production and promotion. Success is measured by the model’s ability to reliably differentiate higher- vs lower-popularity tracks based on audio features.

* **Technical Metric (The "How"):**  
  Root Mean Squared Error (RMSE) on the test set. Current baseline performance:

  - RMSE ≈ 18.1  
  - MAE ≈ 14.8  

* **Acceptance Criteria:**  
  - The pipeline runs end-to-end via `python -m src.main` without errors  
  - The model produces reproducible predictions  
  - The pipeline is leakage-safe (all preprocessing occurs inside the sklearn Pipeline and is fit on training data only)  
  - Required artifacts are generated:
    - `data/processed/clean.csv`
    - `models/model.joblib`
    - `reports/predictions.csv`
    - `reports/metrics.json`

---

## 3. The Data

* **Source:**  
  Spotify Audio Features dataset (`SpotifyAudioFeaturesApril2019.csv`), provided as a CSV file.

* **Target Variable:**  
  `popularity` — a numeric score (0–100) representing the popularity of a track on Spotify.

* **Features Used:**  
  Audio characteristics such as:

  - danceability  
  - energy  
  - loudness  
  - tempo  
  - speechiness  
  - acousticness  
  - instrumentalness  
  - valence  
  - duration_ms  
  - key, mode, and other audio descriptors  

* **Sensitive Info:**  
  This dataset contains **no personally identifiable information (PII)**. It consists only of track metadata and audio features.

  ⚠️ The `data/`, `models/`, and generated artifacts are excluded from version control via `.gitignore` to follow best practices.

---

## 4. Repository Structure

This project follows a strict separation between "Sandbox" (Notebooks) and "Production" (Src).

```text
.
├── README.md
├── environment.yml
├── config.yaml
├── .env
│
├── notebooks/
│   └── Final_Assignment.ipynb
│
├── src/
│   ├── __init__.py
│   ├── load_data.py
│   ├── clean_data.py
│   ├── validate.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── infer.py
│   ├── utils.py
│   └── main.py
│
├── data/
│   ├── raw/
│   │   └── SpotifyAudioFeaturesApril2019.csv
│   ├── processed/
│   │   └── clean.csv
│   └── inference/
│
├── models/
│   └── model.joblib
│
├── reports/
│   ├── predictions.csv
│   ├── metrics.json
│   └── run_config.json
│
└── tests/

## 5. Execution Model

The full machine learning pipeline will eventually be executable through:

`python src/main.py`



