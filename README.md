# Spotify Sound Archetype Discovery

**Author:** Group 8  
**Course:** MLOps: Master in Business Analytics and Data Science  
**Status:** Session 1 (Initialization)

---

## 1. Business Objective

* **The Goal:**  
  Identify distinct clusters of songs based purely on audio features (e.g., energy, danceability, valence, tempo) and evaluate whether these clusters reveal interpretable musical archetypes and differences in popularity.  

  Discover data-driven sound segments that could support improved music recommendation systems, playlist curation strategies, and audience targeting decisions.

* **The User:**  
  The primary users of this analysis are music streaming product teams, data scientists working on recommendation systems, and music analytics teams.  

  They would use cluster assignments and visualizations to segment tracks based on measurable sound characteristics independent of genre labels, and evaluate how different sound profiles relate to popularity and engagement.

---

## 2. Success Metrics

* **Business KPI (The "Why"):**  
  Increase average playlist listening time by 3–5% by identifying and promoting high-performing sound archetypes within recommendation strategies.

* **Technical Metric (The "How"):**  
  Achieve a Silhouette Score ≥ 0.25 to ensure meaningful cluster separation, and retain ≥ 60% cumulative explained variance through PCA to preserve the majority of information in reduced feature space.

* **Acceptance Criteria:**  
  The clustering solution must achieve a Silhouette Score ≥ 0.25, retain ≥ 60% cumulative explained variance through PCA, and demonstrate statistically significant differences in average popularity across at least two identified sound archetypes.

---

## 3. The Data

* **Source:** SpotifyAudioFeaturesApril2019 (Kaggle)  
* **Target Variable:** This is an unsupervised learning task; therefore, no prediction target is defined. The popularity variable (0–100) is used post-clustering to evaluate differences in commercial performance across identified sound archetypes.  
* **Sensitive Info:** The dataset does not contain personally identifiable information (PII) such as names, emails, or payment details.

---

## 4. Repository Structure

This project follows a strict separation between "Sandbox" (Notebooks) and "Production" (Src).

```text
.
├── README.md                # This file (Project definition)
├── environment.yml          # Dependencies (Conda/Pip)
├── config.yaml              # Global configuration (paths, params)
├── .env                     # Secrets placeholder
│
├── notebooks/               # Experimental sandbox
│   └── yourbaseline.ipynb   # From previous work
│
├── src/                     # Production code (The "Factory")
│   ├── __init__.py          # Python package
│   ├── load_data.py         # Ingest raw data
│   ├── clean_data.py        # Preprocessing & cleaning
│   ├── features.py          # Feature engineering
│   ├── validate.py          # Data quality checks
│   ├── train.py             # Model training & saving
│   ├── evaluate.py          # Metrics & plotting
│   ├── infer.py             # Inference logic
│   └── main.py              # Pipeline orchestrator
│
├── data/                    # Local storage (IGNORED by Git)
│   ├── raw/                 # Immutable input data
│   └── processed/           # Cleaned data ready for training
│
├── models/                  # Serialized artifacts (IGNORED by Git)
│
├── reports/                 # Generated metrics, plots, and figures
│
└── tests/                   # Automated tests
```

## 5. Execution Model

The full machine learning pipeline will eventually be executable through:

`python -m src.main`



