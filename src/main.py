"""
Module: Main Pipeline
---------------------
Role: Orchestrate the entire flow (Load -> Clean -> Validate -> Train -> Evaluate).
Usage: python src/main.py
"""

"""
Educational Goal:
- Why this module exists in an MLOps system: Orchestrates the entire pipeline.
- Responsibility (separation of concerns): Managing the sequence of execution.
- Pipeline contract (inputs and outputs): Reads SETTINGS; produces saved artifacts (data, model, reports).

TODO: Replace print statements with standard library logging in a later session
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

import src.load_data as load_data
import src.clean_data as clean_data
import src.validate as validate
import src.features as features
import src.train as train
import src.evaluate as evaluate
import src.infer as infer
from src.utils import save_csv, save_model

# ==========================================
# CONFIGURATION BLOCK
# ==========================================
# LOUD WARNING: This dictionary acts as a bridge to your config.yml.
# Update these lists to match your REAL dataset columns once you move past the dummy logic.
SETTINGS = {
    "is_example_config": False,
    "problem_type": "regression",
    "target_column": "popularity",
    "raw_data_path": Path("data/raw/SpotifyAudioFeaturesApril2019.csv"),
    "clean_data_path": Path("data/processed/clean.csv"),
    "model_path": Path("models/model.joblib"),
    "report_path": Path("reports/predictions.csv"),
    "features": {
        "quantile_bin": ["duration_ms", "tempo"], 
        "categorical_onehot": ["key", "mode"],
        "numeric_passthrough": ["acousticness", "danceability", "energy", "instrumentalness", "liveness", "loudness", "speechiness", "valence"],
        "n_bins": 5
    }
}

def main():
    print("--- Starting MLOps Pipeline ---")
    
    # 1. Infrastructure Setup
    for folder in ["data/raw", "data/processed", "models", "reports"]:
        Path(folder).mkdir(parents=True, exist_ok=True)

    # 2. Load
    df_raw = load_data.load_raw_data(SETTINGS["raw_data_path"])
    
    # 3. Clean
    df_clean = clean_data.clean_dataframe(df_raw, SETTINGS["target_column"])
    save_csv(df_clean, SETTINGS["clean_data_path"])
    
    # 4. Validate
    required_cols = (
        SETTINGS["features"]["quantile_bin"] + 
        SETTINGS["features"]["categorical_onehot"] + 
        SETTINGS["features"]["numeric_passthrough"] + 
        [SETTINGS["target_column"]]
    )
    validate.validate_dataframe(df_clean, required_cols)
    
    # 5. Split
    X = df_clean.drop(columns=[SETTINGS["target_column"]])
    y = df_clean[SETTINGS["target_column"]]
    
    strat = y if SETTINGS["problem_type"] == "classification" else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=strat
        )
    except:
        print("Warning: Stratification failed, proceeding without it.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # 6. Feature Engineering
    # Fail-fast: check that numeric cols are actually numeric
    for col in SETTINGS["features"]["quantile_bin"]:
        if not pd.api.types.is_numeric_dtype(df_clean[col]):
            raise TypeError(f"Column '{col}' must be numeric for quantile binning.")

    preprocessor = features.get_feature_preprocessor(
        quantile_bin_cols=SETTINGS["features"]["quantile_bin"],
        categorical_onehot_cols=SETTINGS["features"]["categorical_onehot"],
        numeric_passthrough_cols=SETTINGS["features"]["numeric_passthrough"],
        n_bins=SETTINGS["features"]["n_bins"]
    )
    
    # 7. Train
    model_pipeline = train.train_model(
        X_train, y_train, preprocessor, SETTINGS["problem_type"]
    )
    save_model(model_pipeline, SETTINGS["model_path"])
    
    # 8. Evaluate
    metric_score = evaluate.evaluate_model(
        model_pipeline, X_test, y_test, SETTINGS["problem_type"]
    )
    
    # 9. Inference (Simulated on the test set)
    df_preds = infer.run_inference(model_pipeline, X_test)
    save_csv(df_preds, SETTINGS["report_path"])
    
    print("--- Pipeline Completed Successfully ---")

if __name__ == "__main__":
    main()