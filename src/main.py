"""
Module: Main Pipeline
---------------------
Role: Orchestrate the entire flow (Load -> Clean -> Validate -> Train -> Evaluate).
Usage: python src/main.py
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

import src.load_data as load_data
import src.clean_data as clean_data
import src.validate as validate
import src.features as features
import src.train as train
import src.evaluate as evaluate
import src.infer as infer
from src.utils import save_csv, save_json, save_model

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Load the external configuration
SETTINGS = load_config("config.yaml")

def main():
    print(f"--- Starting MLOps Pipeline: {SETTINGS['project']['name']} ---")
    
    # 1. Infrastructure Setup
    for folder in ["data/raw", "data/processed", "models", "reports"]:
        Path(folder).mkdir(parents=True, exist_ok=True)

    # 2. Load - Accessing via ['paths']
    # Note: We wrap the string in Path() to ensure compatibility with your modules
    df_raw = load_data.load_raw_data(Path(SETTINGS["paths"]["raw_data"]))
    
    # 3. Clean
    df_clean = clean_data.clean_dataframe(df_raw, SETTINGS["project"]["target_column"])
    save_csv(df_clean, Path(SETTINGS["paths"]["clean_data"]))
    
    # 4. Validate
    required_cols = (
        SETTINGS["features"]["quantile_bin"] + 
        SETTINGS["features"]["categorical_onehot"] + 
        SETTINGS["features"]["numeric_passthrough"] + 
        [SETTINGS["project"]["target_column"]]
    )
    validate.validate_dataframe(
        df_clean,
        required_cols,
        target_column=SETTINGS["project"]["target_column"],
        allow_feature_nulls=True
    )
    
    # 5. Split
    target = SETTINGS["project"]["target_column"]
    X = df_clean.drop(columns=[target])
    y = df_clean[target]
    
    # Accessing test_size and seed from the ['train'] block
    strat = y if SETTINGS["project"]["problem_type"] == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=SETTINGS["train"]["test_size"], 
        random_state=SETTINGS["train"]["seed"], 
        stratify=strat
    )

    # 6. Feature Engineering
    # Accessing ['features'] block
    preprocessor = features.get_feature_preprocessor(
        quantile_bin_cols=SETTINGS["features"]["quantile_bin"],
        categorical_onehot_cols=SETTINGS["features"]["categorical_onehot"],
        numeric_passthrough_cols=SETTINGS["features"]["numeric_passthrough"],
        n_bins=SETTINGS["features"]["n_bins"]
    )
    
    # 7. Train
    model_pipeline = train.train_model(
        X_train,
        y_train,
        preprocessor,
        SETTINGS["project"]["problem_type"],
        train_config=SETTINGS.get("train", {}),
    )
    save_model(model_pipeline, Path(SETTINGS["paths"]["model_path"]))
    
    # 8. Evaluate
    metrics = evaluate.evaluate_model(
        model_pipeline, X_test, y_test, SETTINGS["project"]["problem_type"]
    )
    save_json(metrics, Path("reports/metrics.json"))
    save_json(SETTINGS, Path("reports/run_config.json"))
    
    # 9. Inference
    df_preds = infer.run_inference(model_pipeline, X_test)
    save_csv(df_preds, Path(SETTINGS["paths"]["report_path"]))
    
    print("--- Pipeline Completed Successfully ---")

if __name__ == "__main__":
    main()