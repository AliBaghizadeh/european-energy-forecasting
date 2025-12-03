import os
import warnings

# Fix for LightGBM on Windows (Boost Compute caching issue)
os.environ["BOOST_COMPUTE_USE_OFFLINE_CACHE"] = "0"

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import optuna
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import joblib
import requests

# Monkey-patch requests to override Host header for MLflow
original_request = requests.Session.request


def patched_request(self, method, url, **kwargs):
    if "headers" not in kwargs:
        kwargs["headers"] = {}
    kwargs["headers"]["Host"] = "127.0.0.1:5001"
    return original_request(self, method, url, **kwargs)


requests.Session.request = patched_request

# Import data processing from train.py
from train import load_and_preprocess_data, engineer_features, DATA_PATH, target_col

# Set MLflow Experiment - Remote AWS Server
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://18.153.53.234:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")

# Load data once (shared across all optimizations)
print("Loading and preprocessing data...")
df_energy = engineer_features(load_and_preprocess_data(DATA_PATH))
X = df_energy.drop(columns=[target_col])
y = df_energy[target_col]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

print(f"Dataset: {len(X_scaled)} rows, {len(X_scaled.columns)} features")


# ============================================================================
# 1. XGBoost Optimization
# ============================================================================
def optimize_xgboost(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "objective": "reg:squarederror",
        "device": "cuda",
        "n_jobs": -1,
        "random_state": 42,
    }

    model = XGBRegressor(**params)

    # Cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    mae_scores = []

    for train_idx, val_idx in tscv.split(X_scaled):
        X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mae_scores.append(mean_absolute_error(y_val, preds))

    return np.mean(mae_scores)


# ============================================================================
# 2. LightGBM Optimization
# ============================================================================
def optimize_lightgbm(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "device": "cpu",  # Use CPU for stability on Windows
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1,
    }

    model = LGBMRegressor(**params)

    # Cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    mae_scores = []

    for train_idx, val_idx in tscv.split(X_scaled):
        X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mae_scores.append(mean_absolute_error(y_val, preds))

    return np.mean(mae_scores)


# ============================================================================
# 3. CatBoost Optimization
# ============================================================================
def optimize_catboost(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 200, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "depth": trial.suggest_int("depth", 3, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "task_type": "GPU",
        "random_state": 42,
        "verbose": 0,
        "allow_writing_files": False,
    }

    model = CatBoostRegressor(**params)

    # Cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    mae_scores = []

    for train_idx, val_idx in tscv.split(X_scaled):
        X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mae_scores.append(mean_absolute_error(y_val, preds))

    return np.mean(mae_scores)


# ============================================================================
# Helper function for MLflow logging
# ============================================================================
def log_model_to_mlflow(experiment_name, run_name, model, params, mae_score, input_example=None):
    """Log model to MLflow with detailed error reporting"""
    print(f"\n📤 Logging {run_name} to MLflow...")
    
    try:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name) as run:
            print(f"   Run ID: {run.info.run_id}")
            
            # Log parameters
            mlflow.log_params(params)
            print("   ✓ Parameters logged")
            
            # Log metrics
            mlflow.log_metric("mae", mae_score)
            mlflow.log_metric("rmse", np.sqrt(mae_score))
            print("   ✓ Metrics logged")
            
            # Log model
            # Fix: Use 'name' instead of 'artifact_path' (deprecated)
            # Fix: Add input_example to infer signature
            mlflow.sklearn.log_model(
                model, 
                name="model", 
                input_example=input_example
            )
            print("   ✓ Model logged")
            
            print(f"✅ Successfully logged {run_name}!")
            print(f"   View at: {mlflow.get_tracking_uri()}/#/experiments/{run.info.experiment_id}")
            return True
            
    except Exception as e:
        print(f"❌ Failed to log {run_name} to MLflow:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        import traceback
        print("\n   Full traceback:")
        traceback.print_exc()
        return False


# ============================================================================
# Main Execution
# ============================================================================
if __name__ == "__main__":
    N_TRIALS = 1  # Increase for more thorough search

    results = {}

    # ========== XGBoost ==========
    print("\n" + "=" * 60)
    print("🚀 Optimizing XGBoost...")
    print("=" * 60)
    mlflow.set_experiment("XGBoost_HPO")

    study_xgb = optuna.create_study(direction="minimize", study_name="xgboost_study")
    study_xgb.optimize(optimize_xgboost, n_trials=N_TRIALS, show_progress_bar=True)

    print(f"\n✅ Best XGBoost MAE: {study_xgb.best_value:.2f}")
    print(f"Best params: {study_xgb.best_params}")

    # Train final model with best params
    best_xgb_params = study_xgb.best_params
    best_xgb_params.update(
        {
            "objective": "reg:squarederror",
            "device": "cuda",
            "n_jobs": -1,
            "random_state": 42,
        }
    )
    final_xgb = XGBRegressor(**best_xgb_params)
    final_xgb.fit(X_scaled, y)

    results["xgboost"] = {
        "model": final_xgb,
        "mae": study_xgb.best_value,
        "params": study_xgb.best_params,
    }

    # Log to MLflow
    log_model_to_mlflow(
        "XGBoost_HPO", 
        "Best_XGBoost", 
        final_xgb, 
        best_xgb_params, 
        study_xgb.best_value,
        input_example=X_scaled.iloc[:5]
    )

    # ========== LightGBM ==========
    print("\n" + "=" * 60)
    print("🚀 Optimizing LightGBM...")
    print("=" * 60)
    mlflow.set_experiment("LightGBM_HPO")

    study_lgb = optuna.create_study(direction="minimize", study_name="lightgbm_study")
    study_lgb.optimize(optimize_lightgbm, n_trials=N_TRIALS, show_progress_bar=True)

    print(f"\n✅ Best LightGBM MAE: {study_lgb.best_value:.2f}")
    print(f"Best params: {study_lgb.best_params}")

    # Train final model with best params
    best_lgb_params = study_lgb.best_params
    best_lgb_params.update(
        {"device": "cpu", "n_jobs": -1, "random_state": 42, "verbose": -1}
    )
    final_lgb = LGBMRegressor(**best_lgb_params)
    final_lgb.fit(X_scaled, y)

    results["lightgbm"] = {
        "model": final_lgb,
        "mae": study_lgb.best_value,
        "params": study_lgb.best_params,
    }

    # Log to MLflow
    log_model_to_mlflow(
        "LightGBM_HPO", 
        "Best_LightGBM", 
        final_lgb, 
        best_lgb_params, 
        study_lgb.best_value,
        input_example=X_scaled.iloc[:5]
    )

    # ========== CatBoost ==========
    print("\n" + "=" * 60)
    print("🚀 Optimizing CatBoost...")
    print("=" * 60)
    mlflow.set_experiment("CatBoost_HPO")

    study_cat = optuna.create_study(direction="minimize", study_name="catboost_study")
    study_cat.optimize(optimize_catboost, n_trials=N_TRIALS, show_progress_bar=True)

    print(f"\n✅ Best CatBoost MAE: {study_cat.best_value:.2f}")
    print(f"Best params: {study_cat.best_params}")

    # Train final model with best params
    best_cat_params = study_cat.best_params
    best_cat_params.update(
        {
            "task_type": "GPU",
            "random_state": 42,
            "verbose": 0,
            "allow_writing_files": False,
        }
    )
    final_cat = CatBoostRegressor(**best_cat_params)
    final_cat.fit(X_scaled, y)

    results["catboost"] = {
        "model": final_cat,
        "mae": study_cat.best_value,
        "params": study_cat.best_params,
    }

    # Log to MLflow
    log_model_to_mlflow(
        "CatBoost_HPO", 
        "Best_CatBoost", 
        final_cat, 
        best_cat_params, 
        study_cat.best_value,
        input_example=X_scaled.iloc[:5]
    )

    # ========== Save All Models ==========
    print("\n" + "=" * 60)
    print("💾 Saving models...")
    print("=" * 60)

    MODEL_DIR = Path("Model")
    MODEL_DIR.mkdir(exist_ok=True)

    # Save models
    joblib.dump(final_xgb, MODEL_DIR / "xgboost_model.pkl")
    joblib.dump(final_lgb, MODEL_DIR / "lightgbm_model.pkl")
    joblib.dump(final_cat, MODEL_DIR / "catboost_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    # Save best parameters
    with open(MODEL_DIR / "best_params.txt", "w") as f:
        f.write("=" * 60 + "\n")
        f.write("BEST HYPERPARAMETERS FOR EACH MODEL\n")
        f.write("=" * 60 + "\n\n")

        for model_name, data in results.items():
            f.write(f"\n{model_name.upper()}:\n")
            f.write(f"  MAE: {data['mae']:.2f}\n")
            f.write(f"  Params: {data['params']}\n")
            f.write("-" * 60 + "\n")

    print(f"✅ Saved 3 models to {MODEL_DIR}/")
    print(f"   - xgboost_model.pkl")
    print(f"   - lightgbm_model.pkl")
    print(f"   - catboost_model.pkl")
    print(f"   - scaler.pkl")
    print(f"   - best_params.txt")

    # ========== Summary ==========
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS SUMMARY")
    print("=" * 60)
    for model_name, data in results.items():
        print(f"{model_name.upper():15} | MAE: {data['mae']:7.2f}")
    print("=" * 60)

    print(f"\n✅ Optimization Complete!")
    print(f"To view detailed results: mlflow ui --backend-store-uri file:./mlruns")
