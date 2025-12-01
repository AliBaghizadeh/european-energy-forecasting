import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from train import load_and_preprocess_data, engineer_features, DATA_PATH, target_col

print("Loading and preprocessing data...")
df_energy = engineer_features(load_and_preprocess_data(DATA_PATH))
X = df_energy.drop(columns=[target_col])

print(f"Feature count: {X.shape[1]}")
print(f"Features: {X.columns.tolist()}")

print("Fitting scaler...")
scaler = StandardScaler()
scaler.fit(X)

print("Saving scaler...")
MODEL_DIR = Path("Model")
MODEL_DIR.mkdir(exist_ok=True)
joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

print("✅ Scaler regenerated successfully!")
