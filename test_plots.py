import pandas as pd
from datetime import datetime
from pathlib import Path

# Create test log file
LOG_FILE = Path("logs/inference_log.csv")
LOG_FILE.parent.mkdir(exist_ok=True)

# Create sample data
data = {
    "timestamp": [datetime.now().isoformat() for _ in range(5)],
    "last_load": [5000, 5100, 5200, 5300, 5400],
    "current_temp": [15.0, 16.0, 17.0, 18.0, 19.0],
    "country_id": ["DE", "FR", "DE", "IT", "FR"],
    "pred_xgboost": [4500, 4600, 4700, 4800, 4900],
    "pred_lightgbm": [4510, 4610, 4710, 4810, 4910],
    "pred_catboost": [4520, 4620, 4720, 4820, 4920]
}

df = pd.DataFrame(data)
df.to_csv(LOG_FILE, index=False)

print(f"✅ Created test log file with {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())
