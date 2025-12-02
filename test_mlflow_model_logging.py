import mlflow
import mlflow.sklearn
import os
import requests
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Apply the same Host header patch
original_request = requests.Session.request

def patched_request(self, method, url, **kwargs):
    if 'headers' not in kwargs:
        kwargs['headers'] = {}
    kwargs['headers']['Host'] = '127.0.0.1:5001'
    return original_request(self, method, url, **kwargs)

requests.Session.request = patched_request

# Set MLflow tracking URI
MLFLOW_TRACKING_URI = "http://18.153.53.234:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print(f"Testing MLflow model logging to: {mlflow.get_tracking_uri()}")

try:
    # Create a dummy model
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Try to log it
    mlflow.set_experiment("Test_Logging")
    with mlflow.start_run(run_name="Test_Model"):
        mlflow.log_param("test_param", "test_value")
        mlflow.log_metric("test_mae", 42.0)
        mlflow.sklearn.log_model(model, "model")
        print("✅ Successfully logged test model to MLflow!")
        print(f"🔗 Check: http://18.153.53.234:5000/#/experiments")
        
except Exception as e:
    print(f"❌ Failed to log model: {e}")
    import traceback
    traceback.print_exc()
