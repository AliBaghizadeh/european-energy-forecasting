import mlflow
import os
from mlflow.tracking._tracking_service.client import TrackingServiceClient
from mlflow.utils.rest_utils import http_request
import requests

# Test connection to remote MLflow server
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://18.153.53.234:5000")

# Monkey-patch the requests to override Host header
original_request = requests.Session.request

def patched_request(self, method, url, **kwargs):
    # Override Host header to prevent DNS rebinding error
    if 'headers' not in kwargs:
        kwargs['headers'] = {}
    kwargs['headers']['Host'] = '127.0.0.1:5001'
    return original_request(self, method, url, **kwargs)

requests.Session.request = patched_request

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print(f"Testing connection to: {MLFLOW_TRACKING_URI}")

try:
    # Try to create a test experiment
    experiment_name = "connection_test"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"✅ Created test experiment: {experiment_id}")
    else:
        print(f"✅ Found existing experiment: {experiment.experiment_id}")
    
    # Try to log a test run
    with mlflow.start_run(experiment_id=experiment.experiment_id if experiment else experiment_id):
        mlflow.log_param("test_param", "hello")
        mlflow.log_metric("test_metric", 42)
        print("✅ Successfully logged test run!")
    
    print(f"\n🎉 Connection successful!")
    print(f"View your experiments at: {MLFLOW_TRACKING_URI}")
    
except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("\nTroubleshooting:")
    print("1. Check if MLflow server is running on EC2")
    print("2. Verify Security Group allows port 5000")
    print("3. Check if EC2 public IP is correct")
