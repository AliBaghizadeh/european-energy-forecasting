import gradio as gr
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import os
import csv

# Model paths
MODEL_DIR = Path("./Model")
MODELS = {
    "XGBoost": MODEL_DIR / "xgboost_model.pkl",
    "LightGBM": MODEL_DIR / "lightgbm_model.pkl",
    "CatBoost": MODEL_DIR / "catboost_model.pkl",
}
SCALER_PATH = MODEL_DIR / "scaler.pkl"

# Logging setup
LOG_FILE = Path("logs/inference_log.csv")
LOG_FILE.parent.mkdir(exist_ok=True)

if not LOG_FILE.exists():
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "last_load",
                "current_temp",
                "country_id",
                "pred_xgboost",
                "pred_lightgbm",
                "pred_catboost",
            ]
        )

# Model performance (from HPO)
MODEL_PERFORMANCE = {"XGBoost": 92.81, "LightGBM": 93.42, "CatBoost": 124.74}

# Load models and scaler
print("Loading models...")
loaded_models = {}
for name, path in MODELS.items():
    try:
        loaded_models[name] = joblib.load(path)
        print(f"✅ {name} loaded")
    except Exception as e:
        print(f"❌ Failed to load {name}: {e}")
        loaded_models[name] = None

try:
    scaler = joblib.load(SCALER_PATH)
    print("✅ Scaler loaded")
except Exception as e:
    print(f"❌ Failed to load scaler: {e}")
    scaler = None


# Feature engineering function
def engineer_time_features():
    """Calculates time features for the next hour (t+1)."""
    target_time = datetime.now() + timedelta(hours=1)

    hour = target_time.hour
    day_of_week = target_time.weekday()
    month = target_time.month
    is_weekend = 1 if day_of_week >= 5 else 0

    return [hour, day_of_week, month, is_weekend]


# Prediction function
def predict_all_models(last_load: float, current_temp: float, country_id: str):
    print(f"Received request: Load={last_load}, Temp={current_temp}, Country={country_id}")

    if scaler is None:
        return (
            "❌ Error: Scaler not loaded",
            "❌ Error: Scaler not loaded",
            "❌ Error: Scaler not loaded",
        )

    # Encode Country ID
    id_map = {
        "AT": 0,
        "DE": 1,
        "FR": 2,
        "IT": 3,
        "BE": 4,
        "CH": 5,
        "NL": 6,
        "PL": 7,
        "CZ": 8,
        "ES": 9,
    }
    id_encoded = id_map.get(country_id, 0)

    # Calculate time features
    time_features = engineer_time_features()

    # Create feature array
    num_features = 21  # Correct feature count from training
    input_features = np.zeros((1, num_features))

    # Assign features (ensure indices match training data structure)
    input_features[0, 2] = current_temp  # Index 2: temperature
    input_features[0, 9] = time_features[3]  # Index 9: is_weekend

    # Index 10-15: Cyclic features
    hour = time_features[0]
    day_of_week = time_features[1]
    month = time_features[2]

    input_features[0, 10] = np.sin(2 * np.pi * hour / 24)  # hour_sin
    input_features[0, 11] = np.cos(2 * np.pi * hour / 24)  # hour_cos
    input_features[0, 12] = np.sin(2 * np.pi * day_of_week / 7)  # dayofweek_sin
    input_features[0, 13] = np.cos(2 * np.pi * day_of_week / 7)  # dayofweek_cos
    input_features[0, 14] = np.sin(2 * np.pi * month / 12)  # month_sin
    input_features[0, 15] = np.cos(2 * np.pi * month / 12)  # month_cos

    input_features[0, 16] = last_load  # Index 16: target_lag_1 (Last known load)
    input_features[0, 20] = id_encoded  # Index 20: id_encoded

    # Scale features
    try:
        input_scaled = scaler.transform(input_features)
    except Exception as e:
        print(f"Scaling error: {e}")
        return (
            f"❌ Scaling error: {e}",
            f"❌ Scaling error: {e}",
            f"❌ Scaling error: {e}",
        )

    # Get predictions from all models
    results = {}
    raw_preds = {}

    for name, model in loaded_models.items():
        if model is None:
            results[name] = "❌ Model not loaded"
            raw_preds[name] = 0.0
        else:
            try:
                prediction = model.predict(input_scaled)[0]
                mae = MODEL_PERFORMANCE[name]
                results[name] = f"**{prediction:,.2f} MW**\n\n📊 Model MAE: {mae:.2f} MW"
                raw_preds[name] = float(prediction)
            except Exception as e:
                print(f"{name} error: {e}")
                results[name] = f"❌ Prediction error: {e}"
                raw_preds[name] = 0.0

    # Log request
    try:
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    datetime.now().isoformat(),
                    last_load,
                    current_temp,
                    country_id,
                    raw_preds.get("XGBoost", 0),
                    raw_preds.get("LightGBM", 0),
                    raw_preds.get("CatBoost", 0),
                ]
            )
    except Exception as e:
        print(f"Logging failed: {e}")

    return results["XGBoost"], results["LightGBM"], results["CatBoost"]


# Monitoring Dashboard Functions
def get_recent_logs():
    if not LOG_FILE.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(LOG_FILE)
        return df.tail(20).iloc[::-1]  # Show last 20, newest first
    except:
        return pd.DataFrame()


def plot_load_forecast():
    try:
        df = get_recent_logs()
        if df.empty or len(df) < 1:
            return None

        if "pred_xgboost" not in df.columns:
            return None

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp", "pred_xgboost"])

        if df.empty:
            return None

        return gr.LinePlot(
            df, x="timestamp", y="pred_xgboost", title="XGBoost Predictions Over Time"
        )
    except Exception as e:
        print(f"Error in plot_load_forecast: {e}")
        return None


def plot_temp_dist():
    try:
        df = get_recent_logs()
        if df.empty or len(df) < 1:
            return None

        if "current_temp" not in df.columns:
            return None

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp", "current_temp"])

        if df.empty:
            return None

        return gr.LinePlot(
            df, x="timestamp", y="current_temp", title="Temperature Input Trend"
        )
    except Exception as e:
        print(f"Error in plot_temp_dist: {e}")
        return None


# Gradio Interface
with gr.Blocks(title="⚡ Energy Load Forecast - Multi-Model Comparison") as demo:
    gr.Markdown(
        """
        # ⚡ Energy Load Forecast: Multi-Model Comparison
        
        This application forecasts short-term energy consumption (MW) for the **next hour (t+1)** using **3 optimized models**.
        """
    )

    with gr.Tabs():
        # Tab 1: Prediction Interface
        with gr.TabItem("🔮 Forecast"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Input Parameters")
                    last_load = gr.Slider(
                        0, 10000, step=1, label="Last Known Load (MW)", value=5000
                    )
                    current_temp = gr.Slider(
                        -20, 40, step=0.1, label="Current Temperature (°C)", value=15.0
                    )
                    country_id = gr.Radio(
                        ["AT", "DE", "FR", "IT", "BE", "CH", "NL", "PL", "CZ", "ES"],
                        label="Country ID",
                        value="DE",
                    )
                    predict_btn = gr.Button(
                        "🔮 Predict with All Models", variant="primary"
                    )

            gr.Markdown("### 📊 Model Predictions Comparison")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🥇 XGBoost (Best)")
                    xgb_output = gr.Markdown()

                with gr.Column():
                    gr.Markdown("#### 🥈 LightGBM")
                    lgb_output = gr.Markdown()

                with gr.Column():
                    gr.Markdown("#### 🥉 CatBoost")
                    cat_output = gr.Markdown()

            # Examples
            gr.Examples(
                examples=[
                    [5000, 15.0, "DE"],
                    [3000, 5.5, "FR"],
                    [7000, 30.0, "AT"],
                ],
                inputs=[last_load, current_temp, country_id],
            )

        # Tab 2: Monitoring Dashboard
        with gr.TabItem("📊 Monitoring"):
            gr.Markdown(
                """
                ### 📈 Live Model Monitoring
                
                These plots show the history of predictions made through this app:
                - **XGBoost Predictions**: Model forecasts over time
                - **Temperature Trend**: Input temperatures you've used
                
                *Make some predictions first to see the monitoring plots!*
                """
            )
            refresh_btn = gr.Button("🔄 Refresh Data")

            with gr.Row():
                load_plot = gr.LinePlot(label="XGBoost Predictions")
                temp_plot = gr.LinePlot(label="Temperature Inputs")

            gr.Markdown("### 📝 Recent Inference Logs")
            log_table = gr.DataFrame(
                headers=[
                    "timestamp",
                    "last_load",
                    "current_temp",
                    "country_id",
                    "pred_xgboost",
                ]
            )

            # Refresh logic
            refresh_btn.click(plot_load_forecast, outputs=load_plot)
            refresh_btn.click(plot_temp_dist, outputs=temp_plot)
            refresh_btn.click(get_recent_logs, outputs=log_table)

            # Auto-load on tab select
            demo.load(plot_load_forecast, outputs=load_plot)
            demo.load(plot_temp_dist, outputs=temp_plot)
            demo.load(get_recent_logs, outputs=log_table)

    gr.Markdown(
        """
        ---
        ### 🚀 Model Details
        
        **Feature Engineering:** The models use advanced time-series features including:
        - Cyclical encoding (hour, day of week, month)
        - Lagged features (15min, 1hr, 1day, 1week)
        - Weather data (temperature, radiation)
        - Holiday indicators
        
        **Deployment:** This application is deployed automatically via **GitHub Actions CI/CD** 
        to Hugging Face Spaces. Models are tracked and versioned using **MLflow Model Registry**.
        """
    )

    # Connect button to prediction function
    predict_btn.click(
        fn=predict_all_models,
        inputs=[last_load, current_temp, country_id],
        outputs=[xgb_output, lgb_output, cat_output],
    )

# Launch
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)