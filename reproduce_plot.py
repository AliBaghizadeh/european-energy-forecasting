
import pandas as pd
import gradio as gr
from datetime import datetime

# Mock data creation similar to what's in the app
data = {
    "timestamp": [datetime.now().isoformat(), datetime.now().isoformat()],
    "last_load": [5000, 5200],
    "current_temp": [15.0, 16.0],
    "country_id": ["DE", "FR"],
    "pred_xgboost": [4500.0, 4600.0],
    "pred_lightgbm": [4510.0, 4610.0],
    "pred_catboost": [4520.0, 4620.0]
}
df = pd.DataFrame(data)

def plot_load_forecast(df):
    try:
        if df.empty or len(df) < 1:
            print("DF is empty or too small")
            return None
        
        if 'pred_xgboost' not in df.columns:
            print("Missing pred_xgboost column")
            return None
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp', 'pred_xgboost'])
        
        if df.empty:
            print("DF empty after dropna")
            return None
        
        # Only add country grouping if country_id exists
        if 'country_id' in df.columns:
            country_names = {
                'AT': 'Austria', 'DE': 'Germany', 'FR': 'France', 'IT': 'Italy',
                'BE': 'Belgium', 'CH': 'Switzerland', 'NL': 'Netherlands',
                'PL': 'Poland', 'CZ': 'Czech Republic', 'ES': 'Spain'
            }
            df['Country'] = df['country_id'].map(country_names).fillna(df['country_id'])
            
            print("Generating plot with color='Country'")
            plot = gr.LinePlot(
                df,
                x="timestamp",
                y="pred_xgboost",
                color="Country",
                title="XGBoost Predictions by Country"
            )
            return plot
        else:
            print("Generating plot without grouping")
            return gr.LinePlot(
                df,
                x="timestamp",
                y="pred_xgboost",
                title="XGBoost Predictions Over Time"
            )
    except Exception as e:
        print(f"Error in plot_load_forecast: {e}")
        return None

print("Testing plot generation...")
plot = plot_load_forecast(df)
print(f"Plot generated: {plot}")
