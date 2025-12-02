# Final Verification & Deployment Guide

## 1. Verify AWS MLflow Integration
First, let's make sure your training script is correctly logging to your new AWS server.

1. **Run the optimized training script**:
   ```cmd
   python train_hpo.py
   ```
   *This will run the hyperparameter search (approx. 5-10 mins with the new settings).*

2. **Check the MLflow UI**:
   - Open your browser to: [http://18.153.53.234:5000](http://18.153.53.234:5000)
   - You should see:
     - Experiments named `XGBoost_HPO`, `LightGBM_HPO`, etc.
     - Runs with metrics (MAE) and parameters.
     - Artifacts (models) stored in S3.

## 2. Verify the App Locally
Your app uses the models in the `Model/` folder. `train_hpo.py` automatically updates these files.

1. **Run the app locally**:
   ```cmd
   python App/energy_app.py
   ```
2. **Open the local URL** (usually `http://127.0.0.1:7860`).
3. **Test a prediction**:
   - Select a country (e.g., DE).
   - Click "Fetch Weather".
   - Click "Predict Energy Load".
   - Verify that predictions appear and plots are generated.

## 3. Update GitHub & Hugging Face
Now let's save your work and deploy the new models to the cloud.

1. **Commit changes to GitHub**:
   ```cmd
   git add .
   git commit -m "Migrate MLflow to AWS and optimize HPO"
   git push
   ```

2. **Deploy to Hugging Face Spaces**:
   *If your Hugging Face Space is connected to this repo, the push above will trigger a deployment.*
   
   If you need to upload manually:
   - Go to your Space: [https://huggingface.co/spaces/alibaghizade/time_series_energy](https://huggingface.co/spaces/alibaghizade/time_series_energy)
   - Go to **Files**.
   - Upload the new files from your `Model/` folder (`xgboost_model.pkl`, etc.).
   - Upload the updated `App/energy_app.py` (if you made changes).

## 4. Shutting Down (Cost Saving)
Since you are using AWS Free Tier, remember to **stop your EC2 instance** when you are not using it to save hours.

- **Stop**: Go to AWS Console -> EC2 -> Select Instance -> Instance State -> Stop.
- **Start**: When you want to train again, just click "Start".
  - *Note: Your Public IP might change when you restart! You'll need to update `MLFLOW_TRACKING_URI` if it does.*
