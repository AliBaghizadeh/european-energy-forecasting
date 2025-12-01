# Define the project's default python environment
PYTHON = python

# -----------------
# Core ML Targets
# -----------------

# 1. install: Installs/updates necessary Python packages
install:
	pip install --upgrade pip && \
	pip install -r requirements.txt

# 2. train: Runs the main training script
train:
	$(PYTHON) train.py

# 3. eval: Generates the CML markdown report using the saved metrics and plot
eval:
	echo "## ⚡ Energy Load Forecast Model Report" > report.md
	echo "---" >> report.md
	echo "### 📊 Performance Metrics (Time Series Regression)" >> report.md
	cat ./Results/metrics.txt >> report.md
	echo '\n### 📉 Forecast Visualization' >> report.md
	echo '![Forecast vs Actual](./Results/model_results.png)' >> report.md
	cml comment create report.md

# -----------------
# Utility Targets
# -----------------

# format: For code quality
format:
	black .

# -----------------
# CI/CD Targets
# -----------------

# update-branch: Commits and pushes Results (Model/ is now gitignored)
update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	mkdir -p Results
	git add Results
	git commit -m "Update with new results from $(GITHUB_SHA)" --allow-empty
	git push --force origin HEAD:update

# -----------------
# Deployment Targets
# -----------------

# hf-login: Logs in using the token
hf-login:
	huggingface-cli login --token $(HF_TOKEN) --add-to-git-credential

# push-hub: Uploads files to HuggingFace Space
push-hub:
	huggingface-cli upload alibaghizade/time_series_energy ./App/energy_app.py app.py --repo-type=space --commit-message="Deploy App"
	huggingface-cli upload alibaghizade/time_series_energy ./requirements.txt requirements.txt --repo-type=space --commit-message="Sync Requirements"
	huggingface-cli upload alibaghizade/time_series_energy ./Model /Model --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload alibaghizade/time_series_energy ./Results /Metrics --repo-type=space --commit-message="Sync Metrics"

# deploy: Runs the login followed by the push
deploy: hf-login push-hub

# -----------------
# MLflow Targets
# -----------------

# mlflow-ui: Launches the MLflow UI to view experiments
mlflow-ui:
	mlflow ui

# hpo: Runs the Hyperparameter Optimization script
hpo:
	$(PYTHON) train_hpo.py

# register-models: Registers optimized models in MLflow Model Registry
register-models:
	$(PYTHON) register_models.py