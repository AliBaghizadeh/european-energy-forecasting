# Manual Fix Instructions for MLflow Host Header Issue

## You're already SSH'd into the server!

Skip the SSH command - you're already connected. Just run the commands below directly in your SSH terminal.

## Commands to run (you're already on the server):

### 1. Stop the MLflow service
```bash
sudo systemctl stop mlflow
```

### 2. Update the service file with the correct configuration
```bash
cat <<'EOF' | sudo tee /etc/systemd/system/mlflow.service
[Unit]
Description=MLflow Tracking Server
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu
Environment=PATH=/home/ubuntu/mlflow_env/bin:/usr/local/bin:/usr/bin:/bin
Environment=MLFLOW_TRACKING_URI=http://0.0.0.0:5000
ExecStart=/home/ubuntu/mlflow_env/bin/mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root s3://european-energy-forecasting-2025/ --serve-artifacts --app-name basic-auth
Restart=always

[Install]
WantedBy=multi-user.target
EOF
```

### 3. Reload systemd and restart MLflow
```bash
sudo systemctl daemon-reload
sudo systemctl start mlflow
sudo systemctl status mlflow
```
**Note:** Press `q` to exit the status view and return to the command prompt.

### 4. Verify it's listening
```bash
curl http://localhost:5000/health
```

### 5. Exit SSH and test from your local machine
```bash
exit
```

Then run:
```cmd
python test_mlflow_connection.py
```

---

## Alternative: Use nginx as reverse proxy (if above doesn't work)

If the Host header issue persists, we can set up nginx as a reverse proxy:

```bash
sudo apt install -y nginx
sudo systemctl stop nginx

cat <<'EOF' | sudo tee /etc/nginx/sites-available/mlflow
server {
    listen 5000;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:5001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/mlflow /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default

# Update MLflow to run on port 5001
sudo sed -i 's/--port 5000/--port 5001/g' /etc/systemd/system/mlflow.service
sudo systemctl daemon-reload
sudo systemctl restart mlflow
sudo systemctl start nginx
```
