#!/bin/bash
# Complete nginx + MLflow setup script
# Run this on the EC2 instance

echo "Installing nginx..."
sudo apt update
sudo apt install -y nginx

echo "Creating nginx config for MLflow..."
cat <<'NGINX_EOF' | sudo tee /etc/nginx/sites-available/mlflow
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
NGINX_EOF

echo "Enabling nginx config..."
sudo ln -sf /etc/nginx/sites-available/mlflow /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

echo "Creating MLflow systemd service..."
cat <<'MLFLOW_EOF' | sudo tee /etc/systemd/system/mlflow.service
[Unit]
Description=MLflow Tracking Server
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu
Environment=PATH=/home/ubuntu/mlflow_env/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/home/ubuntu/mlflow_env/bin/mlflow server --host 127.0.0.1 --port 5001 --backend-store-uri sqlite:///mlflow.db --default-artifact-root s3://european-energy-forecasting-2025/ --serve-artifacts
Restart=always

[Install]
WantedBy=multi-user.target
MLFLOW_EOF

echo "Restarting services..."
sudo systemctl daemon-reload
sudo systemctl restart mlflow
sudo systemctl restart nginx
sudo systemctl enable nginx

echo "Waiting for services to start..."
sleep 5

echo "Testing connection..."
curl http://localhost:5000/health

echo ""
echo "If you see 'OK' above, the setup is complete!"
echo "Exit SSH and run: python test_mlflow_connection.py"
