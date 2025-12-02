import paramiko
import json
import time
import os
import sys

def configure_server():
    print("🚀 Starting Server Configuration...")
    
    # Load config
    try:
        with open("aws_config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print("❌ aws_config.json not found. Run setup_aws_infra.py first.")
        sys.exit(1)

    host = config["ec2_public_ip"]
    key_file = f"{config['key_pair_name']}.pem"
    bucket = config["bucket_name"]
    
    print(f"Connecting to {host} using {key_file}...")
    
    # Wait for SSH to be ready (instance might still be booting)
    retries = 5
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    for i in range(retries):
        try:
            ssh.connect(hostname=host, username="ubuntu", key_filename=key_file)
            print("✅ SSH Connection established!")
            break
        except Exception as e:
            print(f"⏳ Waiting for SSH ({i+1}/{retries})... {e}")
            time.sleep(10)
    else:
        print("❌ Could not connect to server.")
        sys.exit(1)

    # Commands to run
    commands = [
        "sudo apt update",
        "sudo apt install -y python3-pip python3-venv",
        "python3 -m venv mlflow_env",
        "source mlflow_env/bin/activate && pip install mlflow boto3",
        # Create systemd service for MLflow
        f"""cat <<EOF | sudo tee /etc/systemd/system/mlflow.service
[Unit]
Description=MLflow Tracking Server
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu
Environment=PATH=/home/ubuntu/mlflow_env/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/home/ubuntu/mlflow_env/bin/mlflow server \\
    --host 0.0.0.0 \\
    --port 5000 \\
    --backend-store-uri sqlite:///mlflow.db \\
    --default-artifact-root s3://{bucket}/
Restart=always

[Install]
WantedBy=multi-user.target
EOF
""",
        "sudo systemctl daemon-reload",
        "sudo systemctl enable mlflow",
        "sudo systemctl start mlflow",
        "sudo systemctl status mlflow --no-pager"
    ]

    for cmd in commands:
        print(f"\nRunning: {cmd.splitlines()[0]}...")
        stdin, stdout, stderr = ssh.exec_command(cmd)
        exit_status = stdout.channel.recv_exit_status()
        
        if exit_status == 0:
            print("✅ Success")
        else:
            print(f"❌ Error: {stderr.read().decode()}")
            # Don't exit on status check failure, just print
            if "status" not in cmd:
                sys.exit(1)

    print(f"\n🎉 MLflow Server is running at: http://{host}:5000")
    ssh.close()

if __name__ == "__main__":
    configure_server()
