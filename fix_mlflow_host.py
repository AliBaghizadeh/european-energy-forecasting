import paramiko
import json
import sys

def fix_mlflow_host():
    print("🔧 Fixing MLflow server configuration...")
    
    # Load config
    try:
        with open("aws_config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print("❌ aws_config.json not found.")
        sys.exit(1)

    host = config["ec2_public_ip"]
    key_file = f"{config['key_pair_name']}.pem"
    bucket = config["bucket_name"]
    
    print(f"Connecting to {host}...")
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(hostname=host, username="ubuntu", key_filename=key_file)
        print("✅ Connected!")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        sys.exit(1)

    # Update the systemd service with gunicorn options to disable Host header check
    update_service_cmd = f"""cat <<EOF | sudo tee /etc/systemd/system/mlflow.service
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
    --default-artifact-root s3://{bucket}/ \\
    --serve-artifacts \\
    --gunicorn-opts "--timeout 120 --forwarded-allow-ips='*'"
Restart=always

[Install]
WantedBy=multi-user.target
EOF
"""
    
    commands = [
        update_service_cmd,
        "sudo systemctl daemon-reload",
        "sudo systemctl restart mlflow",
        "sleep 3",
        "sudo systemctl status mlflow --no-pager"
    ]

    for cmd in commands:
        print(f"\nRunning: {cmd.splitlines()[0][:50]}...")
        stdin, stdout, stderr = ssh.exec_command(cmd)
        exit_status = stdout.channel.recv_exit_status()
        
        if exit_status == 0:
            print("✅ Success")
        else:
            err = stderr.read().decode()
            if err:
                print(f"⚠️ {err}")

    print(f"\n🎉 MLflow server reconfigured!")
    print(f"Try the connection test again: python test_mlflow_connection.py")
    ssh.close()

if __name__ == "__main__":
    fix_mlflow_host()
