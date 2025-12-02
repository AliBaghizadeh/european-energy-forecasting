"""
Script to manually upload weather_api.py to Hugging Face Space
Run this after adding your HF_TOKEN as an environment variable
"""
import os
from huggingface_hub import login, upload_file

# Configuration
REPO_ID = "alibaghizade/time_series_energy"
REPO_TYPE = "space"

# Login (you'll need to set HF_TOKEN environment variable)
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    print("❌ Error: HF_TOKEN environment variable not set")
    print("Please set it with: $env:HF_TOKEN='your_token_here'")
    exit(1)

login(token=hf_token, add_to_git_credential=False)

# Upload weather_api.py
print("Uploading weather_api.py to Hugging Face Space...")
upload_file(
    path_or_fileobj='./App/weather_api.py',
    path_in_repo='weather_api.py',
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
    commit_message='Add weather_api module'
)

print("✅ Successfully uploaded weather_api.py!")
print("The Space should restart automatically.")
