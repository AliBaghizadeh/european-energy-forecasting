from huggingface_hub import HfApi, login
import os

def upload_models():
    print("🚀 Starting Model Upload Script")
    
    # 1. Login
    # Try to get token from env or ask user
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("\n🔑 Please enter your Hugging Face Access Token (Write permission):")
        print("(You can find it at https://huggingface.co/settings/tokens)")
        token = input("Token: ").strip()
    
    try:
        login(token=token, add_to_git_credential=True)
        print("✅ Logged in successfully")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        return

    # 2. Upload
    repo_id = "alibaghizade/time_series_energy"
    local_folder = "./Model"
    
    print(f"\n📦 Uploading '{local_folder}' to Space: {repo_id}...")
    
    try:
        api = HfApi()
        api.upload_folder(
            folder_path=local_folder,
            path_in_repo="Model",  # Upload into 'Model' folder in repo
            repo_id=repo_id,
            repo_type="space",
            commit_message="Upload models manually via script"
        )
        print("\n✅ Upload Complete! 🚀")
        print(f"Visit your Space: https://huggingface.co/spaces/{repo_id}")
        
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")

if __name__ == "__main__":
    upload_models()
