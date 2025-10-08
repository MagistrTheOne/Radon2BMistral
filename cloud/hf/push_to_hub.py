"""
Push model and tokenizer to Hugging Face Hub
"""

import os
from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer

REPO = os.environ.get("HF_REPO", "MagistrTheOne/RADON")
TOKEN = os.environ.get("HF_TOKEN")
assert TOKEN, "HF_TOKEN env is required"

api = HfApi()
create_repo(REPO, repo_type="model", private=False, exist_ok=True, token=TOKEN)

# Push tokenizer (directory with files: tokenizer.json / spiece.model / vocab)
print("[+] Pushing tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    tokenizer.push_to_hub(REPO, use_temp_dir=True, token=TOKEN)
    print("[+] Tokenizer pushed successfully")
except Exception as e:
    print(f"[!] Tokenizer push failed: {e}")

# Example: push config/artifacts (if already built)
# api.upload_file(path_or_fileobj="artifacts/model.safetensors", path_in_repo="model.safetensors", repo_id=REPO, token=TOKEN)
# api.upload_file(path_or_fileobj="configs/model_config_small.json", path_in_repo="config.json", repo_id=REPO, token=TOKEN)

print("[+] Done")

