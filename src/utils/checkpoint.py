#!/usr/bin/env python3
"""
Robust Checkpoint Management & Hugging Face Hub Synchronization.
Features:
  1. Automatic token resolution from arguments, environment, Kaggle Secrets, and ~/.cache/huggingface/token.
  2. Automatic Exponential Backoff Retry (5 attempts) for all downloads & uploads.
  3. Non-blocking Background Upload Threading to prevent network lag from interrupting training.
"""

import os
import re
import time
import json
import threading
from pathlib import Path
from typing import Optional, Tuple, List, Dict

try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:
    HfApi = hf_hub_download = None

DEFAULT_HF_REPO = "Celestios/Persian-simkgc-256d"

def get_resolved_hf_token(explicit_token: Optional[str] = None) -> Optional[str]:
    """Resolves Hugging Face API token from explicit arg, env, Kaggle Secrets, or local cache."""
    if explicit_token and explicit_token.strip():
        return explicit_token.strip()
    
    # 1. Environment Variable
    env_token = os.environ.get("HF_TOKEN")
    if env_token and env_token.strip():
        return env_token.strip()

    # 2. Kaggle Secrets
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        sec = user_secrets.get_secret("HF_TOKEN")
        if sec and sec.strip():
            return sec.strip()
    except Exception:
        pass

    # 3. Standard Hugging Face cache files
    cache_paths = [
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".huggingface" / "token"
    ]
    for cp in cache_paths:
        if cp.exists():
            try:
                t = cp.read_text(encoding="utf-8").strip()
                if t:
                    return t
            except Exception:
                pass
                
    return None

def find_latest_local_checkpoint(output_dir: Path, prefix: str = "epoch_") -> Tuple[Optional[Path], int]:
    """Finds highest completed epoch checkpoint in directory."""
    if not output_dir.exists():
        return None, 0
    epoch_files = list(output_dir.glob(f"*{prefix}*.pt"))
    if not epoch_files:
        return None, 0
    epochs = []
    for f in epoch_files:
        match = re.search(rf"{prefix}(\d+)\.pt", f.name)
        if match:
            epochs.append((int(match.group(1)), f))
    if not epochs:
        return None, 0
    epochs.sort(key=lambda x: x[0], reverse=True)
    return epochs[0][1], epochs[0][0]

def download_from_hf(
    filename: str,
    local_dir: Path,
    repo_id: str = DEFAULT_HF_REPO,
    token: Optional[str] = None,
    max_retries: int = 5
) -> Optional[Path]:
    """Downloads a file from Hugging Face Hub with exponential backoff retries."""
    if hf_hub_download is None:
        print("[WARNING] huggingface_hub is not installed. Cannot download from HF.")
        return None

    resolved_token = get_resolved_hf_token(token)
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[HF Sync] Downloading '{filename}' from {repo_id} (Attempt {attempt}/{max_retries})...")
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(local_dir),
                token=resolved_token,
                local_dir_use_symlinks=False
            )
            p = Path(downloaded)
            if p.exists():
                size_mb = p.stat().st_size / (1024 * 1024)
                print(f"[OK] Downloaded {p.name} ({size_mb:.2f} MB)")
                return p
        except Exception as e:
            wait_time = 2 ** attempt
            print(f"[HF Sync] Download failed (Attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                print(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print("[HF Sync] Max retries reached for download.")
    return None

def _perform_upload(
    local_file: Path,
    path_in_repo: str,
    repo_id: str,
    token: Optional[str],
    commit_msg: str,
    max_retries: int = 5
) -> bool:
    """Internal upload procedure with exponential backoff retries."""
    resolved_token = get_resolved_hf_token(token)
    if not resolved_token:
        print("[WARNING] No valid HF_TOKEN found. Cannot authenticate upload to Hugging Face.")
        return False

    api = HfApi(token=resolved_token)
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[HF Sync] Uploading '{local_file.name}' -> {repo_id}/{path_in_repo} (Attempt {attempt}/{max_retries})...")
            api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
            api.upload_file(
                path_or_fileobj=str(local_file),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_msg
            )
            print(f"[OK] Successfully uploaded {path_in_repo} to Hugging Face!")
            return True
        except Exception as e:
            wait_time = 2 ** attempt
            print(f"[HF Sync] Upload attempt {attempt} failed: {e}")
            if attempt < max_retries:
                print(f"Retrying upload in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"[HF Sync] Max upload retries reached for {local_file.name}.")
    return False

def upload_file_to_hf(
    local_file: Path,
    path_in_repo: Optional[str] = None,
    repo_id: str = DEFAULT_HF_REPO,
    token: Optional[str] = None,
    commit_message: Optional[str] = None,
    async_upload: bool = False
) -> bool:
    """
    Uploads a single checkpoint/artifact file to Hugging Face Hub.
    If async_upload=True, executes in a daemon background thread so training is never stalled.
    """
    if HfApi is None:
        print("[WARNING] huggingface_hub is not installed. Cannot upload to HF.")
        return False
    local_file = Path(local_file)
    if not local_file.exists():
        print(f"[HF Sync] File not found for upload: {local_file}")
        return False

    path_in_repo = path_in_repo or local_file.name
    commit_msg = commit_message or f"Upload {path_in_repo}"

    if async_upload:
        thread = threading.Thread(
            target=_perform_upload,
            args=(local_file, path_in_repo, repo_id, token, commit_msg),
            daemon=True
        )
        thread.start()
        print(f"[HF Sync] Background upload started for {path_in_repo} (non-blocking).")
        return True
    else:
        return _perform_upload(local_file, path_in_repo, repo_id, token, commit_msg)
