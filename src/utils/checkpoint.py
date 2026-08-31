#!/usr/bin/env python3
"""
Checkpoint Management & Hugging Face Hub Synchronization.
Supports:
  1. Local disk auto-resume from the latest epoch checkpoint.
  2. Pulling specific stage checkpoints from Hugging Face Hub.
  3. Pushing isolated stage models (Stage 1A, Stage 1B, Stage 2) to Hugging Face Hub.
"""

import os
import re
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict

try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:
    HfApi = hf_hub_download = None

DEFAULT_HF_REPO = "Celestios/Persian-simkgc-256d"

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
    token: Optional[str] = None
) -> Optional[Path]:
    """Downloads a file from Hugging Face Hub into local directory."""
    if hf_hub_download is None:
        print("[WARNING] huggingface_hub is not installed. Cannot download from HF.")
        return None
    try:
        token = token or os.environ.get("HF_TOKEN")
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"[HF Sync] Downloading '{filename}' from https://huggingface.co/{repo_id}...")
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_dir),
            token=token,
            local_dir_use_symlinks=False
        )
        p = Path(downloaded)
        if p.exists():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"✓ Downloaded {p.name} ({size_mb:.2f} MB)")
            return p
    except Exception as e:
        print(f"[HF Sync] Download failed or file not found on HF: {e}")
    return None

def upload_file_to_hf(
    local_file: Path,
    path_in_repo: Optional[str] = None,
    repo_id: str = DEFAULT_HF_REPO,
    token: Optional[str] = None,
    commit_message: Optional[str] = None
) -> bool:
    """Uploads a single checkpoint/artifact file to Hugging Face Hub."""
    if HfApi is None:
        print("[WARNING] huggingface_hub is not installed. Cannot upload to HF.")
        return False
    local_file = Path(local_file)
    if not local_file.exists():
        print(f"[HF Sync] File not found for upload: {local_file}")
        return False
    try:
        token = token or os.environ.get("HF_TOKEN")
        path_in_repo = path_in_repo or local_file.name
        commit_msg = commit_message or f"Upload {path_in_repo}"
        print(f"[HF Sync] Uploading '{local_file.name}' to https://huggingface.co/{repo_id} ({path_in_repo})...")
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(local_file),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_msg
        )
        print(f"✓ Successfully uploaded {path_in_repo} to Hugging Face!")
        return True
    except Exception as e:
        print(f"[HF Sync] Upload to HF failed: {e}")
        return False
