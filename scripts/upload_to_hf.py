#!/usr/bin/env python3
"""
Upload all available trained checkpoints, teacher targets, and export bundles to Hugging Face Hub.
Works even if training was interrupted or failed mid-way by finding the latest saved epoch checkpoints.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.checkpoint import upload_file_to_hf, get_resolved_hf_token, find_latest_local_checkpoint

DEFAULT_HF_REPO = "Celestios/Persian-simkgc-256d"

def upload_all_assets(
    repo_id: str = DEFAULT_HF_REPO,
    token: str = None,
    base_dir: str = "."
):
    resolved_token = get_resolved_hf_token(token)
    if not resolved_token:
        print("[ERROR] No Hugging Face token found. Provide --token or set HF_TOKEN environment variable.")
        sys.exit(1)

    base = Path(base_dir)
    print("=" * 75)
    print(f"  SCANNING & UPLOADING ARTIFACTS TO: https://huggingface.co/{repo_id}")
    print("=" * 75)

    files_to_upload = []

    # 1. Stage 0 Targets
    cache_dir = base / "cache"
    if (cache_dir / "bge_m3_concept_targets.npy").exists():
        files_to_upload.append((cache_dir / "bge_m3_concept_targets.npy", "bge_m3_concept_targets.npy"))
    if (cache_dir / "concepts_dict.json").exists():
        files_to_upload.append((cache_dir / "concepts_dict.json", "concepts_dict.json"))

    # 2. Stage 1A Checkpoints
    stage1a_dir = base / "checkpoints" / "stage1_embedder"
    final_1a = stage1a_dir / "embedder_l1_8_final.pt"
    if final_1a.exists():
        files_to_upload.append((final_1a, "embedder_l1_8_final.pt"))
    else:
        latest_1a, epoch_1a = find_latest_local_checkpoint(stage1a_dir, prefix="embedder_epoch_")
        if latest_1a and latest_1a.exists():
            print(f"[Stage 1A] Found latest epoch checkpoint: {latest_1a.name} (Epoch {epoch_1a})")
            files_to_upload.append((latest_1a, "embedder_l1_8_final.pt"))

    # Also upload individual epoch checkpoints if present
    if stage1a_dir.exists():
        for ep_f in sorted(stage1a_dir.glob("embedder_epoch_*.pt")):
            files_to_upload.append((ep_f, ep_f.name))

    # 3. Stage 1B Checkpoints
    stage1b_dir = base / "checkpoints" / "stage1_core"
    final_1b = stage1b_dir / "relational_core_l9_12_final.pt"
    if final_1b.exists():
        files_to_upload.append((final_1b, "relational_core_l9_12_final.pt"))
    else:
        latest_1b, epoch_1b = find_latest_local_checkpoint(stage1b_dir, prefix="core_epoch_")
        if latest_1b and latest_1b.exists():
            print(f"[Stage 1B] Found latest epoch checkpoint: {latest_1b.name} (Epoch {epoch_1b})")
            files_to_upload.append((latest_1b, "relational_core_l9_12_final.pt"))

    if stage1b_dir.exists():
        for ep_f in sorted(stage1b_dir.glob("core_epoch_*.pt")):
            files_to_upload.append((ep_f, ep_f.name))

    # 4. Stage 2 Checkpoints
    stage2_dir = base / "checkpoints" / "simkgc_fa_en"
    final_2 = stage2_dir / "simkgc_model.pt"
    if final_2.exists():
        files_to_upload.append((final_2, "simkgc_model.pt"))
    else:
        latest_2, epoch_2 = find_latest_local_checkpoint(stage2_dir, prefix="assembled_epoch_")
        if latest_2 and latest_2.exists():
            print(f"[Stage 2] Found latest epoch checkpoint: {latest_2.name} (Epoch {epoch_2})")
            files_to_upload.append((latest_2, "simkgc_model.pt"))

    if stage2_dir.exists():
        for ep_f in sorted(stage2_dir.glob("assembled_epoch_*.pt")):
            files_to_upload.append((ep_f, ep_f.name))
        if (stage2_dir / "config.json").exists():
            files_to_upload.append((stage2_dir / "config.json", "config.json"))

    # 5. Production Runtime Exports
    export_dir = base / "exports"
    if export_dir.exists():
        for export_file in export_dir.glob("*"):
            if export_file.is_file():
                files_to_upload.append((export_file, export_file.name))

    if not files_to_upload:
        print("[WARNING] No checkpoints or export files found on disk to upload.")
        return

    # Upload all discovered files
    uploaded_count = 0
    for local_path, repo_path in files_to_upload:
        size_mb = local_path.stat().st_size / (1024 * 1024)
        if size_mb >= 1.0:
            print(f"\n[*] Uploading {local_path.name:<32} ({size_mb:.2f} MB) -> {repo_path}")
        else:
            print(f"\n[*] Uploading {local_path.name:<32} ({local_path.stat().st_size/1024:.1f} KB) -> {repo_path}")
            
        success = upload_file_to_hf(local_path, path_in_repo=repo_path, repo_id=repo_id, token=resolved_token)
        if success:
            uploaded_count += 1

    print("\n" + "=" * 75)
    print(f"  [DONE] {uploaded_count}/{len(files_to_upload)} artifacts uploaded to: https://huggingface.co/{repo_id}")
    print("=" * 75)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload checkpoints and exports to Hugging Face")
    parser.add_argument("--repo", default=DEFAULT_HF_REPO, help="Target Hugging Face repo ID")
    parser.add_argument("--token", default=None, help="Hugging Face API write token")
    parser.add_argument("--dir", default=".", help="Base directory containing checkpoints/ and exports/")
    args = parser.parse_args()

    upload_all_assets(repo_id=args.repo, token=args.token, base_dir=args.dir)
