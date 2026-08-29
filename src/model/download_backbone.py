#!/usr/bin/env python3
"""
Model Backbone Downloader with Real-Time Bandwidth Monitor & Strict Hard Cap.
Safely downloads standard transformer backbones directly from HuggingFace
supporting both safetensors and legacy PyTorch formats with live monitoring.
"""

import sys
import os
import json
import urllib.request
from pathlib import Path
from typing import List

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def download_file_with_monitor(url: str, output_path: Path, max_bytes: int = 50 * 1024 * 1024) -> bool:
    """
    Downloads a single file from HuggingFace with live byte monitoring and a strict hard limit.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filename = url.split('/')[-1]
    
    print("=" * 65)
    print(f"FETCHING:   {filename}")
    print(f"HARD LIMIT: {max_bytes / 1024 / 1024:.2f} MB")
    print("=" * 65)

    req = urllib.request.Request(url, headers={"User-Agent": "CentrodeModelDownloader/1.0"})
    downloaded_bytes = 0
    chunk_size = 64 * 1024

    try:
        with urllib.request.urlopen(req, timeout=30) as response, open(output_path, "wb") as f:
            total_size = int(response.headers.get("Content-Length", max_bytes))
            limit_to_use = min(total_size, max_bytes)

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break

                downloaded_bytes += len(chunk)
                f.write(chunk)

                pct = min(100.0, (downloaded_bytes / limit_to_use) * 100.0)
                sys.stdout.write(f"\r[MODEL MONITOR] {downloaded_bytes / 1024 / 1024:.2f} MB / {limit_to_use / 1024 / 1024:.2f} MB ({pct:.1f}%)")
                sys.stdout.flush()

                if downloaded_bytes >= max_bytes:
                    print("\n[GUARD TRIGGERED] File reached max byte cap.")
                    break

        print(f"\n[OK] Saved to {output_path} ({downloaded_bytes / 1024 / 1024:.2f} MB)")
        return True
    except urllib.error.HTTPError as e:
        if output_path.exists():
            output_path.unlink()
        if e.code == 404:
            return False
        print(f"\nHTTP Error downloading {filename}: {e}")
        return False
    except Exception as e:
        if output_path.exists():
            output_path.unlink()
        print(f"\nError downloading {filename}: {e}")
        return False

def download_huggingface_model(repo_id: str, target_dir: Path, max_total_mb: float = 80.0):
    """
    Downloads config, tokenizer files, and weights for a model from Hugging Face safely.
    Handles both modern (safetensors + tokenizer.json) and legacy (pytorch_model.bin + vocab.txt).
    """
    base_url = f"https://huggingface.co/{repo_id}/resolve/main"
    max_bytes = int(max_total_mb * 1024 * 1024)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting controlled download of [{repo_id}] (Cap: {max_total_mb} MB)...")
    
    # 1. Config
    download_file_with_monitor(f"{base_url}/config.json", target_dir / "config.json", max_bytes)
    
    # 2. Tokenizer (Try tokenizer.json first, fallback to vocab.txt)
    tok_saved = download_file_with_monitor(f"{base_url}/tokenizer.json", target_dir / "tokenizer.json", max_bytes)
    if not tok_saved:
        print("tokenizer.json not found. Trying legacy vocab.txt & tokenizer_config.json...")
        download_file_with_monitor(f"{base_url}/vocab.txt", target_dir / "vocab.txt", max_bytes)
        download_file_with_monitor(f"{base_url}/tokenizer_config.json", target_dir / "tokenizer_config.json", max_bytes)
        download_file_with_monitor(f"{base_url}/special_tokens_map.json", target_dir / "special_tokens_map.json", max_bytes)

    # 3. Weights (Try model.safetensors first, fallback to pytorch_model.bin)
    weight_saved = download_file_with_monitor(f"{base_url}/model.safetensors", target_dir / "model.safetensors", max_bytes)
    if not weight_saved:
        print("model.safetensors not found. Trying pytorch_model.bin...")
        download_file_with_monitor(f"{base_url}/pytorch_model.bin", target_dir / "pytorch_model.bin", max_bytes)

    print("\n" + "=" * 65)
    print(f"[COMPLETE] Model assets for [{repo_id}] saved to {target_dir}")
    print("=" * 65)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download model backbone safely.")
    parser.add_argument("--repo", type=str, default="prajjwal1/bert-tiny", help="HuggingFace model ID")
    parser.add_argument("--output", type=str, default="models/backbone", help="Output directory")
    parser.add_argument("--cap", type=float, default=50.0, help="Max MB download cap")
    args = parser.parse_args()

    download_huggingface_model(args.repo, Path(args.output), max_total_mb=args.cap)
