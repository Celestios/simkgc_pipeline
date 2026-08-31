#!/usr/bin/env python3
"""
SimKGC Pipeline - Full Cloud Execution Script for Kaggle GPUs.
Runs 100% in the cloud and directly uploads all trained models & production exports to Hugging Face Hub:
  1. Setup & Environment
  2. Data Cleaning & Inverse Generation
  3. BGE-M3 Teacher Vector Generation
  4. Stage 1A: TextEmbedder Training (Layers 1–8) -> Pushes embedder_l1_8_final.pt to HF
  5. Stage 1B: RelationalCore Training (Layers 9–12) -> Pushes relational_core_l9_12_final.pt to HF
  6. Stage 2: Joint Model Calibration & Auto-Export -> Pushes full release bundle to HF
  7. Verification Smoke Tests
"""

import os
import sys
import subprocess
import shutil
import time
from pathlib import Path
from typing import Optional

# Read HF token from environment or Kaggle Secrets
HF_REPO = "Celestios/Persian-simkgc-256d"
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        HF_TOKEN = user_secrets.get_secret("HF_TOKEN")
        if HF_TOKEN:
            os.environ["HF_TOKEN"] = HF_TOKEN
    except Exception:
        pass

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

def run_step(title: str, cmd: str, cwd: Optional[Path] = None):
    print("\n" + "=" * 80, flush=True)
    print(f"  >>> {title}", flush=True)
    print("=" * 80, flush=True)
    print(f"Executing: {cmd}\n", flush=True)
    
    start_time = time.time()
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=str(cwd) if cwd else None,
        text=True,
        bufsize=1
    )
    elapsed = time.time() - start_time
    if result.returncode != 0:
        print(f"\n[ERROR] Step failed with return code {result.returncode} after {elapsed:.1f}s", flush=True)
        sys.exit(result.returncode)
    print(f"\n[SUCCESS] {title} completed in {elapsed:.1f}s", flush=True)

def main():
    print("================================================================================", flush=True)
    print("   SIMKGC 256D MULTILINGUAL PIPELINE - 100% CLOUD & DIRECT HUGGING FACE SYNC     ", flush=True)
    print(f"   Target Hugging Face Hub: https://huggingface.co/{HF_REPO}                     ", flush=True)
    print("================================================================================", flush=True)
    
    base_dir = Path("/kaggle/working/simkgc_pipeline")
    
    # 1. Clone or Pull Repo
    if not base_dir.exists():
        run_step("1. Clone Repository", "git clone https://github.com/Celestios/simkgc_pipeline.git /kaggle/working/simkgc_pipeline")
    else:
        run_step("1. Update Repository", "git pull origin main", cwd=base_dir)
        
    os.chdir(base_dir)
    print(f"Working Directory: {os.getcwd()}", flush=True)

    # 2. Install Dependencies
    run_step(
        "2. Install Dependencies",
        "pip install -q torch transformers accelerate datasets onnx onnxruntime safetensors pyyaml requests tqdm huggingface_hub"
    )

    # 3. Clean & Prepare Knowledge Graph Data
    raw_dir = base_dir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    subset_path = raw_dir / "conceptnet_subset.json"
    
    if not subset_path.exists():
        matches = list(Path("/kaggle/working").rglob("conceptnet_subset.json"))
        if matches and matches[0] != subset_path:
            shutil.move(str(matches[0]), str(subset_path))
            print(f"Moved ConceptNet subset to: {subset_path}", flush=True)
        else:
            run_step("Download ConceptNet (~450MB)", "python src/data/download_conceptnet_subset.py --full", cwd=base_dir)

    run_step(
        "3. Clean Triples & Generate Inverses",
        "python src/data/cleaner.py --inputs data/raw/conceptnet_subset.json data/synthetic/all_triplets_deduped.json --output data/raw/conceptnet_clean.json --min-weight 0.5",
        cwd=base_dir
    )

    # 4. Extract BGE-M3 Teacher Target Embeddings
    run_step(
        "4. BGE-M3 Teacher Vector Generation",
        f"python src/data/teacher_embedder.py --data data/raw/conceptnet_clean.json --out-npy cache/bge_m3_concept_targets.npy --out-dict cache/concepts_dict.json --from-hf {HF_REPO} --batch-size 2048",
        cwd=base_dir
    )

    # 5. Stage 1A: TextEmbedder Training (Layers 1-8) & Direct Push to HF
    run_step(
        "5. STAGE 1A: Train TextEmbedder (Layers 1-8) -> Upload to HF",
        f"python src/train_embedder.py --push-to-hf --hf-repo {HF_REPO}",
        cwd=base_dir
    )

    # 6. Stage 1B: RelationalCore Training (Layers 9-12) & Direct Push to HF
    run_step(
        "6. STAGE 1B: Train RelationalCore (Layers 9-12) -> Upload to HF",
        f"python src/train_core.py --push-to-hf --hf-repo {HF_REPO}",
        cwd=base_dir
    )

    # 7. Stage 2: Joint Assembled Model Calibration, Auto-Export & Direct Push to HF
    run_step(
        "7. STAGE 2: Joint Calibration, Export & Full Hub Release -> Upload to HF",
        f"python src/train_joint.py --push-to-hf --hf-repo {HF_REPO}",
        cwd=base_dir
    )

    # 8. Live Smoke Test Across All Capabilities
    run_step(
        "8. Verification Smoke Tests",
        "python src/demo_inference.py --benchmark",
        cwd=base_dir
    )

    print("\n================================================================================", flush=True)
    print(f"   [DONE] ALL ARTIFACTS DIRECTLY UPLOADED TO: https://huggingface.co/{HF_REPO}   ", flush=True)
    print("================================================================================", flush=True)

if __name__ == "__main__":
    main()
