#!/usr/bin/env python3
"""
SimKGC Pipeline - Full Cloud Execution Script for Kaggle GPUs.
All pipeline options are easily configurable below in PIPELINE_CONFIG.
"""

import os
import sys
import subprocess
import shutil
import time
from pathlib import Path
from typing import Optional

# =====================================================================
# ⚙️ USER PIPELINE CONFIGURATION (Choose your options here)
# =====================================================================
PIPELINE_CONFIG = {
    # Target Hugging Face Model Repository
    "hf_repo": "Celestios/Persian-simkgc-256d",

    # Stage 0: BGE-M3 Teacher Target Encodings
    # Options: "download" (pulls precomputed ~100MB cache from HF) or "recompute" (encodes fresh on GPU)
    "teacher_targets_mode": "download",

    # Stage 1A: TextEmbedder Training (Layers 1–8)
    "run_stage_1a": True,
    "stage_1a_epochs": 4,
    "stage_1a_batch_size": 512,
    "stage_1a_resume": True,             # Resume from local checkpoint if available
    "stage_1a_from_hf": True,           # Download Stage 1A checkpoint from HF
    "stage_1a_push_to_hf": True,         # Upload Stage 1A model to HF

    # Stage 1B: RelationalCore Training (Layers 9–12)
    "run_stage_1b": True,
    "stage_1b_epochs": 10,
    "stage_1b_batch_size": 512,
    "stage_1b_resume": True,             # Resume from local checkpoint if available
    "stage_1b_from_hf": True,           # Download Stage 1B checkpoint from HF
    "stage_1b_push_to_hf": True,         # Upload Stage 1B model to HF

    # Stage 2: AssembledBiEncoder Joint Calibration & Auto-Export
    "run_stage_2": True,
    "stage_2_epochs": 15,
    "stage_2_batch_size": 512,
    "stage_2_resume": True,              # Resume from local checkpoint if available
    "stage_2_from_hf": True,            # Download Stage 2 model from HF
    "stage_2_push_to_hf": True,          # Upload full release bundle to HF
    "stage_2_auto_export": True,         # Automatically generate ONNX INT8 + 12.8MB Binary

    # Verification & Benchmarks
    "run_smoke_test": True
}
# =====================================================================

HF_REPO = PIPELINE_CONFIG["hf_repo"]
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
        "3. Merge Base Dataset (HF) with Git Synthetic Phrases & Generate Inverses",
        f"python src/data/cleaner.py --inputs data/raw/conceptnet_clean.json data/raw/conceptnet_subset.json data/synthetic/all_triplets_deduped.json --output data/raw/conceptnet_clean.json --min-weight 0.5 --from-hf {HF_REPO}",
        cwd=base_dir
    )

    # 4. Extract BGE-M3 Teacher Target Embeddings
    teacher_cmd = "python src/data/teacher_embedder.py --data data/raw/conceptnet_clean.json --out-npy cache/bge_m3_concept_targets.npy --out-dict cache/concepts_dict.json --batch-size 2048"
    if PIPELINE_CONFIG["teacher_targets_mode"] == "download":
        teacher_cmd += f" --from-hf {HF_REPO}"
    else:
        teacher_cmd += f" --force-recompute --push-to-hf --hf-repo {HF_REPO}"
        
    run_step("4. BGE-M3 Teacher Target Encodings", teacher_cmd, cwd=base_dir)

    # 5. Stage 1A: TextEmbedder Training (Layers 1-8)
    if PIPELINE_CONFIG["run_stage_1a"]:
        cmd_1a = f"python src/train_embedder.py --epochs {PIPELINE_CONFIG['stage_1a_epochs']} --batch-size {PIPELINE_CONFIG['stage_1a_batch_size']}"
        if not PIPELINE_CONFIG["stage_1a_resume"]:
            cmd_1a += " --no-resume"
        if PIPELINE_CONFIG["stage_1a_from_hf"]:
            cmd_1a += f" --from-hf {HF_REPO}"
        if PIPELINE_CONFIG["stage_1a_push_to_hf"]:
            cmd_1a += f" --push-to-hf --hf-repo {HF_REPO}"
        run_step("5. STAGE 1A: Train TextEmbedder (Layers 1-8)", cmd_1a, cwd=base_dir)

    # 6. Stage 1B: RelationalCore Training (Layers 9-12)
    if PIPELINE_CONFIG["run_stage_1b"]:
        cmd_1b = f"python src/train_core.py --epochs {PIPELINE_CONFIG['stage_1b_epochs']} --batch-size {PIPELINE_CONFIG['stage_1b_batch_size']}"
        if not PIPELINE_CONFIG["stage_1b_resume"]:
            cmd_1b += " --no-resume"
        if PIPELINE_CONFIG["stage_1b_from_hf"]:
            cmd_1b += f" --from-hf {HF_REPO}"
        if PIPELINE_CONFIG["stage_1b_push_to_hf"]:
            cmd_1b += f" --push-to-hf --hf-repo {HF_REPO}"
        run_step("6. STAGE 1B: Train RelationalCore (Layers 9-12)", cmd_1b, cwd=base_dir)

    # 7. Stage 2: Joint Assembled Model Calibration & Automated Export
    if PIPELINE_CONFIG["run_stage_2"]:
        cmd_2 = f"python src/train_joint.py --epochs {PIPELINE_CONFIG['stage_2_epochs']} --batch-size {PIPELINE_CONFIG['stage_2_batch_size']}"
        if not PIPELINE_CONFIG["stage_2_resume"]:
            cmd_2 += " --no-resume"
        if not PIPELINE_CONFIG["stage_2_auto_export"]:
            cmd_2 += " --no-export"
        if PIPELINE_CONFIG["stage_2_from_hf"]:
            cmd_2 += f" --from-hf {HF_REPO}"
        if PIPELINE_CONFIG["stage_2_push_to_hf"]:
            cmd_2 += f" --push-to-hf --hf-repo {HF_REPO}"
        run_step("7. STAGE 2: Joint Calibration & Export", cmd_2, cwd=base_dir)

    # 8. Live Smoke Test Across All Capabilities
    if PIPELINE_CONFIG["run_smoke_test"]:
        run_step("8. Verification Smoke Tests", "python src/demo_inference.py --benchmark", cwd=base_dir)

    print("\n================================================================================", flush=True)
    print(f"   [DONE] ALL ARTIFACTS DIRECTLY UPLOADED TO: https://huggingface.co/{HF_REPO}   ", flush=True)
    print("================================================================================", flush=True)

if __name__ == "__main__":
    main()
