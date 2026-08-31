#!/usr/bin/env python3
"""
Teacher Embedder for SimKGC Pipeline.
Uses BAAI/bge-m3 (560M parameters) to pre-encode concept terms into high-quality
256-dimensional normalized vectors with incremental disk persistence and resume support.
Supports downloading precomputed embeddings directly from Hugging Face Hub.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

class BGEConceptProjector(nn.Module):
    """
    Projects BGE-M3's 1024-d representations down to 256-d normalized space.
    Uses an orthogonal projection head initialized deterministically.
    """
    def __init__(self, input_dim: int = 1024, output_dim: int = 256, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.orthogonal_(self.projection.weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.projection(x)
        return F.normalize(projected, p=2, dim=-1)

def try_download_from_hf(
    repo_id: str,
    output_npy_path: Path,
    output_dict_path: Path
) -> bool:
    """Attempts to download precomputed teacher targets directly from Hugging Face Hub."""
    try:
        from huggingface_hub import hf_hub_download
        print(f"\n[Teacher Embedder] Checking for pre-encoded targets on Hugging Face: {repo_id}...")
        
        output_npy_path.parent.mkdir(parents=True, exist_ok=True)
        output_dict_path.parent.mkdir(parents=True, exist_ok=True)

        npy_file = hf_hub_download(
            repo_id=repo_id,
            filename=output_npy_path.name,
            local_dir=str(output_npy_path.parent),
            local_dir_use_symlinks=False
        )
        dict_file = hf_hub_download(
            repo_id=repo_id,
            filename=output_dict_path.name,
            local_dir=str(output_dict_path.parent),
            local_dir_use_symlinks=False
        )

        if Path(npy_file).exists() and Path(dict_file).exists():
            size_mb = Path(npy_file).stat().st_size / (1024 * 1024)
            print(f"✓ Successfully downloaded pre-encoded targets from Hugging Face ({size_mb:.2f} MB)")
            return True
    except Exception as e:
        print(f"[Teacher Embedder] Could not fetch from Hugging Face ({e}). Proceeding to local GPU encoding.")
    return False

def extract_bge_teacher_embeddings(
    concepts: List[str],
    output_npy_path: Path,
    output_dict_path: Path,
    teacher_model_name: str = "BAAI/bge-m3",
    output_dim: int = 256,
    batch_size: int = 1024,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """
    Encodes unique concepts using BGE-M3 on GPU.
    Streams embeddings directly to disk with np.memmap so progress is continuously persisted.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    output_npy_path = Path(output_npy_path)
    output_dict_path = Path(output_dict_path)
    output_npy_path.parent.mkdir(parents=True, exist_ok=True)
    output_dict_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_concepts = len(concepts)
    temp_dat_path = output_npy_path.with_suffix(".mmap.dat")
    progress_file = output_npy_path.parent / "teacher_embed_progress.json"
    
    start_idx = 0
    # Check for existing resumable progress
    if temp_dat_path.exists() and progress_file.exists():
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                pdata = json.load(f)
            if pdata.get("total") == total_concepts:
                start_idx = pdata.get("completed_count", 0)
                print(f"[Teacher Embedder] Resuming existing progress: {start_idx:,}/{total_concepts:,} concepts already saved.")
        except Exception:
            start_idx = 0

    mode = "r+" if temp_dat_path.exists() and start_idx > 0 else "w+"
    mmap_matrix = np.memmap(
        str(temp_dat_path),
        dtype=np.float32,
        mode=mode,
        shape=(total_concepts, output_dim)
    )

    if start_idx >= total_concepts:
        print(f"[Teacher Embedder] All {total_concepts:,} concepts already completed on disk!")
    else:
        print(f"\n[Teacher Embedder] Loading Teacher Model: {teacher_model_name} on {device}...")
        tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
        model = AutoModel.from_pretrained(teacher_model_name)
        model.eval()
        model.to(device)
        
        projector = BGEConceptProjector(input_dim=model.config.hidden_size, output_dim=output_dim)
        projector.eval()
        projector.to(device)
        
        if torch.cuda.device_count() > 1:
            print(f"[Teacher Embedder] Using {torch.cuda.device_count()} GPUs with DataParallel.")
            model = nn.DataParallel(model)
            projector = nn.DataParallel(projector)
            
        print(f"[Teacher Embedder] Encoding {total_concepts - start_idx:,} remaining concepts (Batch size: {batch_size})...")
        pbar = tqdm(total=total_concepts, initial=start_idx, desc="BGE-M3 Concept Encoding", unit="concept")
        
        for i in range(start_idx, total_concepts, batch_size):
            chunk = concepts[i:i + batch_size]
            inputs = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt"
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                # Attention-masked mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                sum_emb = torch.sum(outputs.last_hidden_state * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled = sum_emb / sum_mask
                
                # Project to 256-d normalized space
                proj_emb = projector(pooled)
                mmap_matrix[i:i + len(chunk)] = proj_emb.float().cpu().numpy()
                
            pbar.update(len(chunk))
            
            # Flush to disk every 10,000 concepts to guarantee persistence against connection drops
            current_done = i + len(chunk)
            if current_done % 10000 == 0 or current_done == total_concepts:
                mmap_matrix.flush()
                with open(progress_file, "w", encoding="utf-8") as f:
                    json.dump({"completed_count": current_done, "total": total_concepts}, f)
                    
        pbar.close()

    # Finalize: flush and copy to permanent .npy file
    mmap_matrix.flush()
    print(f"\n[Teacher Embedder] Finalizing persistent array to: {output_npy_path}...")
    np.save(output_npy_path, np.array(mmap_matrix))
    
    with open(output_dict_path, "w", encoding="utf-8") as f:
        json.dump(concepts, f, ensure_ascii=False, indent=2)
        
    # Clean up temporary progress tracker and temp mmap
    if progress_file.exists():
        progress_file.unlink()
    if temp_dat_path.exists():
        temp_dat_path.unlink()
        
    print(f"[Teacher Embedder] Saved target array to {output_npy_path} ({output_npy_path.stat().st_size / (1024 * 1024):.2f} MB)")
    print(f"[Teacher Embedder] Saved dictionary to {output_dict_path}")
    return np.load(output_npy_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract BGE-M3 Teacher Embeddings with Resumable Disk Persistence")
    parser.add_argument("--data", default="data/raw/conceptnet_clean.json", help="Path to cleaned dataset")
    parser.add_argument("--out-npy", default="cache/bge_m3_concept_targets.npy", help="Output .npy path")
    parser.add_argument("--out-dict", default="cache/concepts_dict.json", help="Output concepts JSON path")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size for GPU inference")
    parser.add_argument("--from-hf", default=None, help="Optional Hugging Face repo ID to download pre-encoded targets from")
    parser.add_argument("--force-recompute", action="store_true", help="Force recomputation even if HF repo is given")
    parser.add_argument("--max-concepts", type=int, default=None, help="Optional concept limit")
    args = parser.parse_args()
    
    out_npy = Path(args.out_npy)
    out_dict = Path(args.out_dict)
    
    # 1. Option to download pre-encoded concept vectors from Hugging Face
    if args.from_hf and not args.force_recompute:
        if try_download_from_hf(args.from_hf, out_npy, out_dict):
            sys.exit(0)
            
    # 2. Local GPU encoding
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Dataset {data_path} not found.")
        sys.exit(1)
        
    with open(data_path, "r", encoding="utf-8") as f:
        triples = json.load(f)
        
    unique_concepts = set()
    for t in triples:
        if "head" in t and "tail" in t:
            unique_concepts.add(t["head"])
            unique_concepts.add(t["tail"])
            
    concept_list = sorted(list(unique_concepts))
    if args.max_concepts:
        concept_list = concept_list[:args.max_concepts]
        
    extract_bge_teacher_embeddings(
        concepts=concept_list,
        output_npy_path=out_npy,
        output_dict_path=out_dict,
        batch_size=args.batch_size
    )
