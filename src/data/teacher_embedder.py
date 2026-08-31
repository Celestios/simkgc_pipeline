#!/usr/bin/env python3
"""
BGE-M3 Teacher Embedder with Memory-Mapped Streaming & Hugging Face Hub Sync.
Generates 256-d normalized teacher concept target representations.
Supports downloading precomputed encodings and immediately uploading recomputed encodings to HF Hub.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.checkpoint import download_from_hf, upload_file_to_hf

class BGEConceptProjector(nn.Module):
    """Deterministic linear projection head from BGE-M3 1024d space to 256d target space."""
    def __init__(self, input_dim: int = 1024, output_dim: int = 256):
        super().__init__()
        torch.manual_seed(42)
        self.proj = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)
        for param in self.proj.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.proj(x)
        return torch.nn.functional.normalize(projected, p=2, dim=-1)

def try_download_from_hf(repo_id: str, out_npy: Path, out_dict: Path, token: Optional[str] = None) -> bool:
    """Attempts to download precomputed target encodings from Hugging Face Hub."""
    print(f"\n[HF Sync] Checking for precomputed BGE-M3 targets in: https://huggingface.co/{repo_id}...")
    npy_res = download_from_hf("bge_m3_concept_targets.npy", out_npy.parent, repo_id=repo_id, token=token)
    dict_res = download_from_hf("concepts_dict.json", out_dict.parent, repo_id=repo_id, token=token)
    
    if npy_res and dict_res and npy_res.exists() and dict_res.exists():
        if npy_res != out_npy:
            import shutil
            shutil.copy2(str(npy_res), str(out_npy))
        if dict_res != out_dict:
            import shutil
            shutil.copy2(str(dict_res), str(out_dict))
        print("✓ Successfully loaded precomputed teacher targets from Hugging Face!")
        return True
    print("[HF Sync] Precomputed targets not found on HF. Will compute fresh on GPU.")
    return False

def extract_bge_teacher_embeddings(
    concepts: List[str],
    output_npy_path: str = "cache/bge_m3_concept_targets.npy",
    output_dict_path: str = "cache/concepts_dict.json",
    teacher_model_name: str = "BAAI/bge-m3",
    output_dim: int = 256,
    batch_size: int = 2048,
    push_to_hf: bool = False,
    hf_repo: Optional[str] = None,
    hf_token: Optional[str] = None
) -> np.ndarray:
    """Streams concept encodings to disk using memory mapping, then optionally uploads to HF Hub."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_npy_path = Path(output_npy_path)
    output_dict_path = Path(output_dict_path)
    output_npy_path.parent.mkdir(parents=True, exist_ok=True)
    output_dict_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_concepts = len(concepts)
    temp_dat_path = output_npy_path.with_suffix(".mmap.dat")
    progress_file = output_npy_path.parent / "teacher_embed_progress.json"
    
    start_idx = 0
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
                mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                sum_emb = torch.sum(outputs.last_hidden_state * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled = sum_emb / sum_mask
                proj_emb = projector(pooled)
                mmap_matrix[i:i + len(chunk)] = proj_emb.float().cpu().numpy()
                
            pbar.update(len(chunk))
            
            current_done = i + len(chunk)
            if current_done % 10000 == 0 or current_done == total_concepts:
                mmap_matrix.flush()
                with open(progress_file, "w", encoding="utf-8") as f:
                    json.dump({"completed_count": current_done, "total": total_concepts}, f)
                    
        pbar.close()

    mmap_matrix.flush()
    print(f"\n[Teacher Embedder] Finalizing persistent array to: {output_npy_path}...")
    np.save(output_npy_path, np.array(mmap_matrix))
    
    with open(output_dict_path, "w", encoding="utf-8") as f:
        json.dump(concepts, f, ensure_ascii=False, indent=2)
        
    if progress_file.exists():
        progress_file.unlink()
    if temp_dat_path.exists():
        temp_dat_path.unlink()
        
    print(f"[Teacher Embedder] Saved target array to {output_npy_path} ({output_npy_path.stat().st_size / (1024 * 1024):.2f} MB)")
    print(f"[Teacher Embedder] Saved dictionary to {output_dict_path}")

    # Immediately upload recomputed targets to Hugging Face
    target_repo = hf_repo or "Celestios/Persian-simkgc-256d"
    if push_to_hf or os.environ.get("HF_TOKEN"):
        print(f"\n[HF Sync] Uploading newly encoded teacher targets to: https://huggingface.co/{target_repo}...")
        upload_file_to_hf(output_npy_path, path_in_repo="bge_m3_concept_targets.npy", repo_id=target_repo, token=hf_token,
                          commit_message="Upload precomputed BGE-M3 concept targets (256d)")
        upload_file_to_hf(output_dict_path, path_in_repo="concepts_dict.json", repo_id=target_repo, token=hf_token,
                          commit_message="Upload concepts dictionary")

    return np.load(output_npy_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract BGE-M3 Teacher Embeddings with Resumable Disk Persistence")
    parser.add_argument("--data", default="data/raw/conceptnet_clean.json", help="Path to cleaned dataset")
    parser.add_argument("--out-npy", default="cache/bge_m3_concept_targets.npy", help="Output .npy path")
    parser.add_argument("--out-dict", default="cache/concepts_dict.json", help="Output concepts JSON path")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size for GPU inference")
    parser.add_argument("--from-hf", default=None, help="Optional Hugging Face repo ID to download pre-encoded targets from")
    parser.add_argument("--push-to-hf", action="store_true", help="Upload recomputed targets to Hugging Face Hub")
    parser.add_argument("--hf-repo", default=None, help="Target Hugging Face repo ID")
    parser.add_argument("--force-recompute", action="store_true", help="Force recomputation even if HF repo is given")
    parser.add_argument("--max-concepts", type=int, default=None, help="Optional concept limit")
    args = parser.parse_args()
    
    out_npy = Path(args.out_npy)
    out_dict = Path(args.out_dict)
    
    if args.from_hf and not args.force_recompute:
        if try_download_from_hf(args.from_hf, out_npy, out_dict):
            sys.exit(0)
            
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
        batch_size=args.batch_size,
        push_to_hf=args.push_to_hf,
        hf_repo=args.hf_repo
    )
