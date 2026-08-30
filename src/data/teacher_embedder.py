#!/usr/bin/env python3
"""
Teacher Embedder for SimKGC Pipeline.
Uses BAAI/bge-m3 (560M parameters) to pre-encode concept terms into high-quality
256-dimensional normalized vectors for distillation and production export.
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
    Encodes unique concepts using BGE-M3 on GPU and saves the 256-d target matrix.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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
        
    output_npy_path.parent.mkdir(parents=True, exist_ok=True)
    output_dict_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[Teacher Embedder] Pre-encoding {len(concepts):,} concepts in batches of {batch_size}...")
    all_embeddings = []
    
    pbar = tqdm(total=len(concepts), desc="BGE-M3 Concept Encoding", unit="concept")
    
    for i in range(0, len(concepts), batch_size):
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
            all_embeddings.append(proj_emb.float().cpu().numpy())
            
        pbar.update(len(chunk))
        
    pbar.close()
    
    embeddings = np.vstack(all_embeddings).astype(np.float32)
    print(f"[Teacher Embedder] Finished encoding. Output shape: {embeddings.shape}")
    
    # Save target array and concept-to-index mapping
    np.save(output_npy_path, embeddings)
    with open(output_dict_path, "w", encoding="utf-8") as f:
        json.dump(concepts, f, ensure_ascii=False, indent=2)
        
    print(f"[Teacher Embedder] Saved targets to {output_npy_path} ({output_npy_path.stat().st_size / (1024 * 1024):.2f} MB)")
    print(f"[Teacher Embedder] Saved dictionary to {output_dict_path}")
    return embeddings

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract BGE-M3 Teacher Embeddings for Concepts")
    parser.add_argument("--data", default="data/raw/conceptnet_clean.json", help="Path to cleaned dataset")
    parser.add_argument("--out-npy", default="cache/bge_m3_concept_targets.npy", help="Output .npy path")
    parser.add_argument("--out-dict", default="cache/concepts_dict.json", help="Output concepts JSON path")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for GPU inference")
    parser.add_argument("--max-concepts", type=int, default=None, help="Optional concept limit")
    args = parser.parse_args()
    
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
        output_npy_path=Path(args.out_npy),
        output_dict_path=Path(args.out_dict),
        batch_size=args.batch_size
    )
