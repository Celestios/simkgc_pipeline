#!/usr/bin/env python3
"""
Stage 1 - Sub-Module B: Relational Core Training (Layers 9–12).
Learns pure vector manifold graph reasoning (v_h* + r -> v_t*) directly from BGE-M3 teacher targets.
"""

import os
import sys
import json
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.modular_encoder import RelationalCore
from src.data.dataset import VectorTripleDataset, vector_triple_collate_fn
from src.data.relations import CANONICAL_RELATIONS

def train_core(config_path: str = "config/training_config.yaml",
               batch_size: Optional[int] = None,
               epochs: Optional[int] = None,
               lr: Optional[float] = None,
               temperature: Optional[float] = None):
    """
    Trains RelationalCore (Layers 9–12) on pure 256-d concept vectors with link prediction InfoNCE loss.
    """
    config_p = Path(config_path)
    config = {}
    if config_p.exists():
        with open(config_p, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[STAGE 1B: Relational Core] Using device: {device}")

    # Hyperparameters from config with overrides
    backbone_name = config.get("model", {}).get("backbone_name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    output_dim = config.get("model", {}).get("output_dimensions", [256])[0]
    batch_size = batch_size or config.get("training", {}).get("batch_size", 512)
    epochs = epochs or config.get("training", {}).get("num_train_epochs", 10)
    lr = lr or float(config.get("training", {}).get("learning_rate", 2e-4))
    temperature = temperature or float(config.get("training", {}).get("temperature", 0.05))

    # Paths
    teacher_cache = Path(config.get("distillation", {}).get("teacher_cache_path", "cache/bge_m3_concept_targets.npy"))
    teacher_dict = Path(config.get("distillation", {}).get("teacher_dict_path", "cache/concepts_dict.json"))
    data_files = config.get("data", {}).get("train_files", ["data/raw/conceptnet_clean.json"])
    output_dir = Path("checkpoints/stage1_core")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not teacher_cache.exists() or not teacher_dict.exists():
        print(f"[ERROR] Teacher targets not found at {teacher_cache}. Run src/data/teacher_embedder.py first.")
        return

    print(f"Loading concept targets from {teacher_cache}...")
    teacher_npy = np.load(teacher_cache)
    teacher_embeddings = torch.from_numpy(teacher_npy).float()

    with open(teacher_dict, "r", encoding="utf-8") as f:
        concepts = json.load(f)
    concept_to_idx = {c: i for i, c in enumerate(concepts)}

    # Build canonical relation map
    relation_keys = list(CANONICAL_RELATIONS.keys())
    relation_to_idx = {r: i for i, r in enumerate(relation_keys)}

    # Load triples
    all_triples = []
    for df in data_files:
        p = Path(df)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_triples.extend(data)
                elif isinstance(data, dict) and "triples" in data:
                    all_triples.extend(data["triples"])

    print(f"Loaded {len(all_triples):,} raw triples. Filtering against {len(concepts):,} known concepts...")
    dataset = VectorTripleDataset(all_triples, teacher_embeddings, concept_to_idx, relation_to_idx)
    print(f"Valid vector triples for Relational Core: {len(dataset):,}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=vector_triple_collate_fn,
        drop_last=True
    )

    core = RelationalCore(
        backbone_name=backbone_name,
        input_dim=output_dim,
        output_dim=output_dim,
        num_relations=len(relation_keys),
        split_layer=8,
        total_layers=12
    )
    core.to(device)

    optimizer = torch.optim.AdamW(core.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(dataloader) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    cross_entropy = nn.CrossEntropyLoss()

    core.train()
    print(f"Starting Relational Core training for {epochs} epochs ({len(dataloader)} steps/epoch)...")

    for epoch in range(epochs):
        total_loss = 0.0
        for step, (h_vecs, r_ids, t_vecs) in enumerate(dataloader):
            h_vecs = h_vecs.to(device)
            r_ids = r_ids.to(device)
            t_vecs = t_vecs.to(device)

            optimizer.zero_grad()
            pred_tail_vecs = core(h_vecs, r_ids)

            # 1. InfoNCE contrastive matching across in-batch target vectors
            sim_matrix = torch.matmul(pred_tail_vecs, t_vecs.T) / temperature
            labels = torch.arange(pred_tail_vecs.size(0), device=device, dtype=torch.long)
            loss_infonce = cross_entropy(sim_matrix, labels)

            # 2. Direct alignment distance regularization
            loss_align = torch.mean(1.0 - torch.sum(pred_tail_vecs * t_vecs, dim=-1))

            loss = loss_infonce + (0.3 * loss_align)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if (step + 1) % 50 == 0:
                avg_loss = total_loss / (step + 1)
                print(f"Epoch [{epoch+1}/{epochs}] Step [{step+1}/{len(dataloader)}] - Loss: {avg_loss:.4f} (InfoNCE: {loss_infonce.item():.4f}, Align: {loss_align.item():.4f})")

        checkpoint_path = output_dir / f"core_epoch_{epoch+1}.pt"
        torch.save(core.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    final_path = output_dir / "relational_core_l9_12_final.pt"
    torch.save(core.state_dict(), final_path)
    print(f"\n[DONE] Relational Core training complete. Saved to: {final_path}")

if __name__ == "__main__":
    train_core()
