#!/usr/bin/env python3
"""
Stage 1 - Sub-Module A: Text Embedder Training (Layers 1–8).
Aligns surface text (Persian & English concepts) directly to 256-d BGE-M3 target space.
"""

import os
import sys
import json
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.modular_encoder import TextEmbedder
from src.data.dataset import ConceptDataset, concept_collate_fn
from src.model.loss import SimKGCDistillationLoss

def train_embedder(config_path: str = "config/training_config.yaml",
                   batch_size: Optional[int] = None,
                   epochs: Optional[int] = None,
                   lr: Optional[float] = None,
                   temperature: Optional[float] = None,
                   alpha: Optional[float] = None):
    """
    Trains TextEmbedder (Layers 1–8) on unique concept strings with BGE-M3 teacher targets.
    """
    config_p = Path(config_path)
    config = {}
    if config_p.exists():
        with open(config_p, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[STAGE 1A: Text Embedder] Using device: {device}")

    # Hyperparameters from config with overrides
    backbone_name = config.get("model", {}).get("backbone_name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    output_dim = config.get("model", {}).get("output_dimensions", [256])[0]
    batch_size = batch_size or config.get("training", {}).get("batch_size", 512)
    epochs = epochs or config.get("training", {}).get("num_train_epochs", 3)
    lr = lr or float(config.get("training", {}).get("learning_rate", 3e-4))
    temperature = temperature or float(config.get("training", {}).get("temperature", 0.05))
    alpha = alpha or float(config.get("distillation", {}).get("alpha", 0.7))

    # Paths
    teacher_cache = Path(config.get("distillation", {}).get("teacher_cache_path", "cache/bge_m3_concept_targets.npy"))
    teacher_dict = Path(config.get("distillation", {}).get("teacher_dict_path", "cache/concepts_dict.json"))
    output_dir = Path("checkpoints/stage1_embedder")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not teacher_cache.exists() or not teacher_dict.exists():
        print(f"[ERROR] Teacher targets not found at {teacher_cache}. Run src/data/teacher_embedder.py first.")
        return

    print(f"Loading concept targets from {teacher_cache}...")
    teacher_npy = np.load(teacher_cache)
    teacher_embeddings = torch.from_numpy(teacher_npy).float()

    with open(teacher_dict, "r", encoding="utf-8") as f:
        concepts = json.load(f)

    print(f"Loaded {len(concepts):,} concepts with {output_dim}-d target vectors.")

    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    embedder = TextEmbedder(backbone_name=backbone_name, output_dim=output_dim, split_layer=8)
    embedder.to(device)

    dataset = ConceptDataset(concepts, teacher_embeddings, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: concept_collate_fn(b, tokenizer),
        drop_last=True
    )

    optimizer = torch.optim.AdamW(embedder.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(dataloader) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    criterion = SimKGCDistillationLoss(temperature=temperature, alpha=alpha)

    embedder.train()
    print(f"Starting Text Embedder training for {epochs} epochs ({len(dataloader)} steps/epoch)...")

    for epoch in range(epochs):
        total_loss = 0.0
        for step, (input_ids, attention_mask, targets) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            student_vecs = embedder(input_ids, attention_mask)
            loss = criterion(student_vecs, None, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(embedder.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if (step + 1) % 50 == 0:
                avg_loss = total_loss / (step + 1)
                print(f"Epoch [{epoch+1}/{epochs}] Step [{step+1}/{len(dataloader)}] - Distillation Loss: {avg_loss:.4f}")

        checkpoint_path = output_dir / f"embedder_epoch_{epoch+1}.pt"
        torch.save(embedder.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    final_path = output_dir / "embedder_l1_8_final.pt"
    torch.save(embedder.state_dict(), final_path)
    print(f"\n[DONE] Text Embedder training complete. Saved to: {final_path}")

if __name__ == "__main__":
    train_embedder()
