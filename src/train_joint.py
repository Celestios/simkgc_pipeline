#!/usr/bin/env python3
"""
Stage 2 - Assembled Model Joint Fine-Tuning & Production Calibration.
Loads Stage 1A (TextEmbedder) and Stage 1B (RelationalCore) checkpoints,
assembles them into AssembledBiEncoder, and performs low-LR end-to-end training
with dedicated holdout validation (MRR, Hits@1, Hits@3, Hits@10, and Validation Loss).
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
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.modular_encoder import TextEmbedder, RelationalCore, AssembledBiEncoder
from src.model.loss import SimKGCMatryoshkaLoss, SimKGCDistillationLoss
from src.data.dataset import SimKGCDataset, SimKGCCollator

def evaluate_model(model: nn.Module, val_loader: DataLoader, device: torch.device, output_dim: int, temperature: float = 0.05) -> Dict[str, float]:
    """
    Evaluates AssembledBiEncoder on holdout graph triples:
      1. End-to-End MRR (Mean Reciprocal Rank)
      2. In-Batch Hits@1, Hits@3, Hits@10
      3. Validation Loss
    """
    model.eval()
    criterion = SimKGCMatryoshkaLoss(temperature=temperature, primary_dim=output_dim)
    total_val_loss = 0.0
    total_rr = 0.0
    hits_1 = 0
    hits_3 = 0
    hits_10 = 0
    total_samples = 0
    val_steps = 0

    with torch.no_grad():
        for batch in val_loader:
            hr_ids = batch["hr_input_ids"].to(device)
            hr_mask = batch["hr_attention_mask"].to(device)
            tail_ids = batch["tail_input_ids"].to(device)
            tail_mask = batch["tail_attention_mask"].to(device)
            batch_size = hr_ids.size(0)

            hr_vecs, tail_vecs = model(hr_ids, hr_mask, tail_ids, tail_mask)
            loss = criterion(hr_vecs, tail_vecs)
            total_val_loss += loss.item()
            val_steps += 1

            # In-batch similarity ranking
            sim_matrix = torch.matmul(hr_vecs, tail_vecs.T) / temperature
            labels = torch.arange(batch_size, device=device)

            ranks = torch.argsort(sim_matrix, dim=-1, descending=True)
            target_ranks = (ranks == labels.unsqueeze(1)).nonzero()[:, 1] + 1  # 1-indexed

            total_rr += torch.sum(1.0 / target_ranks.float()).item()
            hits_1 += torch.sum(target_ranks == 1).item()
            hits_3 += torch.sum(target_ranks <= 3).item()
            hits_10 += torch.sum(target_ranks <= 10).item()
            total_samples += batch_size

    return {
        "val_loss": total_val_loss / max(val_steps, 1),
        "mrr": total_rr / max(total_samples, 1),
        "hits@1": hits_1 / max(total_samples, 1),
        "hits@3": hits_3 / max(total_samples, 1),
        "hits@10": hits_10 / max(total_samples, 1)
    }

def train_joint(config_path: str = "config/training_config.yaml",
                embedder_ckpt: str = "checkpoints/stage1_embedder/embedder_l1_8_final.pt",
                core_ckpt: str = "checkpoints/stage1_core/relational_core_l9_12_final.pt"):
    """
    Assembles TextEmbedder and RelationalCore, and calibrates full model end-to-end.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[STAGE 2: Joint Assembly & Fine-Tuning] Using device: {device}")

    backbone_name = config["model"]["backbone_name"]
    output_dim = config["model"]["output_dimensions"][0]
    batch_size = config["training"]["batch_size"]
    epochs = config["training"].get("num_train_epochs", 15)
    lr = float(config["training"].get("learning_rate", 1.5e-5))
    temperature = float(config["training"].get("temperature", 0.05))
    max_seq_length = config["model"].get("max_seq_length", 64)

    # 1. Initialize Sub-Networks
    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    embedder = TextEmbedder(backbone_name=backbone_name, output_dim=output_dim, split_layer=8)
    core = RelationalCore(backbone_name=backbone_name, input_dim=output_dim, output_dim=output_dim, split_layer=8, total_layers=12)

    # 2. Load Checkpoints if available
    embedder_path = Path(embedder_ckpt)
    if embedder_path.exists():
        print(f"Loading Stage 1A TextEmbedder checkpoint: {embedder_path}")
        embedder.load_state_dict(torch.load(embedder_path, map_location="cpu"))
    else:
        print(f"[INFO] No Stage 1A checkpoint found at {embedder_path}, initializing from backbone base.")

    core_path = Path(core_ckpt)
    if core_path.exists():
        print(f"Loading Stage 1B RelationalCore checkpoint: {core_path}")
        core.load_state_dict(torch.load(core_path, map_location="cpu"))
    else:
        print(f"[INFO] No Stage 1B checkpoint found at {core_path}, initializing from backbone base.")

    # 3. Assemble Unified Model
    model = AssembledBiEncoder(embedder, core)
    model.to(device)

    # 4. Teacher Targets for Distillation
    distill_cfg = config.get("distillation", {})
    distill_enabled = distill_cfg.get("enabled", False)
    teacher_embeddings = None
    concept_to_idx = None

    if distill_enabled:
        teacher_cache_path = Path(distill_cfg.get("teacher_cache_path", "cache/bge_m3_concept_targets.npy"))
        teacher_dict_path = Path(distill_cfg.get("teacher_dict_path", "cache/concepts_dict.json"))

        if teacher_cache_path.exists() and teacher_dict_path.exists():
            print(f"[DISTILLATION] Loading teacher targets from {teacher_cache_path}")
            teacher_npy = np.load(teacher_cache_path)
            teacher_embeddings = torch.from_numpy(teacher_npy).float()
            with open(teacher_dict_path, "r", encoding="utf-8") as f:
                teacher_concepts = json.load(f)
            concept_to_idx = {c: i for i, c in enumerate(teacher_concepts)}
            criterion = SimKGCDistillationLoss(temperature=temperature, alpha=float(distill_cfg.get("alpha", 0.5)))
        else:
            criterion = SimKGCMatryoshkaLoss(temperature=temperature, primary_dim=output_dim)
    else:
        criterion = SimKGCMatryoshkaLoss(temperature=temperature, primary_dim=output_dim)

    # 5. Dataset and DataLoaders
    data_files = config["data"]["train_files"]
    full_dataset = SimKGCDataset(data_files)
    train_size = int(config["data"]["train_split"] * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config["data"].get("seed", 42))
    )
    print(f"Dataset split: {train_size:,} train triples | {val_size:,} validation triples ({config['data']['eval_split']*100:.0f}%)")

    collator = SimKGCCollator(tokenizer=tokenizer, max_seq_length=max_seq_length, concept_to_idx=concept_to_idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    # 6. Optimizer with Discriminative Layer Rates
    optimizer_grouped_parameters = [
        {"params": model.text_embedder.parameters(), "lr": lr * 0.5},
        {"params": model.relational_core.parameters(), "lr": lr}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    output_dir = Path(config["export"].get("checkpoint_dir", "checkpoints/simkgc_fa_en"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting Assembled Model Joint Training ({epochs} epochs, {len(train_loader)} steps/epoch)...")
    best_mrr = -1.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            hr_ids = batch["hr_input_ids"].to(device)
            hr_mask = batch["hr_attention_mask"].to(device)
            tail_ids = batch["tail_input_ids"].to(device)
            tail_mask = batch["tail_attention_mask"].to(device)

            optimizer.zero_grad()
            hr_vecs, tail_vecs = model(hr_ids, hr_mask, tail_ids, tail_mask)

            if distill_enabled and "tail_indices" in batch and teacher_embeddings is not None:
                tail_indices = batch["tail_indices"]
                valid_mask = tail_indices >= 0
                if valid_mask.any():
                    t_targets = teacher_embeddings[tail_indices].to(device)
                    loss = criterion(hr_vecs, tail_vecs, t_targets)
                else:
                    loss = SimKGCMatryoshkaLoss(temperature=temperature, primary_dim=output_dim)(hr_vecs, tail_vecs)
            else:
                loss = criterion(hr_vecs, tail_vecs)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if (step + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Step [{step+1}/{len(train_loader)}] - Loss: {total_loss / (step + 1):.4f}")

        # Dedicated Validation Evaluation
        metrics = evaluate_model(model, val_loader, device, output_dim, temperature)
        print(f"\n---> [Stage 2 End-to-End Validation - Epoch {epoch+1}/{epochs}]")
        print(f"     • Val Loss:        {metrics['val_loss']:.4f}")
        print(f"     • End-to-End MRR:  {metrics['mrr']:.4f}")
        print(f"     • Hits@1:          {metrics['hits@1']*100:.2f}%")
        print(f"     • Hits@3:          {metrics['hits@3']*100:.2f}%")
        print(f"     • Hits@10:         {metrics['hits@10']*100:.2f}%\n")

        # Save epoch checkpoint
        ckpt_file = output_dir / f"assembled_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), ckpt_file)
        
        if metrics["mrr"] > best_mrr:
            best_mrr = metrics["mrr"]
            torch.save(model.state_dict(), output_dir / "simkgc_model.pt")
            print(f"  ★ New best model checkpoint saved (MRR: {best_mrr:.4f})")

    # Save tokenizer files to checkpoint dir
    tokenizer.save_pretrained(str(output_dir))
    print(f"\n[DONE] Joint Assembly training complete! Final model saved to: {output_dir / 'simkgc_model.pt'}")

if __name__ == "__main__":
    train_joint()
