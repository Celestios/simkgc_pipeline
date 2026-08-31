#!/usr/bin/env python3
"""
Stage 1 - Sub-Module B: Relational Core Training & Dedicated Validation (Layers 9–12).
Learns pure vector manifold graph reasoning (v_h* + r -> v_t*) directly from BGE-M3 teacher targets.
Supports auto-resuming from local checkpoints and syncing to/from Hugging Face Hub.
"""

import os
import sys
import json
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader, random_split
from transformers import get_cosine_schedule_with_warmup

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.modular_encoder import RelationalCore
from src.data.dataset import VectorTripleDataset, vector_triple_collate_fn
from src.data.relations import CANONICAL_RELATIONS
from src.utils.checkpoint import find_latest_local_checkpoint, download_from_hf, upload_file_to_hf

def evaluate_core(core: nn.Module, val_loader: DataLoader, device: torch.device, temperature: float = 0.05) -> Dict[str, float]:
    """Evaluates RelationalCore on holdout vector triples."""
    core.eval()
    cross_entropy = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_cosine_sim = 0.0
    total_rr = 0.0
    hits_1 = 0
    hits_3 = 0
    hits_10 = 0
    total_samples = 0
    val_steps = 0

    with torch.no_grad():
        for h_vecs, r_ids, t_vecs in val_loader:
            h_vecs = h_vecs.to(device)
            r_ids = r_ids.to(device)
            t_vecs = t_vecs.to(device)
            batch_size = h_vecs.size(0)

            pred_t_vecs = core(h_vecs, r_ids)

            sim_matrix = torch.matmul(pred_t_vecs, t_vecs.T) / temperature
            labels = torch.arange(batch_size, device=device)

            loss_infonce = cross_entropy(sim_matrix, labels)
            loss_align = torch.mean(1.0 - torch.sum(pred_t_vecs * t_vecs, dim=-1))
            total_loss += (loss_infonce + 0.3 * loss_align).item()
            val_steps += 1

            cos_sims = torch.sum(pred_t_vecs * t_vecs, dim=-1)
            total_cosine_sim += cos_sims.sum().item()

            ranks = torch.argsort(sim_matrix, dim=-1, descending=True)
            target_ranks = (ranks == labels.unsqueeze(1)).nonzero()[:, 1] + 1

            total_rr += torch.sum(1.0 / target_ranks.float()).item()
            hits_1 += torch.sum(target_ranks == 1).item()
            hits_3 += torch.sum(target_ranks <= 3).item()
            hits_10 += torch.sum(target_ranks <= 10).item()
            total_samples += batch_size

    return {
        "val_loss": total_loss / max(val_steps, 1),
        "mean_cosine_sim": total_cosine_sim / max(total_samples, 1),
        "mrr": total_rr / max(total_samples, 1),
        "hits@1": hits_1 / max(total_samples, 1),
        "hits@3": hits_3 / max(total_samples, 1),
        "hits@10": hits_10 / max(total_samples, 1),
    }

def train_core(config_path: str = "config/training_config.yaml",
               batch_size: Optional[int] = None,
               epochs: Optional[int] = None,
               lr: Optional[float] = None,
               temperature: Optional[float] = None,
               val_split: float = 0.15,
               resume: bool = True,
               from_hf: Optional[str] = None,
               push_to_hf: bool = False,
               hf_repo: Optional[str] = None,
               hf_token: Optional[str] = None):
    """
    Trains RelationalCore (Layers 9–12) with holdout validation, local/HF resume, and HF upload.
    """
    config_p = Path(config_path)
    config = {}
    if config_p.exists():
        with open(config_p, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[STAGE 1B: Relational Core] Using device: {device}")

    backbone_name = config.get("model", {}).get("backbone_name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    output_dim = config.get("model", {}).get("output_dimensions", [256])[0]
    batch_size = batch_size or config.get("training", {}).get("batch_size", 512)
    epochs = epochs or config.get("training", {}).get("num_train_epochs", 10)
    lr = lr or float(config.get("training", {}).get("learning_rate", 2e-4))
    temperature = temperature or float(config.get("training", {}).get("temperature", 0.05))
    seed = config.get("data", {}).get("seed", 42)

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

    relation_keys = list(CANONICAL_RELATIONS.keys())
    relation_to_idx = {r: i for i, r in enumerate(relation_keys)}

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

    full_dataset = VectorTripleDataset(all_triples, teacher_embeddings, concept_to_idx, relation_to_idx)
    print(f"Valid vector triples: {len(full_dataset):,}")

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    print(f"Dataset split: {train_size:,} train triples | {val_size:,} validation triples ({val_split*100:.0f}%)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=vector_triple_collate_fn,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=vector_triple_collate_fn
    )

    core = RelationalCore(
        backbone_name=backbone_name,
        input_dim=output_dim,
        output_dim=output_dim,
        num_relations=len(relation_keys),
        split_layer=8,
        total_layers=12
    )

    # 1. Resume Check: Local First, then Hugging Face
    start_epoch = 0
    if resume:
        latest_ckpt, last_epoch = find_latest_local_checkpoint(output_dir, prefix="core_epoch_")
        if latest_ckpt and latest_ckpt.exists():
            print(f"[RESUME: Local] Found checkpoint: {latest_ckpt} (Epoch {last_epoch})")
            core.load_state_dict(torch.load(latest_ckpt, map_location="cpu"))
            start_epoch = last_epoch
        elif from_hf:
            hf_ckpt = download_from_hf("relational_core_l9_12_final.pt", output_dir, repo_id=from_hf, token=hf_token)
            if hf_ckpt and hf_ckpt.exists():
                print(f"[RESUME: Hugging Face] Loaded from {from_hf}: {hf_ckpt}")
                core.load_state_dict(torch.load(hf_ckpt, map_location="cpu"))
                start_epoch = 0

    core.to(device)

    optimizer = torch.optim.AdamW(core.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    cross_entropy = nn.CrossEntropyLoss()

    if start_epoch > 0:
        steps_done = start_epoch * len(train_loader)
        for _ in range(steps_done):
            scheduler.step()
        print(f"[RESUME] Advanced scheduler by {steps_done} steps (Epoch {start_epoch}/{epochs}).")

    print(f"\nStarting Relational Core training from Epoch {start_epoch+1} to {epochs} ({len(train_loader)} steps/epoch)...")
    best_mrr = -1.0

    for epoch in range(start_epoch, epochs):
        core.train()
        total_loss = 0.0
        for step, (h_vecs, r_ids, t_vecs) in enumerate(train_loader):
            h_vecs = h_vecs.to(device)
            r_ids = r_ids.to(device)
            t_vecs = t_vecs.to(device)

            optimizer.zero_grad()
            pred_tail_vecs = core(h_vecs, r_ids)

            sim_matrix = torch.matmul(pred_tail_vecs, t_vecs.T) / temperature
            labels = torch.arange(pred_tail_vecs.size(0), device=device, dtype=torch.long)
            loss_infonce = cross_entropy(sim_matrix, labels)

            loss_align = torch.mean(1.0 - torch.sum(pred_tail_vecs * t_vecs, dim=-1))
            loss = loss_infonce + (0.3 * loss_align)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if (step + 1) % 50 == 0:
                avg_loss = total_loss / (step + 1)
                print(f"Epoch [{epoch+1}/{epochs}] Step [{step+1}/{len(train_loader)}] - Loss: {avg_loss:.4f}")

        # Dedicated Holdout Validation
        metrics = evaluate_core(core, val_loader, device, temperature)
        print(f"\n---> [Stage 1B Validation - Epoch {epoch+1}/{epochs}]")
        print(f"     • Val Loss:        {metrics['val_loss']:.4f}")
        print(f"     • Vector MRR:      {metrics['mrr']:.4f}")
        print(f"     • Vector Hits@1:   {metrics['hits@1']*100:.2f}%")
        print(f"     • Vector Hits@3:   {metrics['hits@3']*100:.2f}%")
        print(f"     • Vector Hits@10:  {metrics['hits@10']*100:.2f}%\n")

        checkpoint_path = output_dir / f"core_epoch_{epoch+1}.pt"
        torch.save(core.state_dict(), checkpoint_path)

        if metrics["mrr"] > best_mrr:
            best_mrr = metrics["mrr"]
            best_path = output_dir / "relational_core_l9_12_final.pt"
            torch.save(core.state_dict(), best_path)
            print(f"  ★ New best RelationalCore saved to {best_path} (MRR: {best_mrr:.4f})")

    final_path = output_dir / "relational_core_l9_12_final.pt"
    if not final_path.exists() and (output_dir / f"core_epoch_{epochs}.pt").exists():
        torch.save(core.state_dict(), final_path)

    print(f"\n[DONE] Stage 1B complete! Best RelationalCore saved at: {final_path}")

    # 2. Upload to Hugging Face if enabled
    target_hf_repo = hf_repo or from_hf or config.get("distillation", {}).get("hf_repo")
    if (push_to_hf or target_hf_repo) and final_path.exists():
        repo = target_hf_repo or "Celestios/Persian-simkgc-256d"
        upload_file_to_hf(final_path, path_in_repo="relational_core_l9_12_final.pt", repo_id=repo, token=hf_token)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train RelationalCore (Layers 9-12)")
    parser.add_argument("--epochs", type=int, default=None, help="Total training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from existing checkpoint")
    parser.add_argument("--from-hf", default=None, help="Hugging Face repo ID to resume from")
    parser.add_argument("--push-to-hf", action="store_true", help="Push trained model to Hugging Face Hub")
    parser.add_argument("--hf-repo", default=None, help="Hugging Face repo ID to push to")
    parser.add_argument("--hf-token", default=None, help="Hugging Face API token")
    args = parser.parse_args()

    train_core(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume=not args.no_resume,
        from_hf=args.from_hf,
        push_to_hf=args.push_to_hf,
        hf_repo=args.hf_repo,
        hf_token=args.hf_token
    )
