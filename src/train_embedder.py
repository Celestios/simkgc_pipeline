#!/usr/bin/env python3
"""
Stage 1 - Sub-Module A: Text Embedder Training & Dedicated Validation (Layers 1–8).
Aligns surface text (Persian & English concepts) directly to 256-d BGE-M3 target space.
Supports auto-resuming from local checkpoints and syncing immediately to Hugging Face Hub.
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
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.modular_encoder import TextEmbedder
from src.data.dataset import ConceptDataset, concept_collate_fn
from src.model.loss import SimKGCDistillationLoss
from src.utils.checkpoint import find_latest_local_checkpoint, download_from_hf, upload_file_to_hf

def evaluate_embedder(embedder: nn.Module, val_loader: DataLoader, device: torch.device, criterion: nn.Module) -> Dict[str, float]:
    """Evaluates TextEmbedder on holdout validation concepts."""
    embedder.eval()
    total_val_loss = 0.0
    total_cosine_sim = 0.0
    top1_correct = 0
    top5_correct = 0
    total_samples = 0
    val_steps = 0

    with torch.no_grad():
        for input_ids, attention_mask, targets in val_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)
            batch_size = input_ids.size(0)

            student_vecs = embedder(input_ids, attention_mask)
            loss = criterion(student_vecs, None, targets)
            total_val_loss += loss.item()
            val_steps += 1

            cos_sims = torch.sum(student_vecs * targets, dim=-1)
            total_cosine_sim += cos_sims.sum().item()

            sim_matrix = torch.matmul(student_vecs, targets.T)
            labels = torch.arange(batch_size, device=device)

            top1_preds = torch.argmax(sim_matrix, dim=-1)
            top1_correct += (top1_preds == labels).sum().item()

            top5_preds = torch.topk(sim_matrix, k=min(5, batch_size), dim=-1).indices
            top5_correct += (top5_preds == labels.unsqueeze(1)).any(dim=-1).sum().item()

            total_samples += batch_size

    return {
        "val_loss": total_val_loss / max(val_steps, 1),
        "mean_cosine_sim": total_cosine_sim / max(total_samples, 1),
        "top1_acc": top1_correct / max(total_samples, 1),
        "top5_acc": top5_correct / max(total_samples, 1)
    }

def train_embedder(config_path: str = "config/training_config.yaml",
                   batch_size: Optional[int] = None,
                   epochs: Optional[int] = None,
                   lr: Optional[float] = None,
                   temperature: Optional[float] = None,
                   alpha: Optional[float] = None,
                   val_split: float = 0.15,
                   resume: bool = True,
                   from_hf: Optional[str] = None,
                   push_to_hf: bool = False,
                   hf_repo: Optional[str] = None,
                   hf_token: Optional[str] = None):
    """
    Trains TextEmbedder (Layers 1–8) with holdout validation, auto-resume, and instant HF upload on checkpoint.
    """
    config_p = Path(config_path)
    config = {}
    if config_p.exists():
        with open(config_p, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[STAGE 1A: Text Embedder] Using device: {device}")

    target_hf_repo = hf_repo or from_hf or config.get("distillation", {}).get("hf_repo") or "Celestios/Persian-simkgc-256d"
    backbone_name = config.get("model", {}).get("backbone_name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    output_dim = config.get("model", {}).get("output_dimensions", [256])[0]
    batch_size = batch_size or config.get("training", {}).get("batch_size", 512)
    epochs = epochs or config.get("training", {}).get("num_train_epochs", 4)
    lr = lr or float(config.get("training", {}).get("learning_rate", 3e-4))
    temperature = temperature or float(config.get("training", {}).get("temperature", 0.05))
    alpha = alpha or float(config.get("distillation", {}).get("alpha", 0.6))
    seed = config.get("data", {}).get("seed", 42)

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

    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    embedder = TextEmbedder(backbone_name=backbone_name, output_dim=output_dim, split_layer=8)

    # 1. Resume Check: Local First, then Hugging Face
    start_epoch = 0
    if resume:
        latest_ckpt, last_epoch = find_latest_local_checkpoint(output_dir, prefix="embedder_epoch_")
        if latest_ckpt and latest_ckpt.exists():
            print(f"[RESUME: Local] Found checkpoint: {latest_ckpt} (Epoch {last_epoch})")
            embedder.load_state_dict(torch.load(latest_ckpt, map_location="cpu"))
            start_epoch = last_epoch
        elif from_hf:
            hf_ckpt = download_from_hf("embedder_l1_8_final.pt", output_dir, repo_id=from_hf, token=hf_token)
            if hf_ckpt and hf_ckpt.exists():
                print(f"[RESUME: Hugging Face] Loaded from {from_hf}: {hf_ckpt}")
                embedder.load_state_dict(torch.load(hf_ckpt, map_location="cpu"))
                start_epoch = 0

    embedder.to(device)

    # Train / Validation Split
    full_dataset = ConceptDataset(concepts, teacher_embeddings, tokenizer)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    print(f"Dataset split: {train_size:,} train concepts | {val_size:,} validation concepts ({val_split*100:.0f}%)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: concept_collate_fn(b, tokenizer),
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: concept_collate_fn(b, tokenizer)
    )

    optimizer = torch.optim.AdamW(embedder.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    criterion = SimKGCDistillationLoss(temperature=temperature, alpha=alpha)

    if start_epoch > 0:
        steps_done = start_epoch * len(train_loader)
        for _ in range(steps_done):
            scheduler.step()
        print(f"[RESUME] Advanced scheduler by {steps_done} steps (Epoch {start_epoch}/{epochs}).")

    print(f"\nStarting Text Embedder training from Epoch {start_epoch+1} to {epochs} ({len(train_loader)} steps/epoch)...")
    best_cosine_sim = -1.0

    for epoch in range(start_epoch, epochs):
        embedder.train()
        total_loss = 0.0
        for step, (input_ids, attention_mask, targets) in enumerate(train_loader):
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
                print(f"Epoch [{epoch+1}/{epochs}] Step [{step+1}/{len(train_loader)}] - Distill Loss: {avg_loss:.4f}")

        # Validation
        metrics = evaluate_embedder(embedder, val_loader, device, criterion)
        print(f"\n---> [Stage 1A Validation - Epoch {epoch+1}/{epochs}]")
        print(f"     • Val Loss:           {metrics['val_loss']:.4f}")
        print(f"     • Mean Cosine Sim:    {metrics['mean_cosine_sim']:.4f} (Alignment to BGE-M3)")
        print(f"     • In-Batch Top-1 Acc: {metrics['top1_acc']*100:.2f}%")
        print(f"     • In-Batch Top-5 Acc: {metrics['top5_acc']*100:.2f}%\n")

        checkpoint_path = output_dir / f"embedder_epoch_{epoch+1}.pt"
        torch.save(embedder.state_dict(), checkpoint_path)

        if metrics["mean_cosine_sim"] > best_cosine_sim:
            best_cosine_sim = metrics["mean_cosine_sim"]
            best_path = output_dir / "embedder_l1_8_final.pt"
            torch.save(embedder.state_dict(), best_path)
            print(f"  ★ New best TextEmbedder saved to {best_path} (Cosine Sim: {best_cosine_sim:.4f})")

            # Instant upload to Hugging Face on every best checkpoint
            if push_to_hf or os.environ.get("HF_TOKEN"):
                upload_file_to_hf(best_path, path_in_repo="embedder_l1_8_final.pt", repo_id=target_hf_repo, token=hf_token,
                                  commit_message=f"Stage 1A TextEmbedder (Epoch {epoch+1}, CosSim: {best_cosine_sim:.4f})")

    final_path = output_dir / "embedder_l1_8_final.pt"
    if not final_path.exists() and (output_dir / f"embedder_epoch_{epochs}.pt").exists():
        torch.save(embedder.state_dict(), final_path)

    print(f"\n[DONE] Stage 1A complete! Best TextEmbedder saved at: {final_path}")

    if (push_to_hf or os.environ.get("HF_TOKEN")) and final_path.exists():
        upload_file_to_hf(final_path, path_in_repo="embedder_l1_8_final.pt", repo_id=target_hf_repo, token=hf_token,
                          commit_message="Stage 1A TextEmbedder Final")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train TextEmbedder (Layers 1-8)")
    parser.add_argument("--epochs", type=int, default=None, help="Total training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from existing checkpoint")
    parser.add_argument("--from-hf", default=None, help="Hugging Face repo ID to resume from")
    parser.add_argument("--push-to-hf", action="store_true", help="Push trained model to Hugging Face Hub")
    parser.add_argument("--hf-repo", default=None, help="Hugging Face repo ID to push to")
    parser.add_argument("--hf-token", default=None, help="Hugging Face API token")
    args = parser.parse_args()

    train_embedder(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume=not args.no_resume,
        from_hf=args.from_hf,
        push_to_hf=args.push_to_hf,
        hf_repo=args.hf_repo,
        hf_token=args.hf_token
    )
