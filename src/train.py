import os
import sys
import json
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# Add current folder to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model.biencoder import SimKGCBiEncoder
from src.model.loss import SimKGCMatryoshkaLoss, SimKGCDistillationLoss
from src.model.vocab_pruner import prune_vocab_and_weights
from src.data.dataset import SimKGCDataset, SimKGCCollator

def train():
    config_path = Path("config/training_config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} (Count: {torch.cuda.device_count()})")

    backbone_name = config["model"]["backbone_name"]
    output_dim = config["model"]["output_dimensions"][0]
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["num_train_epochs"]
    lr = float(config["training"]["learning_rate"])
    temperature = float(config["training"]["temperature"])
    max_seq_length = config["model"].get("max_seq_length", 64)

    # 1. Initialize Tokenizer & Model
    print(f"Loading backbone: {backbone_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    except Exception:
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(backbone_name)
    model = SimKGCBiEncoder(backbone_name=backbone_name, output_dim=output_dim)

    # 2. Prune Vocabulary if configured
    if config["model"].get("prune_vocab_to_fa_en", False):
        prune_vocab_and_weights(tokenizer, model)

    model.to(device)

    # Multi-GPU DataParallel if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = nn.DataParallel(model)

    # 3. Check for Distillation Mode
    distill_cfg = config.get("distillation", {})
    distill_enabled = distill_cfg.get("enabled", False)
    teacher_embeddings = None
    concept_to_idx = None
    
    if distill_enabled:
        teacher_cache_path = Path(distill_cfg.get("teacher_cache_path", "cache/bge_m3_concept_targets.npy"))
        teacher_dict_path = Path(distill_cfg.get("teacher_dict_path", "cache/concepts_dict.json"))
        
        if teacher_cache_path.exists() and teacher_dict_path.exists():
            print(f"\n[DISTILLATION MODE] Loading teacher targets from: {teacher_cache_path}")
            teacher_npy = np.load(teacher_cache_path)
            teacher_embeddings = torch.from_numpy(teacher_npy).float()
            
            with open(teacher_dict_path, "r", encoding="utf-8") as f:
                teacher_concepts = json.load(f)
            concept_to_idx = {c: i for i, c in enumerate(teacher_concepts)}
            print(f"[DISTILLATION MODE] Ready with {len(teacher_concepts):,} concept targets.")
            criterion = SimKGCDistillationLoss(
                temperature=temperature,
                alpha=float(distill_cfg.get("alpha", 0.7)),
                aux_dim=128
            )
        else:
            print(f"[WARNING] Distillation enabled but cache not found ({teacher_cache_path}). Falling back to standard InfoNCE.")
            distill_enabled = False
            criterion = SimKGCMatryoshkaLoss(temperature=temperature, primary_dim=output_dim, aux_dim=128)
    else:
        criterion = SimKGCMatryoshkaLoss(temperature=temperature, primary_dim=output_dim, aux_dim=128)

    # 4. Prepare Dataset & DataLoader
    data_files = config["data"]["train_files"]
    full_dataset = SimKGCDataset(data_files)
    train_size = int(config["data"]["train_split"] * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config["data"].get("seed", 42))
    )

    collator = SimKGCCollator(tokenizer, max_seq_length=max_seq_length, concept_to_idx=concept_to_idx)
    actual_batch_size = max(2, min(batch_size, train_size))
    num_workers = min(4, os.cpu_count() or 1)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=actual_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=actual_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0)
    ) if val_size > 0 else None

    # 5. Optimizer, Loss & Scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": config["training"].get("weight_decay", 0.01)},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * config["training"].get("warmup_ratio", 0.1))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=config["training"].get("use_fp16", True) and torch.cuda.is_available())

    # 6. Checkpoint Resume
    start_epoch = 1
    best_val_mrr = 0.0
    chk_dir = Path("checkpoints/simkgc_fa_en")
    chk_file = chk_dir / "simkgc_model.pt"
    meta_file = chk_dir / "best_model_meta.json"

    if chk_file.exists():
        try:
            raw_model = model.module if hasattr(model, "module") else model
            raw_model.load_state_dict(torch.load(chk_file, map_location="cpu"))
            if meta_file.exists():
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    start_epoch = meta.get("best_epoch", 1) + 1
                    best_val_mrr = meta.get("val_mrr", 0.0)
            print(f"★ [RESUMING CHECKPOINT] Loaded trained weights from {chk_file}.")
            print(f"  Resuming from Epoch {start_epoch}/{epochs} (Previous Best Val MRR: {best_val_mrr:.4f}).")
        except Exception as e:
            print(f"Warning: Could not resume checkpoint ({e}). Starting fresh.")

    print("\n" + "=" * 60)
    mode_str = "Teacher-Distilled" if distill_enabled else "Standard SimKGC"
    print(f"Starting {mode_str} Training: Epochs {start_epoch} -> {epochs} | Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    print("=" * 60)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for step, batch in enumerate(train_loader, 1):
            hr_ids = batch["hr_input_ids"].to(device)
            hr_mask = batch["hr_attention_mask"].to(device)
            t_ids = batch["tail_input_ids"].to(device)
            t_mask = batch["tail_attention_mask"].to(device)

            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                hr_vecs, t_vecs = model(hr_ids, hr_mask, t_ids, t_mask)
                
                if distill_enabled and teacher_embeddings is not None and "tail_indices" in batch:
                    tail_idx = batch["tail_indices"]
                    teacher_targets = teacher_embeddings[tail_idx].to(device)
                    loss = criterion(hr_vecs, t_vecs, teacher_targets)
                else:
                    loss = criterion(hr_vecs, t_vecs)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            if step % config["training"].get("logging_steps", 10) == 0 or step == len(train_loader):
                print(f"Epoch [{epoch}/{epochs}] Step [{step}/{len(train_loader)}] Loss: {loss.item():.4f} LR: {scheduler.get_last_lr()[0]:.2e}")

        avg_loss = total_loss / len(train_loader)

        # Validation Evaluation (MRR, Hits@1, Hits@10)
        if val_loader is not None and len(val_loader) > 0:
            model.eval()
            reciprocal_ranks = []
            hits_1 = 0
            hits_10 = 0
            total_val = 0
            
            with torch.no_grad():
                for vbatch in val_loader:
                    v_hr_ids = vbatch["hr_input_ids"].to(device)
                    v_hr_mask = vbatch["hr_attention_mask"].to(device)
                    v_t_ids = vbatch["tail_input_ids"].to(device)
                    v_t_mask = vbatch["tail_attention_mask"].to(device)
                    
                    v_hr_vecs, v_t_vecs = model(v_hr_ids, v_hr_mask, v_t_ids, v_t_mask)
                    
                    if distill_enabled and teacher_embeddings is not None and "tail_indices" in vbatch:
                        v_teacher_targets = teacher_embeddings[vbatch["tail_indices"]].to(device)
                        sim_scores = torch.matmul(v_hr_vecs, v_teacher_targets.T)
                    else:
                        sim_scores = torch.matmul(v_hr_vecs, v_t_vecs.T)
                        
                    for i in range(v_hr_vecs.size(0)):
                        target_score = sim_scores[i, i].item()
                        rank = (sim_scores[i] >= target_score).sum().item()
                        reciprocal_ranks.append(1.0 / rank)
                        if rank == 1:
                            hits_1 += 1
                        if rank <= 10:
                            hits_10 += 1
                        total_val += 1
                        
            val_mrr = sum(reciprocal_ranks) / max(1, total_val)
            h1_pct = (hits_1 / max(1, total_val)) * 100.0
            h10_pct = (hits_10 / max(1, total_val)) * 100.0
            print(f"--> Epoch {epoch} Results: Avg Loss: {avg_loss:.4f} | Val MRR: {val_mrr:.4f} | Hits@1: {h1_pct:.1f}% | Hits@10: {h10_pct:.1f}%")

            # Check and save if this is the Best Model so far
            if val_mrr > best_val_mrr:
                best_val_mrr = val_mrr
                output_dir = Path("checkpoints/simkgc_fa_en")
                output_dir.mkdir(parents=True, exist_ok=True)
                raw_model = model.module if hasattr(model, "module") else model
                torch.save(raw_model.state_dict(), output_dir / "simkgc_model.pt")
                tokenizer.save_pretrained(output_dir)
                raw_model.config.save_pretrained(output_dir)
                
                with open(output_dir / "best_model_meta.json", "w", encoding="utf-8") as f:
                    json.dump({
                        "best_epoch": epoch,
                        "val_mrr": val_mrr,
                        "hits_at_1": h1_pct,
                        "hits_at_10": h10_pct,
                        "mode": "distilled" if distill_enabled else "standard"
                    }, f, indent=2)
                print(f"★ [NEW BEST MODEL] Saved checkpoint for Epoch {epoch} (Val MRR: {val_mrr:.4f}) to {output_dir}")
        else:
            print(f"--> Epoch {epoch} Results: Avg Loss: {avg_loss:.4f}")

        # Regular periodic checkpoint every N epochs
        save_every = config["training"].get("save_every_epochs", 5)
        if epoch % save_every == 0 or epoch == epochs:
            output_dir = Path("checkpoints/simkgc_fa_en")
            output_dir.mkdir(parents=True, exist_ok=True)
            raw_model = model.module if hasattr(model, "module") else model
            torch.save(raw_model.state_dict(), output_dir / "simkgc_model.pt")
            tokenizer.save_pretrained(output_dir)
            raw_model.config.save_pretrained(output_dir)
            print(f"[CHECKPOINT SAVED] Epoch {epoch}/{epochs} saved to {output_dir}")

if __name__ == "__main__":
    train()
