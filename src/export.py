#!/usr/bin/env python3
"""
Production ONNX & Rust Binary Exporter for Centrode.
Exports:
  1. simkgc_256d.onnx / simkgc_256d_int8.onnx (Inference Bi-Encoder for Rust runtime)
  2. concepts_256d_int8.bin (Zero-copy flat memory-mappable INT8 concept matrix of top 50k concepts)
  3. concepts_dict.json (Index to concept string dictionary for the 50k concepts)
  4. relations_ontology.json (32 canonical relations with verbalizers and inverse mappings)
  5. relations_256d_int8.bin (Exact empirical translation offset vectors r_rel for instant 0.001ms geometric navigation)
"""

import os
import sys
import re
import json
import math
import struct
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = nn = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    from transformers import AutoTokenizer, BertTokenizer
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType
except ImportError:
    AutoTokenizer = BertTokenizer = onnx = quantize_dynamic = QuantType = None

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.model.biencoder import SimKGCBiEncoder
    from src.model.modular_encoder import TextEmbedder, RelationalCore, AssembledBiEncoder
except (ImportError, ModuleNotFoundError):
    SimKGCBiEncoder = TextEmbedder = RelationalCore = AssembledBiEncoder = None

try:
    from src.data.relations import CANONICAL_RELATIONS, is_persian_text
except (ImportError, ModuleNotFoundError):
    CANONICAL_RELATIONS = {}
    is_persian_text = lambda x: False

def quantize_int8_matrix(matrix: np.ndarray) -> Tuple[np.ndarray, bytes]:
    """Quantizes float32 matrix (-1.0 to 1.0) to INT8 [-127, 127] bytes."""
    clipped = np.clip(np.round(matrix * 127.0), -127, 127).astype(np.int8)
    return clipped, clipped.tobytes()

def write_ckge_binary(file_path: Path, matrix: np.ndarray, num_items: int, dim: int, quantize_int8: bool = True) -> int:
    """Writes CKGE binary struct header and payload to disk."""
    magic = b"CKGE"
    precision = 1 if quantize_int8 else 0
    header = struct.pack("<4sIII", magic, num_items, dim, precision)
    
    if quantize_int8:
        _, payload = quantize_int8_matrix(matrix)
    else:
        payload = matrix.astype(np.float32).tobytes()
        
    with open(file_path, "wb") as f:
        f.write(header)
        f.write(payload)
        
    return file_path.stat().st_size

def is_valid_concept_string(text: str) -> bool:
    """Lexical quality filter to discard noise, URLs, long conversational text, and pure digits."""
    t = text.strip()
    if len(t) < 2 or len(t) > 45:
        return False
    if len(t.split()) > 4:
        return False
    if t.isdigit() or t.replace(".", "", 1).isdigit():
        return False
    if "http" in t or "www." in t or "[" in t or "]" in t or "{" in t or "}" in t:
        return False
    return True

def select_top_production_concepts(
    data_paths: List[str],
    total_quota: int = 50000,
    fa_quota: int = 15000,
    en_quota: int = 35000
) -> List[str]:
    """
    Selects the top-ranked concepts using the Composite Centrality & Quality Score:
      Score(c) = log(1 + deg(c)) * (1 + 0.3 * min(unique_relations, 10)) * avg_weight
    Guarantees a balanced distribution between Persian (15k) and English (35k) concepts.
    """
    print(f"\n[Concept Curator] Analyzing knowledge graph for Top {total_quota:,} central concepts...")
    
    degrees = defaultdict(int)
    unique_rels = defaultdict(set)
    weight_sums = defaultdict(float)
    
    for path_str in data_paths:
        p = Path(path_str)
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            triples = json.load(f)
            for item in triples:
                h = item.get("head", "").strip()
                t = item.get("tail", "").strip()
                r = item.get("relation", "").strip()
                w = float(item.get("weight", 1.0))
                
                if h and is_valid_concept_string(h):
                    degrees[h] += 1
                    unique_rels[h].add(r)
                    weight_sums[h] += w
                    
                if t and is_valid_concept_string(t):
                    degrees[t] += 1
                    unique_rels[t].add(r)
                    weight_sums[t] += w

    scored_fa = []
    scored_en = []
    
    for concept, deg in degrees.items():
        rel_count = min(len(unique_rels[concept]), 10)
        avg_weight = weight_sums[concept] / deg
        score = math.log1p(deg) * (1.0 + 0.3 * rel_count) * avg_weight
        
        if is_persian_text(concept):
            scored_fa.append((score, concept))
        else:
            scored_en.append((score, concept))
            
    scored_fa.sort(reverse=True)
    scored_en.sort(reverse=True)
    
    print(f"[Concept Curator] Found {len(scored_fa):,} valid Persian and {len(scored_en):,} valid English candidates.")
    
    top_fa = [c for _, c in scored_fa[:fa_quota]]
    top_en = [c for _, c in scored_en[:en_quota]]
    
    # If one language has fewer candidates than quota, backfill from the other
    if len(top_fa) < fa_quota:
        remaining = total_quota - len(top_fa)
        top_en = [c for _, c in scored_en[:remaining]]
    elif len(top_en) < en_quota:
        remaining = total_quota - len(top_en)
        top_fa = [c for _, c in scored_fa[:remaining]]
        
    final_concepts = top_fa + top_en
    print(f"[Concept Curator] Curated exactly {len(final_concepts):,} top concepts:")
    print(f"  - Persian: {len(top_fa):,} concepts")
    print(f"  - English: {len(top_en):,} concepts")
    
    return final_concepts

def export_relations_metadata(output_json_path: Path):
    """Exports canonical ontology relations to JSON metadata file."""
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(CANONICAL_RELATIONS, f, ensure_ascii=False, indent=2)
    print(f"[OK] Exported {len(CANONICAL_RELATIONS)} canonical relations ontology to: {output_json_path}")

class ONNXExportWrapper(nn.Module):
    """Wraps model for standard ONNX single-tower text encoding: (input_ids, attention_mask) -> 256-d."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "encode"):
            return self.model.encode(input_ids, attention_mask)
        elif hasattr(self.model, "text_embedder"):
            return self.model.encode(input_ids, attention_mask)
        else:
            return self.model(input_ids, attention_mask)

def export_to_onnx(model: nn.Module, tokenizer, output_onnx_path: Path, max_length: int = 64):
    """Exports model to standard ONNX graph with dynamic axes."""
    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    
    wrapper = ONNXExportWrapper(model)
    wrapper.eval()
    
    dummy_text = "دانشگاه تهران"
    dummy_inputs = tokenizer(
        dummy_text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    input_ids = dummy_inputs["input_ids"]
    attention_mask = dummy_inputs["attention_mask"]
    
    print(f"\n[ONNX Exporter] Exporting PyTorch model to ONNX: {output_onnx_path}...")
    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask),
        str(output_onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["embedding"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "embedding": {0: "batch_size"}
        },
        opset_version=17,
        do_constant_folding=True
    )
    print(f"[OK] ONNX model exported ({output_onnx_path.stat().st_size / 1024 / 1024:.2f} MB)")

def quantize_onnx_to_int8(input_onnx_path: Path, output_quant_path: Path):
    """Dynamically quantizes ONNX model to INT8 precision."""
    if quantize_dynamic is None:
        print("[WARNING] onnxruntime.quantization not available. Skipping quantization.")
        return
        
    print(f"\n[Quantizer] Quantizing ONNX model to INT8: {output_quant_path}...")
    quantize_dynamic(
        model_input=str(input_onnx_path),
        model_output=str(output_quant_path),
        weight_type=QuantType.QInt8
    )
    print(f"[OK] INT8 Quantized ONNX model exported ({output_quant_path.stat().st_size / 1024 / 1024:.2f} MB)")

def export_relations_matrix(
    data_files: List[str],
    concept_matrix: np.ndarray,
    concepts_list: List[str],
    model: nn.Module,
    tokenizer,
    output_bin_path: Path,
    output_meta_path: Path,
    output_dim: int = 256
):
    """
    Computes empirical translation offset vectors r_rel for all canonical relations.
    Fixes CUDA/CPU device migration safely.
    """
    print(f"\n[Relations Exporter] Computing exact empirical offset vectors...")
    output_bin_path.parent.mkdir(parents=True, exist_ok=True)
    output_meta_path.parent.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is not None:
        model.to(device)
        model.eval()

    c_to_idx = {c: i for i, c in enumerate(concepts_list)}
    relation_keys = sorted(list(CANONICAL_RELATIONS.keys()))
    rel_displacements = defaultdict(list)

    for path_str in data_files:
        p = Path(path_str)
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            triples = json.load(f)
            for item in triples:
                h = item.get("head", "").strip()
                t = item.get("tail", "").strip()
                r = item.get("relation", "").strip()
                
                if h in c_to_idx and t in c_to_idx and r in CANONICAL_RELATIONS:
                    vh = concept_matrix[c_to_idx[h]]
                    vt = concept_matrix[c_to_idx[t]]
                    diff = vt - vh
                    rel_displacements[r].append(diff)

    final_rel_matrix = np.zeros((len(relation_keys), output_dim), dtype=np.float32)

    for i, r_name in enumerate(relation_keys):
        if len(rel_displacements[r_name]) > 0:
            avg_disp = np.mean(rel_displacements[r_name], axis=0)
            norm = np.linalg.norm(avg_disp)
            if norm > 1e-6:
                final_rel_matrix[i] = avg_disp / norm
        else:
            # Fallback: encode template with model on active device
            meta = CANONICAL_RELATIONS[r_name]
            prompt = meta.get("en_template", "").format(head="concept")
            if model is not None and tokenizer is not None:
                inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=64)
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                with torch.no_grad():
                    if hasattr(model, "encode"):
                        vec = model.encode(input_ids, attention_mask)[0].cpu().numpy()
                    else:
                        vec = model(input_ids, attention_mask)[0].cpu().numpy()
                final_rel_matrix[i] = vec / max(np.linalg.norm(vec), 1e-9)

    write_ckge_binary(output_bin_path, final_rel_matrix, len(relation_keys), output_dim, quantize_int8=True)
    
    with open(output_meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "num_relations": len(relation_keys),
            "dim": output_dim,
            "relations": relation_keys
        }, f, ensure_ascii=False, indent=2)

    print(f"[OK] Exported {len(relation_keys)} relation offset vectors to: {output_bin_path}")

def export_concepts_to_rust_binary(
    concepts: List[str],
    embeddings: np.ndarray,
    bin_output_path: Path,
    dict_output_path: Path,
    quantize_int8: bool = True
):
    """Exports concept vectors and dictionary mapping."""
    bin_output_path.parent.mkdir(parents=True, exist_ok=True)
    dict_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    num_concepts, dim = embeddings.shape
    write_ckge_binary(bin_output_path, embeddings, num_concepts, dim, quantize_int8=quantize_int8)
    
    with open(dict_output_path, "w", encoding="utf-8") as f:
        json.dump({
            "num_concepts": num_concepts,
            "dim": dim,
            "concepts": concepts
        }, f, ensure_ascii=False, indent=2)
        
    print(f"[OK] Exported {num_concepts:,} concept vectors ({dim}d, INT8: {quantize_int8}) to:")
    print(f"     Binary:     {bin_output_path} ({bin_output_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"     Dictionary: {dict_output_path}")

def load_model_for_export(checkpoint_dir: Path, backbone_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> nn.Module:
    """Dynamically detects whether checkpoint is AssembledBiEncoder or SimKGCBiEncoder and loads it."""
    model_path = checkpoint_dir / "simkgc_model.pt"
    if not model_path.exists():
        # Fallback to base model
        return SimKGCBiEncoder(backbone_name=backbone_name)

    state_dict = torch.load(model_path, map_location="cpu")
    keys = list(state_dict.keys())
    
    if any(k.startswith("text_embedder.") or k.startswith("relational_core.") for k in keys):
        print("[Model Loader] Detected Modular AssembledBiEncoder checkpoint structure.")
        embedder = TextEmbedder(backbone_name=backbone_name, output_dim=256, split_layer=8)
        core = RelationalCore(backbone_name=backbone_name, input_dim=256, output_dim=256, split_layer=8, total_layers=12)
        model = AssembledBiEncoder(embedder, core)
    else:
        print("[Model Loader] Detected Standard SimKGCBiEncoder checkpoint structure.")
        model = SimKGCBiEncoder(backbone_name=backbone_name)
        
    model.load_state_dict(state_dict)
    model.eval()
    return model

def run_production_export(
    checkpoint_dir: Path,
    data_files: list,
    output_dir: Path,
    max_concepts: int = 50000,
    fa_quota: int = 15000,
    en_quota: int = 35000,
    teacher_cache_path: Optional[Path] = None,
    teacher_dict_path: Optional[Path] = None
):
    """Full export sequence producing all production assets for Centrode."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    except Exception:
        try:
            tokenizer = BertTokenizer.from_pretrained(str(checkpoint_dir))
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            
    # Save tokenizer directly to exports directory for isolated runtime bundling
    tokenizer.save_pretrained(str(output_dir))
    
    # Load model with polymorphic state dict detection
    model = load_model_for_export(checkpoint_dir)
    
    # 1. ONNX & INT8 Export
    onnx_path = output_dir / "simkgc_256d.onnx"
    quant_path = output_dir / "simkgc_256d_int8.onnx"
    export_to_onnx(model, tokenizer, onnx_path)
    quantize_onnx_to_int8(onnx_path, quant_path)
    
    # 2. Relations Ontology Metadata
    export_relations_metadata(output_dir / "relations_ontology.json")
    
    # 3. Curate Top 50,000 Concepts
    selected_concepts = select_top_production_concepts(
        data_paths=data_files,
        total_quota=max_concepts,
        fa_quota=fa_quota,
        en_quota=en_quota
    )
    
    # 4. Check if Teacher Cache exists for instant vector lookup
    embeddings = None
    if teacher_cache_path and teacher_dict_path and teacher_cache_path.exists() and teacher_dict_path.exists():
        print(f"\n[Teacher Cache] Reusing high-grade BGE-M3 teacher vectors from {teacher_cache_path}...")
        teacher_npy = np.load(teacher_cache_path)
        with open(teacher_dict_path, "r", encoding="utf-8") as f:
            cached_concepts = json.load(f)
        c_to_idx = {c: i for i, c in enumerate(cached_concepts)}
        
        found_embeddings = []
        valid_concepts = []
        for c in selected_concepts:
            if c in c_to_idx:
                found_embeddings.append(teacher_npy[c_to_idx[c]])
                valid_concepts.append(c)
                
        if len(found_embeddings) >= int(0.9 * len(selected_concepts)):
            embeddings = np.vstack(found_embeddings)
            selected_concepts = valid_concepts
            print(f"[Teacher Cache] Successfully extracted {len(selected_concepts):,} concept vectors from teacher cache!")

    # Fallback: compute embeddings via model if teacher cache was missing or partial
    if embeddings is None and len(selected_concepts) > 0:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nPre-encoding {len(selected_concepts):,} concepts on {device} (Batch size: 2048)...")
        
        model.to(device)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
            
        all_embeddings = []
        chunk_size = 2048 if torch.cuda.is_available() else 256
        
        for i in range(0, len(selected_concepts), chunk_size):
            batch = selected_concepts[i:i + chunk_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            with torch.inference_mode():
                raw_m = model.module if hasattr(model, "module") else model
                emb = raw_m.encode(input_ids, attention_mask).float().cpu().numpy()
                all_embeddings.append(emb)
                
        embeddings = np.vstack(all_embeddings)
        
    export_concepts_to_rust_binary(
        concepts=selected_concepts,
        embeddings=embeddings,
        bin_output_path=output_dir / "concepts_256d_int8.bin",
        dict_output_path=output_dir / "concepts_dict.json",
        quantize_int8=True
    )
    
    # 5. Export True Geometric Translation Offset Vectors (relations_256d_int8.bin)
    export_relations_matrix(
        data_files=data_files,
        concept_matrix=embeddings,
        concepts_list=selected_concepts,
        model=model,
        tokenizer=tokenizer,
        output_bin_path=output_dir / "relations_256d_int8.bin",
        output_meta_path=output_dir / "relations_metadata.json"
    )
    
    print("\n[SUCCESS] Production export sequence completed successfully!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export Centrode Production Model & 50K Concept Matrix")
    parser.add_argument("--checkpoint", default="checkpoints/simkgc_fa_en", help="Path to checkpoint directory")
    parser.add_argument("--data", default="data/raw/conceptnet_clean.json", help="Path to clean dataset")
    parser.add_argument("--output", default="exports", help="Output export directory")
    parser.add_argument("--max-concepts", type=int, default=50000, help="Max concepts in shipped binary")
    parser.add_argument("--fa-quota", type=int, default=15000, help="Persian concept quota")
    parser.add_argument("--en-quota", type=int, default=35000, help="English concept quota")
    parser.add_argument("--teacher-cache", default="cache/bge_m3_concept_targets.npy", help="Teacher cache path")
    parser.add_argument("--teacher-dict", default="cache/concepts_dict.json", help="Teacher dict path")
    args = parser.parse_args()
    
    t_cache = Path(args.teacher_cache) if args.teacher_cache else None
    t_dict = Path(args.teacher_dict) if args.teacher_dict else None
    
    run_production_export(
        checkpoint_dir=Path(args.checkpoint),
        data_files=[args.data],
        output_dir=Path(args.output),
        max_concepts=args.max_concepts,
        fa_quota=args.fa_quota,
        en_quota=args.en_quota,
        teacher_cache_path=t_cache,
        teacher_dict_path=t_dict
    )
