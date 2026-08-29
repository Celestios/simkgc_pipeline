#!/usr/bin/env python3
"""
Production ONNX & Rust Binary Exporter for Centrode.
Exports:
  1. simkgc_256d.onnx / simkgc_256d_int8.onnx (Inference Bi-Encoder for Rust runtime)
  2. concepts_256d_int8.bin (Zero-copy flat memory-mappable INT8 concept matrix)
  3. concepts_dict.json (Index to concept string dictionary)
  4. relations_ontology.json (List of canonical relations with descriptions for Centrode UI)
"""

import sys
import json
import struct
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict
from transformers import AutoTokenizer, BertTokenizer
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

from src.model.biencoder import SimKGCBiEncoder
from src.data.relations import CANONICAL_RELATIONS

def export_relations_metadata(output_path: Path):
    """Exports canonical relations metadata for Centrode Flutter/Rust UI."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(CANONICAL_RELATIONS, f, ensure_ascii=False, indent=2)
    print(f"[OK] Exported relations ontology metadata to: {output_path}")

def export_to_onnx(model: SimKGCBiEncoder, tokenizer, output_path: Path, max_length: int = 64):
    """Exports PyTorch Bi-Encoder single forward query pass to ONNX."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    
    dummy_text = "concept [SEP] relation"
    dummy_inputs = tokenizer(dummy_text, return_tensors="pt", max_length=max_length, padding="max_length", truncation=True)
    dummy_input_ids = dummy_inputs["input_ids"]
    dummy_attention_mask = dummy_inputs["attention_mask"]

    class QueryEncoderWrapper(torch.nn.Module):
        def __init__(self, biencoder):
            super().__init__()
            self.biencoder = biencoder
        def forward(self, input_ids, attention_mask):
            return self.biencoder.encode(input_ids, attention_mask)

    wrapper = QueryEncoderWrapper(model)
    print(f"Exporting PyTorch model to ONNX: {output_path}...")
    
    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_attention_mask),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["embedding"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "embedding": {0: "batch_size"}
        },
        opset_version=14,
        do_constant_folding=True,
        dynamo=False
    )
    print(f"[OK] ONNX export complete ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")

def quantize_onnx_to_int8(input_onnx_path: Path, output_int8_path: Path):
    """Quantizes ONNX weights to INT8 precision for ultra-fast CPU inference."""
    print(f"Quantizing ONNX model to INT8: {output_int8_path}...")
    quantize_dynamic(
        model_input=str(input_onnx_path),
        model_output=str(output_int8_path),
        weight_type=QuantType.QInt8
    )
    print(f"[OK] INT8 Quantized ONNX saved ({output_int8_path.stat().st_size / 1024 / 1024:.2f} MB)")

def export_concepts_to_rust_binary(
    concepts: List[str],
    embeddings: np.ndarray,
    bin_output_path: Path,
    dict_output_path: Path,
    quantize_int8: bool = True
):
    """
    Serializes pre-encoded concept matrix into flat binary buffer for zero-copy mmap in Rust.
    Header: 'CKGE' [4s] + NumConcepts [u32] + Dim [u32] + Precision [u32] (16 bytes).
    """
    bin_output_path.parent.mkdir(parents=True, exist_ok=True)
    num_concepts, dim = embeddings.shape
    
    if quantize_int8:
        quantized_matrix = np.clip(np.round(embeddings * 127.0), -127, 127).astype(np.int8)
        precision_code = 1
        payload = quantized_matrix.tobytes()
    else:
        precision_code = 4
        payload = embeddings.astype(np.float32).tobytes()
        
    header = struct.pack("<4sIII", b"CKGE", num_concepts, dim, precision_code)
    
    with open(bin_output_path, "wb") as f:
        f.write(header)
        f.write(payload)
        
    with open(dict_output_path, "w", encoding="utf-8") as f:
        json.dump({
            "num_concepts": num_concepts,
            "dim": dim,
            "concepts": concepts
        }, f, ensure_ascii=False, indent=2)
        
    print(f"[OK] Exported {num_concepts:,} concept vectors ({dim}d, INT8: {quantize_int8}) to:")
    print(f"     Binary:     {bin_output_path} ({bin_output_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"     Dictionary: {dict_output_path}")

def collect_unique_concepts(data_paths: list) -> list:
    """Collects unique head and tail concepts from dataset files."""
    concepts = set()
    for p in data_paths:
        file_path = Path(p)
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                items = json.load(f)
                for it in items:
                    if "head" in it and it["head"]:
                        concepts.add(it["head"].strip())
                    if "tail" in it and it["tail"]:
                        concepts.add(it["tail"].strip())
    return sorted(list(concepts))

def run_production_export(checkpoint_dir: Path, data_files: list, output_dir: Path):
    """Full export sequence producing all 4 production assets for Centrode."""
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    except Exception:
        tokenizer = BertTokenizer.from_pretrained(checkpoint_dir)
        
    model = SimKGCBiEncoder(backbone_name=str(checkpoint_dir))
    model.load_state_dict(torch.load(checkpoint_dir / "simkgc_model.pt", map_location="cpu"))
    model.eval()
    
    # 1. ONNX & INT8 Export
    onnx_path = output_dir / "simkgc_256d.onnx"
    quant_path = output_dir / "simkgc_256d_int8.onnx"
    export_to_onnx(model, tokenizer, onnx_path)
    quantize_onnx_to_int8(onnx_path, quant_path)
    
    # 2. Relations Ontology Metadata
    export_relations_metadata(output_dir / "relations_ontology.json")
    
    # 3. Concept Pre-encoding (in chunks)
    concepts = collect_unique_concepts(data_files)
    if len(concepts) > 0:
        print(f"Pre-encoding {len(concepts):,} unique concepts in batches...")
        all_embeddings = []
        chunk_size = 256
        for i in range(0, len(concepts), chunk_size):
            batch = concepts[i:i + chunk_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt")
            with torch.no_grad():
                emb = model.encode(inputs["input_ids"], inputs["attention_mask"]).numpy()
                all_embeddings.append(emb)
        embeddings = np.vstack(all_embeddings)
            
        export_concepts_to_rust_binary(
            concepts=concepts,
            embeddings=embeddings,
            bin_output_path=output_dir / "concepts_256d_int8.bin",
            dict_output_path=output_dir / "concepts_dict.json",
            quantize_int8=True
        )

if __name__ == "__main__":
    chk = Path("checkpoints/simkgc_fa_en")
    run_production_export(chk, ["data/raw/conceptnet_clean.json", "data/synthetic/generated_triples.json"], Path("exports"))
