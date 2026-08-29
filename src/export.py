#!/usr/bin/env python3
"""
Model Exporter and Rust Binary Serializer.
Exports trained SimKGC PyTorch models to ONNX, applies INT8 quantization,
and encodes concept dictionaries into zero-copy binary matrices for Centrode.
Contains zero hardcoded data.
"""

import os
import sys
import json
import struct
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer

# Add current folder to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model.biencoder import SimKGCBiEncoder

class _ONNXEncoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, input_ids, attention_mask):
        return self.model.encode(input_ids, attention_mask)

def export_to_onnx(model: torch.nn.Module, tokenizer, onnx_output_path: Path, max_seq_length: int = 64):
    """
    Exports the trained PyTorch encoder to an ONNX graph with dynamic sequence and batch dimensions.
    """
    onnx_output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    wrapper = _ONNXEncoderWrapper(model)
    
    # Dummy tensor for tracing
    dummy_input_ids = torch.zeros((1, max_seq_length), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, max_seq_length), dtype=torch.long)
    
    print(f"Exporting PyTorch model to ONNX: {onnx_output_path}...")
    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_attention_mask),
        str(onnx_output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["embeddings"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "embeddings": {0: "batch_size"}
        },
        opset_version=14,
        do_constant_folding=True,
        dynamo=False
    )
    print(f"[OK] ONNX export complete ({onnx_output_path.stat().st_size / 1024 / 1024:.2f} MB)")

def quantize_onnx_to_int8(onnx_path: Path, quantized_path: Path):
    """
    Applies dynamic INT8 quantization using ONNX Runtime.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType
    print(f"Quantizing ONNX model to INT8: {quantized_path}...")
    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(quantized_path),
        weight_type=QuantType.QInt8
    )
    print(f"[OK] INT8 Quantized ONNX saved ({quantized_path.stat().st_size / 1024 / 1024:.2f} MB)")

def export_concepts_to_rust_binary(concepts: list, embeddings: np.ndarray,
                                   bin_output_path: Path, dict_output_path: Path,
                                   quantize_int8: bool = True):
    """
    Serializes concept embeddings into a flat binary matrix for zero-overhead Rust loading.
    
    Binary Header (16 bytes):
      - Magic: 'CKGE' (4 bytes ASCII)
      - Num Concepts: uint32 (4 bytes)
      - Embedding Dim: uint32 (4 bytes)
      - Precision: uint32 (1 = INT8, 4 = Float32)
    Payload:
      - Flattened raw byte buffer of shape [Num Concepts, Dim]
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
        
    print(f"[OK] Exported {num_concepts} concept vectors ({dim}d, INT8: {quantize_int8}) to:")
    print(f"     Binary:     {bin_output_path} ({bin_output_path.stat().st_size / 1024:.1f} KB)")
    print(f"     Dictionary: {dict_output_path}")

def collect_unique_concepts(data_paths: list) -> list:
    """Collects unique head and tail concepts from all dataset files."""
    concepts = set()
    for p in data_paths:
        file_path = Path(p)
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                items = json.load(f)
                for it in items:
                    if "head" in it:
                        concepts.add(it["head"])
                    if "tail" in it:
                        concepts.add(it["tail"])
    return sorted(list(concepts))

def run_production_export(checkpoint_dir: Path, data_files: list, output_dir: Path):
    """Loads checkpoint, runs full ONNX export, and encodes concept dictionary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    except Exception:
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(checkpoint_dir)
    model = SimKGCBiEncoder(backbone_name=str(checkpoint_dir))
    model.load_state_dict(torch.load(checkpoint_dir / "simkgc_model.pt", map_location="cpu"))
    model.eval()
    
    # 1. ONNX & INT8 Export
    onnx_path = output_dir / "simkgc_256d.onnx"
    quant_path = output_dir / "simkgc_256d_int8.onnx"
    export_to_onnx(model, tokenizer, onnx_path)
    quantize_onnx_to_int8(onnx_path, quant_path)
    
    # 2. Extract concepts and pre-encode
    concepts = collect_unique_concepts(data_files)
    if len(concepts) > 0:
        print(f"Pre-encoding {len(concepts):,} unique concepts from dataset in batches...")
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
    if (chk / "simkgc_model.pt").exists():
        run_production_export(chk, ["data/raw/conceptnet_subset.json", "data/synthetic/generated_triples.json"], Path("exports"))
    else:
        print("No checkpoint found at checkpoints/simkgc_fa_en. Run src/train.py first.")
