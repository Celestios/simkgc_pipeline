#!/usr/bin/env python3
"""
High-Performance Batched LLM Knowledge Graph Expander (Smart Teacher).
Uses Qwen 2.5 with parallel batched GPU tensor generation (16x-32x concurrency),
left-padding for causal LM generation, and real-time live tqdm logging.

Contains ZERO hardcoded concept lists.
"""

import os
import sys
import json
import time
import torch
from collections import Counter
from pathlib import Path
from typing import List, Dict, Set
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def extract_concepts_by_centrality(dataset_path: Path) -> List[str]:
    """
    Extracts unique concept entities ranked by their frequency/centrality in the graph.
    The highest-degree concept hubs are expanded first.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
        
    print(f"Analyzing graph topology and node degree centrality from {dataset_path}...")
    counts = Counter()
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            if "head" in item and item["head"]:
                counts[item["head"].strip()] += 1
            if "tail" in item and item["tail"]:
                counts[item["tail"].strip()] += 1
                
    ranked_concepts = [concept for concept, count in counts.most_common()]
    print(f"Extracted {len(ranked_concepts):,} unique concepts (Ranked from core hubs to leaf nodes).")
    return ranked_concepts

def format_teacher_prompt(concept: str) -> str:
    """Generates structured prompting for the teacher model."""
    return f"""وظیفه: برای مفهوم '{concept}'، ۵ رابطه مفهومی و علمی دقیق به زبان فارسی به صورت JSON تولید کن:
[
  {{"head": "{concept}", "relation": "...", "tail": "..."}}
]"""

def run_batched_teacher_expansion(
    input_dataset_path: Path = Path("data/raw/conceptnet_clean.json"),
    output_path: Path = Path("data/synthetic/generated_triples.json"),
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    max_concepts: int = 5000,
    batch_size: int = 16
) -> List[Dict]:
    """
    Executes high-throughput batched teacher expansion on GPU (16-32 concepts in parallel).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    concepts = extract_concepts_by_centrality(input_dataset_path)
    
    target_concepts = concepts if max_concepts <= 0 else concepts[:max_concepts]
    print(f"\nTargeting top {len(target_concepts):,} central concepts (Batch Size: {batch_size}).")
    
    print(f"Loading {model_id} onto GPU...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    generated_triples = []
    seen = set()
    start_time = time.time()
    
    print("\n" + "=" * 65)
    print(f"STARTING BATCHED GPU GENERATION: {len(target_concepts):,} concepts")
    print("=" * 65)
    
    pbar = tqdm(total=len(target_concepts), desc="Expanding Concepts", unit="concept")
    
    for i in range(0, len(target_concepts), batch_size):
        batch_concepts = target_concepts[i:i + batch_size]
        prompts = [format_teacher_prompt(c) for c in batch_concepts]
        
        inputs = tokenizer(prompts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(model.device)
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=180,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
            
        # Slice output to generated tokens only
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_len:]
        decoded_responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        batch_new_triples = 0
        for resp in decoded_responses:
            try:
                start = resp.find("[")
                end = resp.rfind("]") + 1
                if start != -1 and end != 0:
                    items = json.loads(resp[start:end])
                    for t in items:
                        if "head" in t and "relation" in t and "tail" in t:
                            h = str(t["head"]).strip()
                            r = str(t["relation"]).strip()
                            tl = str(t["tail"]).strip()
                            if h and tl and r and h != tl and len(h) < 60 and len(tl) < 60:
                                key = (h, r, tl)
                                if key not in seen:
                                    seen.add(key)
                                    generated_triples.append({
                                        "head": h,
                                        "relation": r,
                                        "tail": tl,
                                        "lang": "fa",
                                        "source": "qwen2.5_teacher"
                                    })
                                    batch_new_triples += 1
            except Exception:
                pass
                
        pbar.update(len(batch_concepts))
        elapsed = time.time() - start_time
        speed = len(generated_triples) / max(1.0, elapsed)
        pbar.set_postfix({
            "Triples": f"{len(generated_triples):,}",
            "Speed": f"{speed:.1f} trip/s"
        })

    pbar.close()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(generated_triples, f, ensure_ascii=False, indent=2)
        
    print("\n" + "=" * 65)
    print(f"[COMPLETE] Generated {len(generated_triples):,} authentic teacher expansions.")
    print(f"Saved to: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
    print("=" * 65)
    
    # Cleanup GPU memory
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return generated_triples

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="High-Throughput Batched LLM Teacher Expansion.")
    parser.add_argument("--input", type=str, default="data/raw/conceptnet_clean.json", help="Input dataset path")
    parser.add_argument("--output", type=str, default="data/synthetic/generated_triples.json", help="Output path")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Teacher model ID")
    parser.add_argument("--max-concepts", type=int, default=5000, help="Max unique concepts to expand (0 for all)")
    parser.add_argument("--batch-size", type=int, default=16, help="GPU batch size (16-32 for T4 GPU)")
    args = parser.parse_args()

    run_batched_teacher_expansion(
        input_dataset_path=Path(args.input),
        output_path=Path(args.output),
        model_id=args.model,
        max_concepts=args.max_concepts,
        batch_size=args.batch_size
    )
