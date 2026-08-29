#!/usr/bin/env python3
"""
Dynamic LLM Knowledge Graph Expander (Smart Teacher).
Automatically extracts unique concepts directly from the cleaned database
(e.g., data/raw/conceptnet_clean.json) and uses a high-capacity Causal LM
(e.g., Qwen2.5 on Kaggle GPU) to generate rich multi-hop and conceptual assertions.

Contains ZERO hardcoded concept lists.
"""

import os
import sys
import json
import torch
from pathlib import Path
from typing import List, Dict, Set
from transformers import AutoModelForCausalLM, AutoTokenizer

def extract_concepts_from_dataset(dataset_path: Path) -> List[str]:
    """Dynamically extracts all unique concept entities from the database."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    concepts = set()
    for item in data:
        if "head" in item and item["head"]:
            concepts.add(item["head"].strip())
        if "tail" in item and item["tail"]:
            concepts.add(item["tail"].strip())
            
    sorted_concepts = sorted(list(concepts))
    print(f"Extracted {len(sorted_concepts):,} unique concepts dynamically from {dataset_path}")
    return sorted_concepts

def format_teacher_prompt(concept: str, lang: str = "fa") -> str:
    """Generates structured prompting for the teacher model."""
    if lang == "fa":
        return f"""وظیفه: برای مفهوم '{concept}'، ۵ رابطه مفهومی، ساختاری و علمی دقیق به زبان فارسی به صورت JSON تولید کن.
فرمت خروجی صرفاً یک آرایه JSON معتبر باشد:
[
  {{"head": "{concept}", "relation": "...", "tail": "..."}}
]"""
    else:
        return f"""Task: For the concept '{concept}', generate 5 precise relational and scientific triples as JSON:
[
  {{"head": "{concept}", "relation": "...", "tail": "..."}}
]"""

def run_teacher_expansion(
    input_dataset_path: Path = Path("data/raw/conceptnet_clean.json"),
    output_path: Path = Path("data/synthetic/generated_triples.json"),
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    max_concepts: int = 500,
    batch_size: int = 4
) -> List[Dict]:
    """
    Executes automated teacher expansion on Kaggle GPU across dynamic database concepts.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    concepts = extract_concepts_from_dataset(input_dataset_path)
    
    # Process up to max_concepts
    target_concepts = concepts[:max_concepts]
    
    print(f"Initializing Teacher Model ({model_id}) on GPU...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    generated_triples = []
    seen = set()
    
    print(f"Expanding {len(target_concepts)} dynamic concepts using {model_id}...")
    for i, concept in enumerate(target_concepts, 1):
        prompt = format_teacher_prompt(concept, lang="fa")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        response_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        try:
            start = response_text.find("[")
            end = response_text.rfind("]") + 1
            if start != -1 and end != 0:
                items = json.loads(response_text[start:end])
                for t in items:
                    if "head" in t and "relation" in t and "tail" in t:
                        h = t["head"].strip()
                        r = t["relation"].strip()
                        tl = t["tail"].strip()
                        if h and r and tl and h != tl:
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
        except Exception as e:
            pass
            
        if i % 25 == 0 or i == len(target_concepts):
            print(f"[{i}/{len(target_concepts)}] Concepts expanded -> {len(generated_triples):,} valid triples generated so far.")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(generated_triples, f, ensure_ascii=False, indent=2)
        
    print(f"\n[COMPLETE] Generated {len(generated_triples):,} authentic teacher expansions saved to: {output_path}")
    
    # Cleanup GPU memory
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return generated_triples

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run LLM Teacher Expansion dynamically from database.")
    parser.add_argument("--input", type=str, default="data/raw/conceptnet_clean.json", help="Input dataset path")
    parser.add_argument("--output", type=str, default="data/synthetic/generated_triples.json", help="Output path")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Teacher model ID")
    parser.add_argument("--max-concepts", type=int, default=500, help="Max unique concepts to expand")
    args = parser.parse_args()

    run_teacher_expansion(
        input_dataset_path=Path(args.input),
        output_path=Path(args.output),
        model_id=args.model,
        max_concepts=args.max_concepts
    )
