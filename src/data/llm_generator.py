#!/usr/bin/env python3
"""
Synthetic Triplet Generator.
Uses a causal language model (e.g. Qwen2.5 on Kaggle/local) to generate
high-quality commonsense and scientific triplets for seed concepts.
Contains zero hardcoded data.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict

def load_seed_domains(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data.get("domains", [])

def format_generation_prompt(concept: str, lang: str = "fa") -> str:
    if lang == "fa":
        return f"""برای مفهوم '{concept}'، ۵ سه تایی رابطه مفهومی و علمی دقیق به صورت JSON با ساختار زیر تولید کن:
[
  {{"head": "{concept}", "relation": "...", "tail": "..."}}
]"""
    else:
        return f"""For the concept '{concept}', generate 5 precise semantic and scientific relational triples as JSON:
[
  {{"head": "{concept}", "relation": "...", "tail": "..."}}
]"""

def run_llm_generation(model, tokenizer, domain_seeds_path: Path, output_path: Path, max_new_tokens: int = 256):
    """
    Executes triplet generation across all domain seeds using an initialized HuggingFace CausalLM.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    domains = load_seed_domains(domain_seeds_path)
    all_generated = []
    
    for domain in domains:
        domain_name = domain.get("name", "general")
        for concept in domain.get("seeds_fa", []):
            prompt = format_generation_prompt(concept, "fa")
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.7)
            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse JSON block from response
            try:
                start = response_text.find("[")
                end = response_text.rfind("]") + 1
                if start != -1 and end != 0:
                    triples = json.loads(response_text[start:end])
                    for t in triples:
                        t["domain"] = domain_name
                        t["lang"] = "fa"
                        all_generated.append(t)
            except Exception as e:
                print(f"Failed to parse generation for [{concept}]: {e}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_generated, f, ensure_ascii=False, indent=2)
        
    print(f"Saved {len(all_generated)} generated triples to {output_path}")
    return all_generated
