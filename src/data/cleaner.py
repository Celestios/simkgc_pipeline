#!/usr/bin/env python3
"""
Knowledge Graph Triple Cleaner, Invertor, and Multi-Source Dataset Merger.
Merges base knowledge graphs (downloaded from Hugging Face) with newly generated
synthetic phrase triplets from Git, applies typography normalization, and generates
bidirectional inverse graph links.
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
from src.data.relations import (
    CANONICAL_RELATIONS,
    CANONICAL_RELATION_NAMES,
    canonicalize_relation,
    get_inverse_relation
)
from src.utils.checkpoint import download_from_hf, get_resolved_hf_token

def is_persian_text(text: str) -> bool:
    """Detects if a string contains Persian/Arabic characters."""
    return bool(re.search(r'[\u0600-\u06FF]', text))

def normalize_concept_text(text: str, lang: str = "en") -> str:
    """
    Normalizes concept strings, enforces standard Persian characters (ی, ک),
    handles zero-width non-joiners (ZWNJ), and strips punctuation and tags like (noun).
    """
    text = text.strip().strip('"\'')
    text = re.sub(r'\(.*?\)', '', text)  # Strip ConceptNet grammatical tags
    if is_persian_text(text) or lang == "fa":
        text = text.replace('\u064A', '\u06CC').replace('\u0649', '\u06CC')
        text = text.replace('\u0643', '\u06A9')
        text = re.sub(r'[\u200B-\u200D\uFEFF]', '\u200c', text)
        text = re.sub(r'\s+', ' ', text)
    else:
        text = re.sub(r'\s+', ' ', text)
        
    return text.strip()

normalize_text = normalize_concept_text

def clean_knowledge_graph(triples: List[Dict], min_weight: float = 1.0, generate_inverses: bool = True) -> List[Dict]:
    """
    Cleans raw assertions, maps to canonical relations, and generates bidirectional inverse links
    with verified language attributes on both forward and inverse triples.
    """
    cleaned_map: Dict[Tuple[str, str, str], Dict] = {}
    
    for item in triples:
        head_raw = item.get("head", "")
        rel_raw = item.get("relation", "")
        tail_raw = item.get("tail", "")
        lang = item.get("lang", "en")
        weight = float(item.get("weight", 2.0))
        
        if weight < min_weight:
            continue
            
        canonical_rel = canonicalize_relation(rel_raw)
        if not canonical_rel or canonical_rel not in CANONICAL_RELATION_NAMES:
            continue
            
        head = normalize_concept_text(head_raw, lang)
        tail = normalize_concept_text(tail_raw, lang)
        
        if not head or not tail or head == tail:
            continue
            
        # Allow rich phrases up to 70 characters
        if len(head) > 70 or len(tail) > 70:
            continue

        head_lang = "fa" if is_persian_text(head) else "en"
        tail_lang = "fa" if is_persian_text(tail) else "en"
            
        # 1. Forward Triple
        forward_key = (head, canonical_rel, tail)
        if forward_key not in cleaned_map or weight > cleaned_map[forward_key]["weight"]:
            cleaned_map[forward_key] = {
                "head": head,
                "relation": canonical_rel,
                "tail": tail,
                "lang": head_lang,
                "weight": weight
            }
            
        # 2. Bidirectional Inverse Triple (Tagged with tail entity's true script language)
        if generate_inverses:
            inv_rel = get_inverse_relation(canonical_rel)
            if inv_rel and inv_rel in CANONICAL_RELATION_NAMES:
                inverse_key = (tail, inv_rel, head)
                if inverse_key not in cleaned_map or weight > cleaned_map[inverse_key]["weight"]:
                    cleaned_map[inverse_key] = {
                        "head": tail,
                        "relation": inv_rel,
                        "tail": head,
                        "lang": tail_lang,
                        "weight": weight
                    }
                    
    return list(cleaned_map.values())

def clean_dataset_files(
    input_paths: List[str],
    output_path: str,
    min_weight: float = 1.0,
    from_hf: Optional[str] = None,
    hf_token: Optional[str] = None
) -> int:
    """
    Reads multiple input JSON files (from disk or auto-downloaded from Hugging Face),
    merges them with local/git synthetic phrase datasets, and writes the unified output.
    """
    # 1. Auto-download from Hugging Face if a specified file is missing locally
    if from_hf:
        for ip in input_paths:
            p = Path(ip)
            if not p.exists():
                print(f"[Cleaner] Input file '{p.name}' not found locally. Attempting download from {from_hf}...")
                download_from_hf(p.name, p.parent, repo_id=from_hf, token=hf_token)

    all_triples = []
    loaded_files = 0

    for ip in input_paths:
        p = Path(ip)
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_triples.extend(data)
                        loaded_files += 1
                        print(f"  [+] Loaded {len(data):,} triples from {p}")
                    elif isinstance(data, dict) and "triples" in data:
                        all_triples.extend(data["triples"])
                        loaded_files += 1
                        print(f"  [+] Loaded {len(data['triples']):,} triples from {p}")
            except Exception as e:
                print(f"  [-] Error reading {p}: {e}")

    print(f"\n[Cleaner] Total Raw Assertions Loaded: {len(all_triples):,} across {loaded_files} source files.")
    cleaned = clean_knowledge_graph(all_triples, min_weight=min_weight, generate_inverses=True)
    
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
        
    print(f"✓ Combined & Canonicalized: {len(cleaned):,} high-quality bidirectional triples saved to: {output_path}")
    return len(cleaned)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean, canonicalize, and merge knowledge graph datasets.")
    parser.add_argument("--inputs", nargs="+", default=[
        "data/raw/conceptnet_clean.json",
        "data/raw/conceptnet_subset.json",
        "data/synthetic/all_triplets_deduped.json"
    ], help="Input file paths to merge")
    parser.add_argument("--output", default="data/raw/conceptnet_clean.json", help="Unified output file path")
    parser.add_argument("--min-weight", type=float, default=0.5, help="Minimum assertion confidence weight")
    parser.add_argument("--from-hf", default=None, help="Optional Hugging Face repo ID to pull base dataset from")
    args = parser.parse_args()
    
    clean_dataset_files(
        input_paths=args.inputs,
        output_path=args.output,
        min_weight=args.min_weight,
        from_hf=args.from_hf
    )
