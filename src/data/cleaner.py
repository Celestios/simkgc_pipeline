#!/usr/bin/env python3
"""
High-Precision Knowledge Graph Cleaner with Canonical Relation Filtering & Bidirectional Inverses.
Filters assertions to ensure:
  1. Only canonical relations (32 categories) are preserved.
  2. Generates bidirectional inverse relations automatically with correct language tagging.
  3. Persian text is standardized (Arabic characters -> Persian, ZWNJ normalization).
  4. ConceptNet noisy suffixes, self-loops, and duplicates are purged.
  5. Supports merging multiple raw/synthetic datasets in one unified pass.
"""

import sys
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Set

# Ensure repo root is in Python module search path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.data.relations import (
        canonicalize_relation,
        get_inverse_relation,
        CANONICAL_RELATION_NAMES,
        is_persian_text
    )
except ImportError:
    from relations import (
        canonicalize_relation,
        get_inverse_relation,
        CANONICAL_RELATION_NAMES,
        is_persian_text
    )

PERSIAN_CHAR_MAP = {
    'ي': 'ی',
    'ك': 'ک',
    'ى': 'ی',
    'ة': 'ه',
    'ۀ': 'ه',
    '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
    '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9',
}

def normalize_concept_text(text: str, lang: str = "fa") -> str:
    """Normalizes concept labels."""
    if not text:
        return ""
        
    text = text.strip()
    text = re.sub(r'\s*\((noun|verb|adjective|adverb|n|v|adj|adv|phrase)\)', '', text, flags=re.IGNORECASE)
    
    if lang == "fa" or is_persian_text(text):
        for k, v in PERSIAN_CHAR_MAP.items():
            text = text.replace(k, v)
        text = re.sub(r'[\u200B-\u200D\uFEFF]', '\u200c', text)
        text = re.sub(r'\s+', ' ', text)
    else:
        text = re.sub(r'\s+', ' ', text).lower()
        
    return text.strip()

# Backward compatibility alias
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
            
        if len(head) > 50 or len(tail) > 50:
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

def clean_dataset_files(input_paths: List[str], output_path: str, min_weight: float = 1.0) -> int:
    """Reads multiple input JSON files, merges and cleans them, and writes output."""
    all_triples = []
    for ip in input_paths:
        p = Path(ip)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_triples.extend(data)
                elif isinstance(data, dict) and "triples" in data:
                    all_triples.extend(data["triples"])

    print(f"Loaded {len(all_triples):,} raw assertions across {len(input_paths)} files.")
    cleaned = clean_knowledge_graph(all_triples, min_weight=min_weight, generate_inverses=True)
    
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
        
    print(f"Saved {len(cleaned):,} high-quality canonical assertions to {output_path}")
    return len(cleaned)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and canonicalize knowledge graph triples.")
    parser.add_argument("--inputs", nargs="+", default=["data/raw/conceptnet_subset.json"], help="Input file paths")
    parser.add_argument("--output", default="data/raw/conceptnet_clean.json", help="Output file path")
    parser.add_argument("--min-weight", type=float, default=1.0, help="Minimum assertion confidence weight")
    args = parser.parse_args()
    
    clean_dataset_files(args.inputs, args.output, args.min_weight)
