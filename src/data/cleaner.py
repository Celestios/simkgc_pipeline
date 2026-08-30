#!/usr/bin/env python3
"""
High-Precision Knowledge Graph Cleaner with Canonical Relation Filtering & Bidirectional Inverses.
Filters assertions to ensure:
  1. Only canonical relations (32 categories) are preserved.
  2. Generates bidirectional inverse relations automatically.
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
        CANONICAL_RELATION_NAMES
    )
except ImportError:
    from relations import (
        canonicalize_relation,
        get_inverse_relation,
        CANONICAL_RELATION_NAMES
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
    
    if lang == "fa":
        for k, v in PERSIAN_CHAR_MAP.items():
            text = text.replace(k, v)
        text = re.sub(r'[\u200B-\u200D\uFEFF]', '\u200c', text)
        text = re.sub(r'\s+', ' ', text)
    else:
        text = re.sub(r'\s+', ' ', text).lower()
        
    return text.strip()

# Alias for backwards compatibility
normalize_text = normalize_concept_text

def clean_knowledge_graph(triples: List[Dict], min_weight: float = 1.0, generate_inverses: bool = True) -> List[Dict]:
    """
    Cleans raw assertions, maps to canonical relations, and generates bidirectional inverse links.
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
            
        # 1. Forward Triple
        forward_key = (head, canonical_rel, tail)
        if forward_key not in cleaned_map or weight > cleaned_map[forward_key]["weight"]:
            cleaned_map[forward_key] = {
                "head": head,
                "relation": canonical_rel,
                "tail": tail,
                "lang": lang,
                "weight": weight
            }
            
        # 2. Bidirectional Inverse Triple
        if generate_inverses:
            inv_rel = get_inverse_relation(canonical_rel)
            if inv_rel and inv_rel in CANONICAL_RELATION_NAMES:
                inverse_key = (tail, inv_rel, head)
                if inverse_key not in cleaned_map or weight > cleaned_map[inverse_key]["weight"]:
                    cleaned_map[inverse_key] = {
                        "head": tail,
                        "relation": inv_rel,
                        "tail": head,
                        "lang": lang,
                        "weight": weight
                    }
            
    return list(cleaned_map.values())

def clean_dataset_files(input_paths: List[Path], output_path: Path, min_weight: float = 1.0) -> int:
    """Reads one or multiple JSON datasets, cleans, merges and writes canonical dataset."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_raw = []
    
    for inp in input_paths:
        p = Path(inp)
        if not p.exists():
            print(f"[Warning] Input file {p} does not exist, skipping.")
            continue
        print(f"Reading dataset: {p}...")
        with open(p, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                all_raw.extend(data)
                print(f"  -> Loaded {len(data):,} assertions from {p.name}")
            except Exception as e:
                print(f"  -> Error loading {p}: {e}")
                
    print(f"\nTotal loaded: {len(all_raw):,} raw assertions across {len(input_paths)} files.")
    print("Cleaning, deduplicating, generating inverses, and mapping to 32 canonical relations...")
    cleaned = clean_knowledge_graph(all_raw, min_weight=min_weight, generate_inverses=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
        
    print("=" * 65)
    print(f"[CLEAN COMPLETE] {len(all_raw):,} raw -> {len(cleaned):,} canonical bidirectional assertions.")
    print(f"Saved to: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print("=" * 65)
    return len(cleaned)

# Alias
clean_dataset_file = lambda inp, out, min_weight=1.0: clean_dataset_files([inp], out, min_weight)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean, Canonicalize, and Merge Knowledge Graph Datasets")
    parser.add_argument("--input", nargs="+", default=["data/raw/conceptnet_subset.json", "data/synthetic/all_triplets_deduped.json"], help="Input dataset path(s)")
    parser.add_argument("--output", type=str, default="data/raw/conceptnet_clean.json", help="Clean dataset output path")
    parser.add_argument("--min-weight", type=float, default=0.5, help="Minimum assertion confidence weight")
    args = parser.parse_args()
    
    clean_dataset_files([Path(p) for p in args.input], Path(args.output), min_weight=args.min_weight)
