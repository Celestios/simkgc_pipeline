#!/usr/bin/env python3
"""
High-Precision Knowledge Graph Cleaner with Canonical Relation Filtering.
Filters assertions to ensure:
  1. Only canonical relations (IsA, PartOf, HasProperty, UsedFor, etc.) are preserved.
  2. Persian text is standardized (Arabic characters -> Persian, ZWNJ normalization).
  3. ConceptNet noisy suffixes and self-loops are purged.
  4. Deduplicates assertions preserving maximum weight.
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
    from src.data.relations import canonicalize_relation, CANONICAL_RELATION_NAMES
except ImportError:
    from relations import canonicalize_relation, CANONICAL_RELATION_NAMES

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

def clean_knowledge_graph(triples: List[Dict], min_weight: float = 1.0) -> List[Dict]:
    """Cleans raw assertions and maps relations to canonical ontology."""
    cleaned_map: Dict[Tuple[str, str, str], Dict] = {}
    
    for item in triples:
        head_raw = item.get("head", "")
        rel_raw = item.get("relation", "")
        tail_raw = item.get("tail", "")
        lang = item.get("lang", "en")
        weight = float(item.get("weight", 1.0))
        
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
            
        key = (head, canonical_rel, tail)
        if key not in cleaned_map or weight > cleaned_map[key]["weight"]:
            cleaned_map[key] = {
                "head": head,
                "relation": canonical_rel,
                "tail": tail,
                "lang": lang,
                "weight": weight
            }
            
    return list(cleaned_map.values())

def clean_dataset_file(input_path: Path, output_path: Path, min_weight: float = 1.0) -> int:
    """Reads raw JSON dataset in chunks, cleans and writes canonical dataset."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Reading raw dataset from {input_path}...")
    
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    print(f"Loaded {len(raw_data):,} raw assertions. Cleaning and filtering to canonical relations...")
    cleaned = clean_knowledge_graph(raw_data, min_weight=min_weight)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
        
    print("=" * 65)
    print(f"[CLEAN COMPLETE] {len(raw_data):,} raw -> {len(cleaned):,} canonical assertions.")
    print(f"Saved to: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print("=" * 65)
    return len(cleaned)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and Canonicalize Knowledge Graph Data")
    parser.add_argument("--input", type=str, default="data/raw/conceptnet_subset.json", help="Raw dataset path")
    parser.add_argument("--output", type=str, default="data/raw/conceptnet_clean.json", help="Cleaned dataset path")
    parser.add_argument("--min-weight", type=float, default=1.0, help="Minimum assertion weight")
    args = parser.parse_args()

    clean_dataset_file(Path(args.input), Path(args.output), min_weight=args.min_weight)
