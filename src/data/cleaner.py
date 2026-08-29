#!/usr/bin/env python3
"""
Data Cleaning and Text Normalization Module for Knowledge Graph Completion.
Handles:
  1. Persian text normalization (Arabic Yeh/Kaf -> Persian Ye/Ke, ZWNJ handling).
  2. ConceptNet suffix stripping (e.g. 'car (noun)' -> 'car', '/c/fa/ایران/n' -> 'ایران').
  3. Graph cleaning: self-loop removal, deduplication, low-weight edge filtering, and casing.
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Tuple, Set

# Persian character normalization mappings
PERSIAN_CHAR_MAP = {
    'ي': 'ی',
    'ك': 'ک',
    'ى': 'ی',
    'ة': 'ه',
    'ۀ': 'ه',
    '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
    '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9',
}

def normalize_text(text: str, lang: str = "fa") -> str:
    """
    Normalizes Persian and English concept strings.
    """
    if not text:
        return ""
        
    text = text.strip()
    
    # Strip ConceptNet linguistic tags like ' (noun)', ' (verb)', ' (adjective)'
    text = re.sub(r'\s*\((noun|verb|adjective|adverb|n|v|adj|adv|phrase)\)', '', text, flags=re.IGNORECASE)
    
    # Persian specific normalization
    if lang == "fa":
        for k, v in PERSIAN_CHAR_MAP.items():
            text = text.replace(k, v)
        # Normalize multiple spaces and ZWNJ
        text = re.sub(r'[\u200B-\u200D\uFEFF]', '\u200c', text) # Normalize to standard ZWNJ
        text = re.sub(r'\s+', ' ', text)
    else:
        # English: normalize whitespace and lowercase
        text = re.sub(r'\s+', ' ', text).lower()
        
    return text.strip()

def clean_knowledge_graph(triples: List[Dict], min_weight: float = 1.0) -> List[Dict]:
    """
    Cleans raw triplet dataset:
      - Normalizes text in head, relation, tail.
      - Removes self-loops (head == tail).
      - Filters low-confidence or negative-weight assertions.
      - Deduplicates triples preserving highest weight.
    """
    cleaned_map: Dict[Tuple[str, str, str], Dict] = {}
    
    for item in triples:
        head_raw = item.get("head", "")
        rel_raw = item.get("relation", "")
        tail_raw = item.get("tail", "")
        lang = item.get("lang", "en")
        weight = float(item.get("weight", 1.0))
        
        if weight < min_weight:
            continue
            
        head = normalize_text(head_raw, lang)
        tail = normalize_text(tail_raw, lang)
        rel = normalize_text(rel_raw, "en") # Keep relations standardized
        
        # Discard invalid, empty, or self-loop triples
        if not head or not tail or not rel or head == tail:
            continue
            
        # Discard single punctuation tokens
        if len(head) <= 1 and not head.isalnum():
            continue
        if len(tail) <= 1 and not tail.isalnum():
            continue
            
        key = (head, rel, tail)
        if key not in cleaned_map or weight > cleaned_map[key]["weight"]:
            cleaned_map[key] = {
                "head": head,
                "relation": rel,
                "tail": tail,
                "lang": lang,
                "weight": weight
            }
            
    cleaned_list = list(cleaned_map.values())
    return cleaned_list

def clean_dataset_file(input_path: Path, output_path: Path, min_weight: float = 1.0) -> int:
    """Loads JSON dataset, cleans all records, and writes cleaned output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    cleaned = clean_knowledge_graph(raw_data, min_weight=min_weight)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
        
    print(f"Data Cleaning: {len(raw_data):,} raw -> {len(cleaned):,} clean triples ({len(raw_data) - len(cleaned):,} noisy items removed).")
    return len(cleaned)

if __name__ == "__main__":
    raw_file = Path("data/raw/conceptnet_subset.json")
    clean_file = Path("data/raw/conceptnet_clean.json")
    if raw_file.exists():
        clean_dataset_file(raw_file, clean_file)
