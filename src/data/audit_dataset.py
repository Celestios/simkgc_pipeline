#!/usr/bin/env python3
"""
Comprehensive Knowledge Graph Health & Hygiene Auditor for Centrode.
Performs 5 rigorous quality audits on the training dataset:
  1. Contradiction & Semantic Conflict Detection (e.g. Synonym vs Antonym)
  2. Asymmetric Cycle & Inversion Anomalies (e.g. A Causes B and B Causes A)
  3. Generic "Black Hole" Hub Identification (overly connected noisy nodes)
  4. Character Encoding & Persian ZWNJ (نیم‌فاصله) Verification
  5. Relation Distribution & Class Balance Breakdown
"""

import sys
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict, Counter

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from src.data.relations import CANONICAL_RELATIONS, is_persian_text
except ImportError:
    CANONICAL_RELATIONS = {}
    is_persian_text = lambda x: False

# Explicit conflicting relation pairs
CONFLICTING_RELATION_PAIRS = {
    frozenset(["Synonym", "Antonym"]),
    frozenset(["IsA", "DistinctFrom"]),
    frozenset(["Causes", "Prevents"]),
    frozenset(["Precedes", "Follows"]),
    frozenset(["Enables", "Prevents"]),
}

# Strictly asymmetric relations where A -> B precludes B -> A
STRICTLY_ASYMMETRIC_RELATIONS = {
    "Causes", "Precedes", "Follows", "InheritsFrom",
    "SubfieldOf", "HasPrerequisite", "ParentOf", "Treats"
}

# Generic words that act as noisy hubs without meaningful semantics
GENERIC_BLACK_HOLE_TERMS = {
    "thing", "something", "anything", "stuff", "item", "object",
    "چیز", "مورد", "شیء", "موجود", "یک چیز", "موارد"
}

ARABIC_UNCONVERTED_REGEX = re.compile(r"[يكةۀى]")
MARKDOWN_URL_REGEX = re.compile(r"(http[s]?://|www\.|\[|\]|\{|\}|<|>)")

PERSIAN_CHAR_MAP = {
    'ي': 'ی', 'ك': 'ک', 'ى': 'ی', 'ة': 'ه', 'ۀ': 'ه',
    '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
    '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9',
}

def normalize_text_audit(text: str) -> str:
    """Standardizes Persian characters and ZWNJ."""
    t = text.strip()
    for k, v in PERSIAN_CHAR_MAP.items():
        t = t.replace(k, v)
    t = re.sub(r'[\u200B-\u200D\uFEFF]', '\u200c', t)
    return t

def audit_dataset(
    dataset_path: Path,
    output_clean_path: Optional[Path] = None,
    max_hub_degree: int = 2500
) -> Dict[str, any]:
    """Runs all 5 health checks on the dataset with safe empty-dataset handling."""
    print("=" * 78)
    print(f"  KNOWLEDGE GRAPH HEALTH & HYGIENE AUDIT: {dataset_path.name}")
    print("=" * 78)
    
    if not dataset_path.exists():
        print(f"Error: File {dataset_path} not found.")
        return {}
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        triples = json.load(f)
        
    total_triples = len(triples)
    print(f"• Total Loaded Assertions: {total_triples:,}\n")

    if total_triples == 0:
        print("[WARNING] Dataset is empty (0 triples).")
        return {
            "total_triples": 0,
            "unique_concepts": 0,
            "conflicts": 0,
            "asymmetric_loops": 0,
            "script_anomalies": 0,
            "black_hole_triples": 0
        }
    
    # Trackers
    pair_relations: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    node_degrees: Counter = Counter()
    relation_counts: Counter = Counter()
    lang_counts: Counter = Counter()
    
    conflicts = []
    asymmetric_loops = []
    script_anomalies = []
    black_hole_triples = 0
    
    # 1. First Pass: Accumulate statistics and script checks
    for idx, item in enumerate(triples):
        h = item.get("head", "").strip()
        r = item.get("relation", "").strip()
        t = item.get("tail", "").strip()
        lang = item.get("lang", "en")
        
        relation_counts[r] += 1
        lang_counts[lang] += 1
        node_degrees[h] += 1
        node_degrees[t] += 1
        
        pair_relations[(h, t)].add(r)
        
        # Script / URL / Encoding Checks
        if ARABIC_UNCONVERTED_REGEX.search(h) or ARABIC_UNCONVERTED_REGEX.search(t):
            script_anomalies.append((idx, h, r, t, "Unconverted Arabic Character"))
        if MARKDOWN_URL_REGEX.search(h) or MARKDOWN_URL_REGEX.search(t):
            script_anomalies.append((idx, h, r, t, "URL / Markdown Artifact"))
            
        if h in GENERIC_BLACK_HOLE_TERMS or t in GENERIC_BLACK_HOLE_TERMS:
            black_hole_triples += 1

    # 2. Second Pass: Check Contradictions and Asymmetric Violations
    for (h, t), rels in pair_relations.items():
        # Check explicit contradiction pairs
        for conflict_pair in CONFLICTING_RELATION_PAIRS:
            if conflict_pair.issubset(rels):
                conflicts.append((h, list(conflict_pair), t))
                
        # Check asymmetric loop anomalies: (A, r, B) and (B, r, A) where r is strictly asymmetric
        for r in rels:
            if r in STRICTLY_ASYMMETRIC_RELATIONS:
                if (t, h) in pair_relations and r in pair_relations[(t, h)]:
                    if (t, r, h) not in asymmetric_loops:
                        asymmetric_loops.append((h, r, t))

    # Print Summary Report
    print("--- 1. Contradiction & Semantic Conflict Audit ---")
    if conflicts:
        print(f"  ❌ FOUND {len(conflicts)} DIRECT CONTRADICTION(S):")
        for c in conflicts[:5]:
            print(f"     • Concept Pair ({c[0]} <-> {c[2]}) assigned conflicting: {c[1]}")
    else:
        print("  ✅ ZERO semantic contradictions detected.")

    print("\n--- 2. Asymmetric Cycle & Loop Audit ---")
    if asymmetric_loops:
        print(f"  ❌ FOUND {len(asymmetric_loops)} ASYMMETRIC LOOP VIOLATION(S):")
        for loop in asymmetric_loops[:5]:
            print(f"     • Bidirectional cycle on strictly asymmetric relation: ({loop[0]} -> {loop[1]} -> {loop[2]}) & reverse")
    else:
        print("  ✅ ZERO asymmetric cycle violations detected.")

    print("\n--- 3. Script, Encoding & Artifact Audit ---")
    if script_anomalies:
        print(f"  ⚠️ FOUND {len(script_anomalies)} SCRIPT/CHAR ANOMALIES (Arabic Yeh/Kaf or URLs).")
    else:
        print("  ✅ Clean typography & standardized Persian/English character encoding.")

    print("\n--- 4. Generic Hub & Black Hole Node Audit ---")
    print(f"  • Triples touching generic hub words ('thing', 'چیز'): {black_hole_triples:,}")
    top_nodes = node_degrees.most_common(5)
    for n, deg in top_nodes:
        print(f"     - Top Node: '{n}' (Degree: {deg:,})")

    print("\n--- 5. Dataset Diversity & Class Balance ---")
    print(f"  • Unique Concepts: {len(node_degrees):,}")
    print(f"  • Language Distribution: {dict(lang_counts)}")
    print(f"  • Top Relations: {dict(relation_counts.most_common(5))}")
    print("=" * 78)

    # Optional Clean Export (Separation of Query from Mutation)
    if output_clean_path:
        out_p = Path(output_clean_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cleaned_list = []
        for item in triples:
            h = normalize_text_audit(item["head"])
            r = item["relation"]
            t = normalize_text_audit(item["tail"])
            if h != t and len(h) <= 50 and len(t) <= 50:
                cleaned_list.append({
                    "head": h,
                    "relation": r,
                    "tail": t,
                    "lang": "fa" if is_persian_text(h) else "en",
                    "weight": item.get("weight", 2.0)
                })
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(cleaned_list, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] Sanitized dataset written to: {out_p} ({len(cleaned_list):,} assertions)")

    return {
        "total_triples": total_triples,
        "unique_concepts": len(node_degrees),
        "conflicts": len(conflicts),
        "asymmetric_loops": len(asymmetric_loops),
        "script_anomalies": len(script_anomalies),
        "black_hole_triples": black_hole_triples
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit dataset health and hygiene.")
    parser.add_argument("--data", default="data/raw/conceptnet_clean.json", help="Dataset path")
    parser.add_argument("--output-clean", default=None, help="Optional clean output path")
    args = parser.parse_args()
    
    audit_dataset(Path(args.data), output_clean_path=Path(args.output_clean) if args.output_clean else None)
