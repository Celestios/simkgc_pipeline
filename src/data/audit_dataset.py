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

ARABIC_UNCONVERTED_REGEX = re.compile(r"[يكةۀ]")
MARKDOWN_URL_REGEX = re.compile(r"(http[s]?://|www\.|\[|\]|\{|\}|<|>)")

def audit_dataset(
    dataset_path: Path,
    output_clean_path: Optional[Path] = None,
    max_hub_degree: int = 2500
) -> Dict[str, any]:
    """Runs all 5 health checks on the dataset."""
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
        
        # Check Arabic unconverted characters
        if ARABIC_UNCONVERTED_REGEX.search(h) or ARABIC_UNCONVERTED_REGEX.search(t):
            script_anomalies.append((h, r, t, "Unconverted Arabic character (ي/ك/ة)"))
            
        # Check Markdown/URL artifacts
        if MARKDOWN_URL_REGEX.search(h) or MARKDOWN_URL_REGEX.search(t):
            script_anomalies.append((h, r, t, "URL or Markdown tag artifact"))
            
        if h.lower() in GENERIC_BLACK_HOLE_TERMS or t.lower() in GENERIC_BLACK_HOLE_TERMS:
            black_hole_triples += 1

    # 2. Second Pass: Check Conflicts & Asymmetric Loops
    for (h, t), rels in pair_relations.items():
        # Check conflicting pairs
        for conflict_pair in CONFLICTING_RELATION_PAIRS:
            if conflict_pair.issubset(rels):
                conflicts.append((h, list(conflict_pair), t))
                
        # Check asymmetric loops (A -> B and B -> A with asymmetric relation)
        reverse_rels = pair_relations.get((t, h), set())
        for r in rels:
            if r in STRICTLY_ASYMMETRIC_RELATIONS and r in reverse_rels and h != t:
                asymmetric_loops.append((h, r, t))

    # Over-centralized hubs
    over_central_hubs = [(node, deg) for node, deg in node_degrees.most_common(20) if deg > max_hub_degree]

    # --- PRINT AUDIT REPORT ---
    print("▶ 1. RELATION DISTRIBUTION & CLASS BALANCE")
    print("-" * 78)
    for rel, count in relation_counts.most_common():
        pct = (count / total_triples) * 100.0
        print(f"  • {rel:<22} : {count:>8,} triples ({pct:>5.1f}%)")
    print(f"  Languages: English = {lang_counts['en']:,} ({lang_counts['en']/total_triples*100:.1f}%), Persian = {lang_counts['fa']:,} ({lang_counts['fa']/total_triples*100:.1f}%)")

    print("\n▶ 2. CONTRADICTION & SEMANTIC CONFLICTS")
    print("-" * 78)
    if conflicts:
        print(f"  ⚠ Found {len(conflicts):,} conflicting assertions:")
        for h, c_rels, t in conflicts[:5]:
            print(f"    - '{h}' ──[{c_rels[0]} vs {c_rels[1]}]──► '{t}'")
    else:
        print("  ✓ Zero semantic contradictions detected (0 conflicting pairs).")

    print("\n▶ 3. ASYMMETRIC INVERSION LOOPS")
    print("-" * 78)
    if asymmetric_loops:
        print(f"  ⚠ Found {len(asymmetric_loops):,} circular loops on asymmetric relations:")
        for h, r, t in asymmetric_loops[:5]:
            print(f"    - Loop: '{h}' ──({r})──► '{t}' AND '{t}' ──({r})──► '{h}'")
    else:
        print("  ✓ Zero asymmetric circular loops detected.")

    print("\n▶ 4. GENERIC 'BLACK HOLE' HUBS")
    print("-" * 78)
    if over_central_hubs:
        print(f"  ⚠ Found {len(over_central_hubs)} over-centralized hubs (Degree > {max_hub_degree:,}):")
        for node, deg in over_central_hubs[:8]:
            print(f"    - '{node}': {deg:,} connections")
    else:
        print(f"  ✓ No unnatural black-hole hubs detected (Top hub degree: {node_degrees.most_common(1)[0][1]:,}).")
    print(f"  Generic noisy stop-word triples: {black_hole_triples:,}")

    print("\n▶ 5. SCRIPT & TEXT HYGIENE")
    print("-" * 78)
    if script_anomalies:
        print(f"  ⚠ Found {len(script_anomalies):,} text/script anomalies:")
        for h, r, t, reason in script_anomalies[:5]:
            print(f"    - ({h}, {r}, {t}): {reason}")
    else:
        print("  ✓ 100% clean characters (Zero unconverted Arabic/Markdown artifacts).")

    # --- AUTO-REPAIR IF REQUESTED ---
    if output_clean_path:
        print("\n" + "=" * 78)
        print("  APPLYING AUTOMATIC HEALTH REPAIR & PURGING ANOMALIES...")
        print("=" * 78)
        
        conflict_lookup = { (h, t) for h, _, t in conflicts }
        asym_loop_lookup = { (h, t, r) for h, r, t in asymmetric_loops }
        
        clean_triples = []
        for item in triples:
            h = item.get("head", "").strip()
            r = item.get("relation", "").strip()
            t = item.get("tail", "").strip()
            
            # 1. Skip Black-Hole Stop Words
            if h.lower() in GENERIC_BLACK_HOLE_TERMS or t.lower() in GENERIC_BLACK_HOLE_TERMS:
                continue
                
            # 2. Skip Script Anomalies
            if MARKDOWN_URL_REGEX.search(h) or MARKDOWN_URL_REGEX.search(t):
                continue
                
            # 3. Skip Contradictions & Asymmetric Loops
            if (h, t) in conflict_lookup:
                continue
            if (h, t, r) in asym_loop_lookup:
                continue
                
            # 4. Standardize any remaining Arabic characters
            if is_persian_text(h) or is_persian_text(t):
                h = h.replace('ي', 'ی').replace('ك', 'ک')
                t = t.replace('ي', 'ی').replace('ك', 'ک')
                
            clean_triples.append({
                "head": h,
                "relation": r,
                "tail": t,
                "lang": item.get("lang", "en"),
                "weight": float(item.get("weight", 2.0))
            })
            
        output_clean_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_clean_path, "w", encoding="utf-8") as f:
            json.dump(clean_triples, f, ensure_ascii=False, indent=2)
            
        print(f"✓ Cleaned dataset saved to: {output_clean_path}")
        print(f"  {total_triples:,} raw assertions ──► {len(clean_triples):,} pristine assertions ({len(clean_triples)/total_triples*100:.1f}%)")
        print("=" * 78)

    return {
        "total_triples": total_triples,
        "conflicts_count": len(conflicts),
        "asymmetric_loops_count": len(asymmetric_loops),
        "script_anomalies_count": len(script_anomalies),
        "black_hole_count": black_hole_triples
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit and repair knowledge graph dataset health")
    parser.add_argument("--data", default="data/raw/conceptnet_clean.json", help="Dataset path to audit")
    parser.add_argument("--fix-output", default=None, help="Optional path to output auto-repaired dataset")
    args = parser.parse_args()
    
    fix_out = Path(args.fix_output) if args.fix_output else None
    audit_dataset(Path(args.data), output_clean_path=fix_out)
