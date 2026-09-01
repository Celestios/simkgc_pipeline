#!/usr/bin/env python3
"""
Synthetic Pairs to Knowledge Graph Triplet Compiler.
Converts modular per-relation pair files (TSV / CSV / Text) into validated,
deduplicated JSON knowledge graph triplets.
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple

def is_persian_text(text: str) -> bool:
    """Detects if string contains Persian/Arabic characters."""
    return bool(re.search(r'[\u0600-\u06FF]', text))

def normalize_concept(text: str, is_fa: bool) -> str:
    """Standardizes Persian typography, ZWNJ, and English casing."""
    text = text.strip().strip('"\'')
    if is_fa:
        # Normalize Arabic characters to Persian
        text = text.replace('\u064A', '\u06CC').replace('\u0649', '\u06CC')
        text = text.replace('\u0643', '\u06A9')
        text = re.sub(r'[\u200B-\u200D\uFEFF]', '\u200c', text)
        text = re.sub(r'\s+', ' ', text)
    else:
        text = re.sub(r'\s+', ' ', text).strip()
    return text.strip()

def compile_pairs_to_triplets(
    input_dir: str = "data/synthetic_pairs",
    output_json: str = "data/synthetic/all_triplets_deduped.json",
    merge_existing: bool = True
) -> List[Dict]:
    input_path = Path(input_dir)
    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_triplets: List[Dict] = []
    seen_keys: Set[Tuple[str, str, str]] = set()

    if merge_existing and out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                existing_triplets = json.load(f)
            for t in existing_triplets:
                h = t.get("head", "").strip()
                r = t.get("relation", "").strip()
                tail = t.get("tail", "").strip()
                if h and r and tail:
                    seen_keys.add((h, r, tail))
            print(f"[Compiler] Loaded {len(existing_triplets):,} existing triplets from {out_path}.")
        except Exception as e:
            print(f"[Compiler] Warning loading existing file: {e}")

    new_triplets: List[Dict] = []
    
    if not input_path.exists():
        print(f"[Compiler] Input directory {input_path} not found. Creating empty directory.")
        input_path.mkdir(parents=True, exist_ok=True)
        return existing_triplets

    # Scan all .txt, .tsv, .csv files in synthetic_pairs/
    pair_files = list(input_path.glob("*.txt")) + list(input_path.glob("*.tsv")) + list(input_path.glob("*.csv"))
    print(f"\n[Compiler] Scanning {len(pair_files)} pair files in {input_path}...")

    for pf in sorted(pair_files):
        # Extract relation name from filename (e.g. DependsOn_fa.txt -> DependsOn)
        fname = pf.stem
        # Match standard relation names
        match = re.match(r"([A-Za-z]+)(?:_.*)?", fname)
        if not match:
            print(f"[-] Skipping unrecognized filename format: {pf.name}")
            continue
            
        relation = match.group(1)
        file_count = 0

        with open(pf, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Support tab, pipe, or comma delimiters
                parts = []
                if "\t" in line:
                    parts = [p.strip() for p in line.split("\t") if p.strip()]
                elif "|" in line:
                    parts = [p.strip() for p in line.split("|") if p.strip()]
                elif "," in line:
                    parts = [p.strip() for p in line.split(",") if p.strip()]

                if len(parts) < 2:
                    continue

                raw_head, raw_tail = parts[0], parts[1]
                is_fa = is_persian_text(raw_head) or is_persian_text(raw_tail)
                lang = "fa" if is_fa else "en"

                head = normalize_concept(raw_head, is_fa)
                tail = normalize_concept(raw_tail, is_fa)

                if not head or not tail or head == tail:
                    continue

                # Filter out overly long sentences (> 60 chars)
                if len(head) > 60 or len(tail) > 60:
                    continue

                key = (head, relation, tail)
                if key not in seen_keys:
                    seen_keys.add(key)
                    item = {
                        "head": head,
                        "relation": relation,
                        "tail": tail,
                        "lang": lang,
                        "weight": 2.0
                    }
                    new_triplets.append(item)
                    file_count += 1

        print(f"  • {pf.name:<32} -> {file_count:,} new triples parsed (Relation: {relation})")

    all_triplets = existing_triplets + new_triplets
    print("\n" + "=" * 65)
    print(f"[Compiler] Total Unique Triples Compiled: {len(all_triplets):,} ({len(new_triplets):,} newly added)")
    print(f"[Compiler] Writing output to: {out_path}...")
    print("=" * 65)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_triplets, f, ensure_ascii=False, indent=2)

    print("✓ Compilation successfully completed!")
    return all_triplets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile modular relation pair files into JSON triplets")
    parser.add_argument("--input-dir", default="data/synthetic_pairs", help="Directory with pair files")
    parser.add_argument("--output", default="data/synthetic/all_triplets_deduped.json", help="Target JSON path")
    parser.add_argument("--no-merge", action="store_true", help="Overwrite existing output file without merging")
    args = parser.parse_args()

    compile_pairs_to_triplets(
        input_dir=args.input_dir,
        output_json=args.output,
        merge_existing=not args.no_merge
    )
