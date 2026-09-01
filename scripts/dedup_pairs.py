#!/usr/bin/env python3
"""Deduplicate pair files and remove malformed lines."""
import os, re
from pathlib import Path

pair_dir = Path("data/synthetic_pairs")
files = sorted(pair_dir.glob("*.txt"))

total_removed = 0

for fpath in files:
    lines = fpath.read_text(encoding="utf-8").splitlines()
    clean = []
    seen = set()
    removed = 0

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split("\t")
        if len(parts) < 2:
            removed += 1
            continue
        head, tail = parts[0].strip(), parts[1].strip()
        if not head or not tail:
            removed += 1
            continue
        key = (head, tail)
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        clean.append(f"{head}\t{tail}")

    fpath.write_text("\n".join(clean) + "\n", encoding="utf-8")
    total_removed += removed
    print(f"{fpath.name}: {len(lines)} -> {len(clean)} (removed {removed})")

print(f"\nTotal removed: {total_removed}")
