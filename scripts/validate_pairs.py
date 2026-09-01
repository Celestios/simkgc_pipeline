#!/usr/bin/env python3
"""Validate synthetic pair files for character contamination and duplicates."""
import os, re, sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

pair_dir = Path("data/synthetic_pairs")
files = sorted(pair_dir.glob("*.txt"))

total_dupes = 0
total_char_issues = 0
total_no_tab = 0

for fpath in files:
    fname = fpath.name
    is_fa = "_fa" in fname

    lines = [l.strip() for l in fpath.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]

    dupes = []
    seen = set()
    char_issues = []
    no_tab = 0

    for i, line in enumerate(lines, 1):
        parts = line.split("\t")
        if len(parts) < 2:
            no_tab += 1
            continue
        head, tail = parts[0].strip(), parts[1].strip()
        key = (head, tail)

        if key in seen:
            dupes.append((i, head[:40], tail[:40]))
        seen.add(key)

        has_fa = bool(re.search(r"[\u0600-\u06FF]", head + tail))

        if is_fa and not has_fa:
            char_issues.append((i, "MISSING_PERSIAN", head[:40], tail[:40]))
        elif not is_fa and has_fa:
            char_issues.append((i, "PERSIAN_IN_EN", head[:40], tail[:40]))

    total_dupes += len(dupes)
    total_char_issues += len(char_issues)
    total_no_tab += no_tab

    status = "OK" if not dupes and not char_issues and not no_tab else "ISSUES"
    print(f"{status} | {fname}: {len(lines)} pairs, {len(dupes)} dupes, {len(char_issues)} char issues, {no_tab} no-tab")
    for d in dupes[:5]:
        print(f"  DUPE L{d[0]}: '{d[1]}' -> '{d[2]}'")
    for c in char_issues[:5]:
        print(f"  CHAR L{c[0]}: {c[1]} | '{c[2]}' -> '{c[3]}'")

print(f"\nTOTAL: {total_dupes} dupes, {total_char_issues} char issues, {total_no_tab} no-tab across {len(files)} files")
