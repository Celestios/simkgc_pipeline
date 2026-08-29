#!/usr/bin/env python3
"""
Direct ConceptNet Database Downloader (Full & Subset Modes).
Streams authentic assertions directly from the official ConceptNet S3 repository
(https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz).

Supports:
  - Full Mode (--full): Downloads and parses the entire ~450 MB compressed ConceptNet database
    streaming all Persian (fa) and English (en) assertions.
  - Subset Mode (--cap-mb): Controlled download with real-time bandwidth meter and hard cap.
"""

import sys
import json
import gzip
import io
import argparse
import urllib.request
from pathlib import Path
from typing import List, Dict

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

S3_CONCEPTNET_URL = "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"

def clean_concept_uri(uri: str) -> str:
    parts = uri.strip("/").split("/")
    return parts[2].replace("_", " ") if len(parts) >= 3 else ""

def clean_relation_label(rel_uri: str) -> str:
    return rel_uri.strip("/").split("/")[-1]

def download_and_extract_full(output_path: Path):
    """
    Streams and extracts the complete ConceptNet database for Persian and English.
    Designed for fast cloud/Kaggle environments.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_gz_path = output_path.parent / "conceptnet_dump.csv.gz"
    
    print("=" * 65)
    print("CONCEPTNET FULL DATABASE DOWNLOADER (S3)")
    print(f"URL:    {S3_CONCEPTNET_URL}")
    print(f"TARGET: {output_path}")
    print("=" * 65)

    # 1. Download full gzipped archive with progress
    if not temp_gz_path.exists():
        req = urllib.request.Request(S3_CONCEPTNET_URL, headers={"User-Agent": "CentrodeFullDownloader/1.0"})
        downloaded = 0
        chunk_size = 256 * 1024
        
        with urllib.request.urlopen(req, timeout=60) as response, open(temp_gz_path, "wb") as out_f:
            total_bytes = int(response.headers.get("Content-Length", 450 * 1024 * 1024))
            print(f"Downloading full compressed archive ({total_bytes / 1024 / 1024:.1f} MB)...")
            
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                downloaded += len(chunk)
                out_f.write(chunk)
                
                pct = min(100.0, (downloaded / total_bytes) * 100.0)
                sys.stdout.write(f"\r[FULL DOWNLOAD] {downloaded / 1024 / 1024:.1f} MB / {total_bytes / 1024 / 1024:.1f} MB ({pct:.1f}%)")
                sys.stdout.flush()
        print(f"\n[OK] Archive downloaded: {temp_gz_path} ({downloaded / 1024 / 1024:.1f} MB)")
    else:
        print(f"Using cached dump archive: {temp_gz_path}")

    # 2. Stream parse line by line
    print("Extracting all authentic Persian (fa) and English (en) assertions...")
    triples = []
    seen = set()
    line_count = 0

    with gzip.open(temp_gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            line_count += 1
            if line_count % 1_000_000 == 0:
                print(f"Processed {line_count / 1_000_000:.0f}M lines -> {len(triples):,} valid FA/EN assertions extracted...")
                
            fields = line.strip().split("\t")
            if len(fields) < 4:
                continue
                
            rel_uri, head_uri, tail_uri = fields[1], fields[2], fields[3]
            
            lang = "fa" if head_uri.startswith("/c/fa/") else ("en" if head_uri.startswith("/c/en/") else None)
            if not lang:
                continue
                
            head = clean_concept_uri(head_uri)
            tail = clean_concept_uri(tail_uri)
            relation = clean_relation_label(rel_uri)
            
            if head and tail and head.lower() != tail.lower():
                key = (head, relation, tail)
                if key not in seen:
                    seen.add(key)
                    triples.append({
                        "head": head,
                        "relation": relation,
                        "tail": tail,
                        "lang": lang,
                        "raw_uri": fields[0] if len(fields) > 0 else ""
                    })

    # 3. Save JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(triples, f, ensure_ascii=False, indent=2)

    # Cleanup temporary archive to save disk space
    if temp_gz_path.exists():
        temp_gz_path.unlink()

    print("\n" + "=" * 65)
    print(f"[COMPLETE] Extracted {len(triples):,} authentic assertions across {line_count:,} lines.")
    print(f"Saved to: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print("=" * 65)

def download_and_extract_subset(output_path: Path, max_bytes: int = 5 * 1024 * 1024, max_triples: int = 1500):
    """Downloads a small slice for local testing."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 65)
    print(f"BANDWIDTH MONITOR: Initiating controlled subset stream from S3")
    print(f"HARD LIMIT CAP:    {max_bytes / 1024 / 1024:.2f} MB")
    print("=" * 65)

    req = urllib.request.Request(
        S3_CONCEPTNET_URL,
        headers={"Range": f"bytes=0-{max_bytes - 1}", "User-Agent": "CentrodeSubsetDownloader/1.0"}
    )
    buffer = io.BytesIO()
    downloaded = 0
    chunk_size = 64 * 1024

    with urllib.request.urlopen(req, timeout=30) as response:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            downloaded += len(chunk)
            buffer.write(chunk)
            pct = min(100.0, (downloaded / max_bytes) * 100.0)
            sys.stdout.write(f"\r[SUBSET MONITOR] {downloaded / 1024 / 1024:.2f} MB / {max_bytes / 1024 / 1024:.2f} MB ({pct:.1f}%)")
            sys.stdout.flush()
            if downloaded >= max_bytes:
                break

    print(f"\nDecompressing subset in memory...")
    triples = []
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(buffer.getvalue())) as decompressor:
            for line_bytes in decompressor:
                line = line_bytes.decode("utf-8", errors="ignore").strip()
                fields = line.split("\t")
                if len(fields) < 4:
                    continue
                rel_uri, head_uri, tail_uri = fields[1], fields[2], fields[3]
                lang = "fa" if head_uri.startswith("/c/fa/") else ("en" if head_uri.startswith("/c/en/") else None)
                if not lang:
                    continue
                head = clean_concept_uri(head_uri)
                tail = clean_concept_uri(tail_uri)
                relation = clean_relation_label(rel_uri)
                if head and tail and head.lower() != tail.lower():
                    triples.append({
                        "head": head,
                        "relation": relation,
                        "tail": tail,
                        "lang": lang,
                        "raw_uri": fields[0] if len(fields) > 0 else ""
                    })
                    if len(triples) >= max_triples:
                        break
    except (gzip.BadGzipFile, EOFError):
        pass

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(triples, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(triples):,} records to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ConceptNet Downloader")
    parser.add_argument("--full", action="store_true", help="Download complete ~450MB database (Kaggle/Cloud)")
    parser.add_argument("--output", type=str, default="data/raw/conceptnet_subset.json", help="Output path")
    parser.add_argument("--cap-mb", type=float, default=5.0, help="Max MB cap for subset mode")
    args = parser.parse_args()

    out_file = Path(args.output)
    if args.full:
        download_and_extract_full(out_file)
    else:
        download_and_extract_subset(out_file, max_bytes=int(args.cap_mb * 1024 * 1024))
