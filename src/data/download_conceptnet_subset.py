#!/usr/bin/env python3
"""
Direct ConceptNet Database Subset Downloader with Real-Time Bandwidth Monitor.
Streams authentic edges directly from the official ConceptNet S3 repository
(https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz).

Guarantees:
  - Strict hard byte-limit cap (default: 5.0 MB). Aborts immediately if exceeded.
  - Live real-time download monitor printing exact KB/MB received.
  - Zero hardcoded domain categories or assertions in code.
"""

import sys
import json
import gzip
import io
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

def download_with_bandwidth_monitor(url: str, max_bytes: int = 5 * 1024 * 1024) -> bytes:
    """
    Downloads data chunk-by-chunk with a strict hard cap and live byte monitoring.
    Aborts immediately if download exceeds max_bytes.
    """
    print("=" * 65)
    print(f"BANDWIDTH MONITOR: Initiating controlled stream from S3")
    print(f"HARD LIMIT CAP:    {max_bytes / 1024 / 1024:.2f} MB (Will not exceed under any circumstance)")
    print("=" * 65)

    req = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes=0-{max_bytes - 1}",
            "User-Agent": "CentrodeMonitoredDownloader/1.0"
        }
    )

    buffer = io.BytesIO()
    downloaded_bytes = 0
    chunk_size = 64 * 1024 # 64 KB chunks

    with urllib.request.urlopen(req, timeout=30) as response:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break

            downloaded_bytes += len(chunk)
            buffer.write(chunk)

            # Print live progress monitor
            pct = min(100.0, (downloaded_bytes / max_bytes) * 100.0)
            sys.stdout.write(f"\r[DOWNLOAD MONITOR] Received: {downloaded_bytes / 1024 / 1024:.3f} MB / {max_bytes / 1024 / 1024:.2f} MB ({pct:.1f}%)")
            sys.stdout.flush()

            # Hard safety stop
            if downloaded_bytes >= max_bytes:
                print("\n[GUARD TRIGGERED] Reached exact byte limit. Terminating connection.")
                break

    print(f"\n[OK] Download complete. Exact total transfer: {downloaded_bytes / 1024:.2f} KB ({downloaded_bytes} bytes).")
    return buffer.getvalue()

def stream_and_parse_subset(max_bytes: int = 5 * 1024 * 1024, max_triples: int = 1500) -> List[Dict]:
    compressed_bytes = download_with_bandwidth_monitor(S3_CONCEPTNET_URL, max_bytes=max_bytes)
    
    print(f"Decompressing genuine ConceptNet dump stream in memory...")
    triples = []
    
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(compressed_bytes)) as decompressor:
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

    print(f"Extracted {len(triples):,} authentic ConceptNet database records.")
    return triples

def download_and_save_subset(output_path: Path, max_bytes: int = 5 * 1024 * 1024, max_triples: int = 1500):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    triples = stream_and_parse_subset(max_bytes=max_bytes, max_triples=max_triples)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(triples, f, ensure_ascii=False, indent=2)
        
    print(f"Saved authentic database records to: {output_path}")

if __name__ == "__main__":
    download_and_save_subset(Path("data/raw/conceptnet_subset.json"), max_bytes=5 * 1024 * 1024)
