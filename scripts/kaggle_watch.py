#!/usr/bin/env python3
"""
Kaggle Cloud Job Manager & Live Log Streamer.
Allows pushing, monitoring status, streaming live terminal logs,
and downloading output assets from Kaggle cloud execution.
"""

import sys
import time
import subprocess
from pathlib import Path

KERNEL_ID = "celestios/simkgc-pipeline"
LOG_DIR = Path("kaggle_logs")

def run_cmd(cmd: str) -> str:
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return res.stdout.strip()

def push_kernel():
    print(f"[*] Pushing kernel '{KERNEL_ID}' to Kaggle cloud...")
    out = run_cmd("kaggle kernels push")
    print(out)

def check_status() -> str:
    out = run_cmd(f"kaggle kernels status {KERNEL_ID}")
    print(f"[Status] {out}")
    return out

def stream_logs(interval: int = 15):
    print(f"[*] Streaming live logs from Kaggle cloud ({KERNEL_ID})...")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    last_content = ""

    while True:
        status = run_cmd(f"kaggle kernels status {KERNEL_ID}")
        print(f"\r[{time.strftime('%H:%M:%S')}] {status}", end="", flush=True)

        # Pull latest output logs
        run_cmd(f"kaggle kernels output {KERNEL_ID} -p {LOG_DIR}")
        
        # Check log files
        log_files = list(LOG_DIR.glob("*.log")) + list(LOG_DIR.glob("simkgc_kaggle.log"))
        for log_f in log_files:
            try:
                content = log_f.read_text(encoding="utf-8", errors="ignore")
                if len(content) > len(last_content):
                    new_text = content[len(last_content):]
                    print("\n" + new_text, end="", flush=True)
                    last_content = content
            except Exception:
                pass

        if "complete" in status.lower() or "error" in status.lower() or "failed" in status.lower():
            print(f"\n[*] Kernel finished with status: {status}")
            break

        time.sleep(interval)

def download_outputs(dest: str = "exports"):
    print(f"[*] Downloading trained assets from Kaggle cloud to '{dest}'...")
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)
    out = run_cmd(f"kaggle kernels output {KERNEL_ID} -p {dest_path}")
    print(out)
    print(f"✓ Downloaded outputs to: {dest_path.resolve()}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/kaggle_watch.py push     # Push & launch cloud training")
        print("  python scripts/kaggle_watch.py status   # Check current status")
        print("  python scripts/kaggle_watch.py watch    # Stream live terminal logs")
        print("  python scripts/kaggle_watch.py pull     # Download output assets")
        sys.exit(1)

    action = sys.argv[1].lower()
    if action == "push":
        push_kernel()
    elif action == "status":
        check_status()
    elif action in ("watch", "logs"):
        stream_logs()
    elif action in ("pull", "download"):
        download_outputs()
    else:
        print(f"Unknown action: {action}")
