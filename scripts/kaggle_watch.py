#!/usr/bin/env python3
"""
Kaggle Cloud Job Manager & Live Log Monitor.
Allows launching, monitoring live cloud status, streaming completed logs,
and managing Kaggle GPU runs directly from your terminal.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

KERNEL_ID = "khodaverdishahin/simkgc"
LOG_DIR = Path("kaggle_logs")

if not os.environ.get("KAGGLE_API_TOKEN"):
    token_file = Path.home() / ".kaggle" / "access_token"
    if token_file.exists():
        os.environ["KAGGLE_API_TOKEN"] = token_file.read_text(encoding="utf-8").strip()

def run_cmd(cmd: str) -> str:
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (res.stdout or res.stderr).strip()

def push_kernel():
    print(f"[*] Pushing kernel '{KERNEL_ID}' to Kaggle cloud GPU...")
    out = run_cmd("kaggle kernels push")
    print(out)
    print("\n✓ Cloud GPU training started on Kaggle!")
    print(f"Live web console: https://www.kaggle.com/code/{KERNEL_ID}")
    print("Run: python scripts/kaggle_watch.py watch  (to monitor status in terminal)")

def check_status() -> str:
    out = run_cmd(f"kaggle kernels status {KERNEL_ID}")
    print(f"Status: {out}")
    print(f"Web Console: https://www.kaggle.com/code/{KERNEL_ID}")
    return out

def stream_logs(interval: int = 5):
    print(f"[*] Connected to Kaggle cloud monitor ({KERNEL_ID})")
    print(f"[*] Live Web Console: https://www.kaggle.com/code/{KERNEL_ID}")
    print("[*] Press Ctrl+C anytime to exit (cloud execution will continue in background).\n")
    
    start_time = time.time()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    while True:
        try:
            status = run_cmd(f"kaggle kernels status {KERNEL_ID}")
            elapsed = int(time.time() - start_time)
            mins, secs = divmod(elapsed, 60)
            
            print(f"\r[{mins:02d}:{secs:02d}] Current Cloud Status: {status} ...", end="", flush=True)

            if "complete" in status.lower():
                print(f"\n\n[✓ COMPLETED] Cloud job finished successfully!")
                print(f"All models & exports uploaded to: https://huggingface.co/Celestios/Persian-simkgc-256d")
                
                # Fetch final logs
                run_cmd(f"kaggle kernels output {KERNEL_ID} -p {LOG_DIR}")
                for log_f in LOG_DIR.glob("*.log"):
                    print("\n--- FINAL LOG OUTPUT ---")
                    print(log_f.read_text(encoding="utf-8", errors="ignore"))
                break
            elif "error" in status.lower() or "failed" in status.lower() or "cancel" in status.lower():
                print(f"\n\n[!] Cloud job ended with status: {status}")
                run_cmd(f"kaggle kernels output {KERNEL_ID} -p {LOG_DIR}")
                for log_f in LOG_DIR.glob("*.log"):
                    print("\n--- LOG OUTPUT ---")
                    print(log_f.read_text(encoding="utf-8", errors="ignore"))
                break

            time.sleep(interval)
        except KeyboardInterrupt:
            print("\n\n[*] Stopped watching monitor. Cloud job is still running uninterrupted in background.")
            print(f"Check anytime at: https://www.kaggle.com/code/{KERNEL_ID}")
            break

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/kaggle_watch.py push     # Launch cloud execution")
        print("  python scripts/kaggle_watch.py watch    # Monitor status & fetch logs")
        print("  python scripts/kaggle_watch.py status   # Check current cloud status")
        sys.exit(1)

    action = sys.argv[1].lower()
    if action == "push":
        push_kernel()
    elif action == "status":
        check_status()
    elif action in ("watch", "logs"):
        stream_logs()
    else:
        print(f"Unknown action: {action}")
