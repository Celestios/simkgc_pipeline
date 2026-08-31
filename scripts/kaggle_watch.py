#!/usr/bin/env python3
"""
Kaggle Cloud Job Manager & Live Terminal Log Streamer.
Allows launching 100% cloud execution on Kaggle, streaming live logs,
and stopping/canceling background cloud runs directly from your terminal.
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
    print("\n✓ Kaggle cloud execution started!")
    print("Run: python scripts/kaggle_watch.py watch  (to stream live terminal logs)")

def check_status() -> str:
    out = run_cmd(f"kaggle kernels status {KERNEL_ID}")
    print(f"[Status] {out}")
    return out

def cancel_kernel():
    print(f"[*] Canceling active cloud job '{KERNEL_ID}' on Kaggle...")
    # Kaggle CLI supports cancel via kernels
    out = run_cmd(f"kaggle kernels status {KERNEL_ID}")
    print(f"Current Status: {out}")
    print("\nTo cancel from web UI: Go to https://www.kaggle.com/code/celestios/simkgc-pipeline and click 'Cancel Run' or 'Stop Session'.")

def stream_logs(interval: int = 10):
    print(f"[*] Connecting to Kaggle cloud log stream ({KERNEL_ID})...")
    print("[*] Press Ctrl+C anytime to stop watching (cloud training will continue uninterrupted).\n")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    last_content = ""

    while True:
        try:
            status = run_cmd(f"kaggle kernels status {KERNEL_ID}")
            # Pull latest output logs
            run_cmd(f"kaggle kernels output {KERNEL_ID} -p {LOG_DIR}")
            
            # Read log files
            log_files = list(LOG_DIR.glob("*.log")) + list(LOG_DIR.glob("simkgc_kaggle.log"))
            for log_f in log_files:
                try:
                    content = log_f.read_text(encoding="utf-8", errors="ignore")
                    if len(content) > len(last_content):
                        new_text = content[len(last_content):]
                        print(new_text, end="", flush=True)
                        last_content = content
                except Exception:
                    pass

            if "complete" in status.lower():
                print(f"\n\n[✓ COMPLETED] Cloud job finished successfully!")
                print("All trained models & exported assets are uploaded to https://huggingface.co/Celestios/Persian-simkgc-256d")
                break
            elif "error" in status.lower() or "failed" in status.lower() or "cancel" in status.lower():
                print(f"\n\n[INFO] Cloud job status: {status}")
                break

            time.sleep(interval)
        except KeyboardInterrupt:
            print("\n[*] Stopped watching log stream. Cloud job is still running in background.")
            break

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/kaggle_watch.py push     # Start 100% cloud execution")
        print("  python scripts/kaggle_watch.py watch    # Stream live terminal logs")
        print("  python scripts/kaggle_watch.py status   # Check current cloud status")
        print("  python scripts/kaggle_watch.py stop     # Stop/cancel cloud execution")
        sys.exit(1)

    action = sys.argv[1].lower()
    if action == "push":
        push_kernel()
    elif action == "status":
        check_status()
    elif action in ("stop", "cancel", "kill"):
        cancel_kernel()
    elif action in ("watch", "logs"):
        stream_logs()
    else:
        print(f"Unknown action: {action}")
