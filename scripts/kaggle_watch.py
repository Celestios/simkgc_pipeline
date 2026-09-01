#!/usr/bin/env python3
"""
Kaggle Cloud Job Manager & Live Terminal Log Streamer.
Handles ephemeral authentication injection for private Kaggle cloud executions,
streams real-time live terminal logs directly from Kaggle's cloud API.
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

def get_kaggle_api():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        return api
    except Exception as e:
        print(f"[ERROR] Failed to authenticate Kaggle API: {e}")
        return None

def run_cmd(cmd: str) -> str:
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (res.stdout or res.stderr).strip()

def get_local_hf_token() -> str:
    token_file = Path.home() / ".huggingface" / "token"
    if token_file.exists():
        t = token_file.read_text(encoding="utf-8").strip()
        if t:
            return t
    cache_token = Path.home() / ".cache" / "huggingface" / "token"
    if cache_token.exists():
        t = cache_token.read_text(encoding="utf-8").strip()
        if t:
            return t
    return os.environ.get("HF_TOKEN", "").strip()

def push_kernel():
    hf_token = get_local_hf_token()
    kaggle_script = Path("simkgc_kaggle.py")
    orig_code = kaggle_script.read_text(encoding="utf-8")

    try:
        if hf_token:
            injected_code = orig_code.replace(
                'HF_TOKEN = os.environ.get("HF_TOKEN")',
                f'HF_TOKEN = os.environ.get("HF_TOKEN") or "{hf_token}"'
            )
            kaggle_script.write_text(injected_code, encoding="utf-8")
            print(f"[*] Authenticated private payload with Hugging Face Token (Prefix: {hf_token[:8]}...)")

        print(f"[*] Pushing new kernel version for '{KERNEL_ID}' to Kaggle cloud GPU...")
        out = run_cmd("kaggle kernels push -p .")
        print(out)
        print("\n[OK] Cloud GPU (T4 x2) training started on Kaggle!")
        print(f"Live web console: https://www.kaggle.com/code/{KERNEL_ID}")
        print("Run: python scripts/kaggle_watch.py watch  (to stream live terminal logs)")
    finally:
        kaggle_script.write_text(orig_code, encoding="utf-8")

def check_status() -> str:
    out = run_cmd(f"kaggle kernels status {KERNEL_ID}")
    print(f"Status: {out}")
    print(f"Web Console: https://www.kaggle.com/code/{KERNEL_ID}")
    return out

def stream_logs(interval: int = 5):
    api = get_kaggle_api()
    if not api:
        print("[ERROR] Kaggle API client could not be initialized.")
        return

    owner_slug, kernel_slug = KERNEL_ID.split("/")
    print(f"[*] Connecting to Kaggle cloud log stream for '{KERNEL_ID}'...")
    print(f"[*] Live Web Console: https://www.kaggle.com/code/{KERNEL_ID}")
    print("[*] Press Ctrl+C anytime to stop watching (cloud execution will continue).\n")

    last_log_len = 0
    start_time = time.time()

    while True:
        try:
            status_res = api.kernels_status(KERNEL_ID)
            status = str(status_res.status if hasattr(status_res, 'status') else status_res)
            
            try:
                LOG_DIR.mkdir(parents=True, exist_ok=True)
                api.kernels_output(KERNEL_ID, str(LOG_DIR), force=True, quiet=True)
                
                log_file = LOG_DIR / f"{kernel_slug}.log"
                if log_file.exists():
                    current_log = log_file.read_text(encoding="utf-8", errors="ignore")
                    if len(current_log) > last_log_len:
                        new_content = current_log[last_log_len:]
                        print(new_content, end="", flush=True)
                        last_log_len = len(current_log)
            except Exception:
                pass

            if "complete" in status.lower():
                print(f"\n\n[OK] Cloud job finished successfully!")
                print(f"All models & exports uploaded to: https://huggingface.co/Celestios/Persian-simkgc-256d")
                break
            elif "error" in status.lower() or "failed" in status.lower() or "cancel" in status.lower():
                print(f"\n\n[!] Cloud job ended with status: {status}")
                if hasattr(status_res, 'failure_message') and status_res.failure_message:
                    print(f"Failure Reason: {status_res.failure_message}")
                break

            time.sleep(interval)
        except KeyboardInterrupt:
            print("\n\n[*] Stopped watching log stream. Cloud job is still running in background.")
            print(f"Check anytime at: https://www.kaggle.com/code/{KERNEL_ID}")
            break

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/kaggle_watch.py push     # Launch cloud execution on GPU T4 x2")
        print("  python scripts/kaggle_watch.py watch    # Stream live terminal logs")
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
