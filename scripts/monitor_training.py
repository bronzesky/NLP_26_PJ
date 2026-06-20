"""
scripts/monitor_training.py

Monitor training log every N seconds.
Detects NaN loss, dead process, prints latest metrics.
Read-only — makes no changes to anything.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def is_process_alive(pid_file: str) -> bool:
    try:
        pid = int(Path(pid_file).read_text().strip())
        os.kill(pid, 0)
        return True
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        return False


def tail_jsonl(log_file: str, n: int = 20) -> list[dict]:
    try:
        lines = Path(log_file).read_text(encoding="utf-8").strip().splitlines()
        records = []
        for line in lines[-n:]:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return records
    except FileNotFoundError:
        return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str,
                        default="outputs/han_deberta_large_full/training_log.jsonl")
    parser.add_argument("--pid_file", type=str,
                        default="logs/han_deberta_full.pid")
    parser.add_argument("--interval", type=int, default=300,
                        help="Check interval in seconds")
    args = parser.parse_args()

    print(f"Monitoring {args.log_file} every {args.interval}s")
    print("Press Ctrl+C to stop\n")

    last_step = -1
    last_epoch = -1

    while True:
        alive = is_process_alive(args.pid_file)
        records = tail_jsonl(args.log_file)

        if not records:
            print(f"[{time.strftime('%H:%M:%S')}] No log entries yet. Process alive: {alive}")
        else:
            latest = records[-1]
            step = latest.get("step", latest.get("epoch", "?"))
            loss = latest.get("avg_loss", "?")
            dev_acc = latest.get("dev_accuracy", None)
            epoch = latest.get("epoch", "?")

            # NaN detection
            nan_alert = ""
            if isinstance(loss, float) and (loss != loss or loss > 100):
                nan_alert = " *** ALERT: NaN/Inf LOSS DETECTED ***"

            status_line = (
                f"[{time.strftime('%H:%M:%S')}] "
                f"epoch={epoch} step={step} loss={loss}"
            )
            if dev_acc is not None:
                status_line += f" dev_acc={dev_acc:.4f}"
            status_line += f" | process={'ALIVE' if alive else 'DEAD'}"
            status_line += nan_alert
            print(status_line)

            # Epoch summary (only print new epochs)
            for rec in records:
                if "dev_accuracy" in rec and rec.get("epoch", -1) != last_epoch:
                    last_epoch = rec["epoch"]
                    print(f"  -> Epoch {rec['epoch']} complete: "
                          f"loss={rec.get('avg_loss', '?')} "
                          f"dev_acc={rec['dev_accuracy']:.4f}")

        if not alive and records:
            print("\nProcess has ended. Training complete or crashed.")
            break

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
